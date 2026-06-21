# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Optane tier manager for KVCacheManager integration.

This module provides OptaneManager which orchestrates block movement
across GPU, CPU, and Optane memory tiers during request lifecycle.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from vllm.config.optane import OptaneConfig
from vllm.core.eviction_policy_optane import (
    EvictionPolicy,
    EvictionTier,
    create_eviction_policy,
)
from vllm.distributed.kv_events_optane import OptaneEventQueue
from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.block_pool_optane import OptaneBlockPool, OptaneBlockMetadata
from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)


class BlockTierPlacement(Enum):
    """Enum for block placement decisions."""

    GPU = "gpu"  # Keep in GPU memory
    CPU = "cpu"  # Move to CPU memory
    OPTANE = "optane"  # Move to Optane memory
    SSD = "ssd"  # Move to SSD storage
    REMOTE = "remote"  # Move to remote storage


@dataclass
class TierStats:
    """Statistics for a memory tier."""

    num_blocks: int = 0
    num_bytes: int = 0
    num_evictions: int = 0
    num_promotions: int = 0
    total_store_time_us: float = 0.0
    total_retrieve_time_us: float = 0.0


@dataclass
class OptaneManagerStats:
    """Statistics tracked by OptaneManager."""

    gpu_tier: TierStats = field(default_factory=TierStats)
    cpu_tier: TierStats = field(default_factory=TierStats)
    optane_tier: TierStats = field(default_factory=TierStats)
    ssd_tier: TierStats = field(default_factory=TierStats)
    remote_tier: TierStats = field(default_factory=TierStats)

    total_blocks_allocated: int = 0
    total_blocks_freed: int = 0
    total_tier_crossings: int = 0


class OptaneManager:
    """
    Orchestrates KV cache block placement across memory tiers.

    The OptaneManager coordinates with KVCacheManager to:
    1. Intercept block allocation and decide optimal tier placement
    2. Handle block migration between tiers during eviction
    3. Track block state transitions across tiers
    4. Emit KV-Events for observability
    5. Collect per-tier metrics

    Architecture:
        KVCacheManager
             |
             v
        OptaneManager (this class)
             |
        +----+----+
        |         |
        v         v
    BlockPool  OptaneBlockPool
    (GPU)      (Optane/CPU)
    """

    def __init__(
        self,
        optane_config: OptaneConfig,
        block_pool: BlockPool,
        optane_block_pool: Optional[OptaneBlockPool] = None,
        enable_metrics: bool = True,
    ):
        """
        Initialize OptaneManager.

        Args:
            optane_config: Configuration for Optane integration
            block_pool: GPU BlockPool instance
            optane_block_pool: Optional OptaneBlockPool for Optane tier.
                              If None, only GPU tier will be used.
            enable_metrics: Whether to track tier statistics
        """
        self.optane_config = optane_config
        self.block_pool = block_pool
        self.optane_block_pool = optane_block_pool
        self.enable_metrics = enable_metrics

        # Initialize eviction policy for multi-tier decisions
        self.eviction_policy: Optional[EvictionPolicy] = None
        if optane_config.is_enabled() and optane_block_pool is not None:
            self.eviction_policy = create_eviction_policy(
                optane_config.optane_eviction_policy
            )

        # Event queue for KV-Event integration
        self.event_queue = OptaneEventQueue()

        # Statistics tracking
        self.stats = OptaneManagerStats()

        # Block tier mapping: block_id -> current tier
        self.block_to_tier: dict[int, EvictionTier] = {}

        # Track pending block migrations
        self.pending_promotions: set[int] = set()  # block_ids to promote to GPU
        self.pending_demotions: set[int] = set()  # block_ids to demote from GPU

        logger.info(
            f"OptaneManager initialized with config: "
            f"enabled={optane_config.is_enabled()}, "
            f"policy={optane_config.optane_eviction_policy}"
        )

    def on_block_allocated(self, block: KVCacheBlock) -> BlockTierPlacement:
        """
        Called when a block is allocated by KVCacheManager.

        Decides which tier the block should be placed in based on:
        - Current tier utilization
        - Request priority
        - Eviction policy

        Args:
            block: The allocated KVCacheBlock

        Returns:
            BlockTierPlacement indicating where to store the block
        """
        if not self.optane_config.is_enabled():
            # Optane disabled, keep all blocks on GPU
            self.block_to_tier[block.block_id] = EvictionTier.GPU
            if self.enable_metrics:
                self.stats.gpu_tier.num_blocks += 1
                self.stats.total_blocks_allocated += 1
            return BlockTierPlacement.GPU

        # Check GPU tier utilization
        gpu_usage = self.block_pool.get_usage()

        # Decision logic: if GPU > threshold, prefer Optane
        gpu_threshold = 0.8  # TODO: make configurable
        if gpu_usage > gpu_threshold and self.optane_block_pool is not None:
            self.block_to_tier[block.block_id] = EvictionTier.OPTANE
            if self.enable_metrics:
                self.stats.optane_tier.num_blocks += 1
            return BlockTierPlacement.OPTANE
        else:
            self.block_to_tier[block.block_id] = EvictionTier.GPU
            if self.enable_metrics:
                self.stats.gpu_tier.num_blocks += 1

            return BlockTierPlacement.GPU

        self.stats.total_blocks_allocated += 1

    def on_block_freed(self, block: KVCacheBlock) -> None:
        """
        Called when a block is freed by KVCacheManager.

        Updates tier tracking and handles block cleanup across tiers.

        Args:
            block: The freed KVCacheBlock
        """
        if block.block_id not in self.block_to_tier:
            return  # Block not tracked (shouldn't happen normally)

        tier = self.block_to_tier.pop(block.block_id)

        if self.enable_metrics:
            tier_stats = self._get_tier_stats(tier)
            tier_stats.num_blocks = max(0, tier_stats.num_blocks - 1)
            self.stats.total_blocks_freed += 1

        # If block was in Optane tier, clean up in OptaneBlockPool
        if tier == EvictionTier.OPTANE and self.optane_block_pool is not None:
            try:
                self.optane_block_pool.evict_block(block.block_id)
            except Exception as e:
                logger.warning(
                    f"Failed to evict block {block.block_id} from Optane: {e}"
                )

    def on_block_evicted(self, block: KVCacheBlock) -> None:
        """
        Called when a block is evicted from GPU BlockPool.

        Determines if block should be promoted to Optane tier or
        discarded based on eviction policy.

        Args:
            block: The evicted KVCacheBlock
        """
        if not self.optane_config.is_enabled() or self.optane_block_pool is None:
            return

        current_tier = self.block_to_tier.get(block.block_id, EvictionTier.GPU)

        # Only promote blocks that were on GPU tier
        if current_tier != EvictionTier.GPU:
            return

        # Query eviction policy for next tier decision
        if self.eviction_policy is None:
            return

        # Get eviction candidates to determine promotion priority
        candidates = self.eviction_policy.get_candidates(
            num_candidates=1, block=block
        )

        if candidates:
            next_tier = self.eviction_policy.get_next_tier(current_tier)

            if next_tier == EvictionTier.OPTANE:
                # Mark block for promotion to Optane
                self.pending_promotions.add(block.block_id)
                self.block_to_tier[block.block_id] = EvictionTier.OPTANE

                if self.enable_metrics:
                    self.stats.gpu_tier.num_evictions += 1
                    self.stats.optane_tier.num_promotions += 1
                    self.stats.total_tier_crossings += 1

    def on_block_prefetched(self, block: KVCacheBlock) -> None:
        """
        Called when a block is prefetched from a lower tier.

        Updates tier tracking when Optane block is loaded back to GPU.

        Args:
            block: The prefetched KVCacheBlock
        """
        current_tier = self.block_to_tier.get(block.block_id, EvictionTier.GPU)

        if current_tier in (EvictionTier.OPTANE, EvictionTier.CPU):
            self.block_to_tier[block.block_id] = EvictionTier.GPU

            if self.enable_metrics:
                tier_stats = self._get_tier_stats(current_tier)
                tier_stats.num_blocks = max(0, tier_stats.num_blocks - 1)
                self.stats.gpu_tier.num_blocks += 1
                self.stats.gpu_tier.num_promotions += 1
                self.stats.total_tier_crossings += 1

    def get_block_tier(self, block_id: int) -> EvictionTier:
        """
        Get the current tier of a block.

        Args:
            block_id: The block ID

        Returns:
            The EvictionTier where the block currently resides
        """
        return self.block_to_tier.get(block_id, EvictionTier.GPU)

    def get_stats(self) -> OptaneManagerStats:
        """
        Get current statistics.

        Returns:
            OptaneManagerStats containing tier utilization and counters
        """
        return self.stats

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats = OptaneManagerStats()

    def take_events(self) -> list:
        """
        Atomically take all pending KV-Events.

        Returns:
            List of KV-Event objects
        """
        return self.event_queue.take_events()

    def _get_tier_stats(self, tier: EvictionTier) -> TierStats:
        """Get statistics for a tier."""
        if tier == EvictionTier.GPU:
            return self.stats.gpu_tier
        elif tier == EvictionTier.CPU:
            return self.stats.cpu_tier
        elif tier == EvictionTier.OPTANE:
            return self.stats.optane_tier
        elif tier == EvictionTier.SSD:
            return self.stats.ssd_tier
        else:  # EvictionTier.REMOTE
            return self.stats.remote_tier
