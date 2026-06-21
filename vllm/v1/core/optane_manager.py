# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OptaneManager: Manages Optane persistent memory tier for KV cache blocks.

This module provides the OptaneManager class which bridges vLLM's KV cache
management system with the Optane persistent memory tier. It handles block
promotion, demotion, eviction, and observability through KV-events.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import logging

import torch

from vllm.logger import init_logger
from vllm.config.optane import OptaneConfig
from vllm.core.eviction_policy_optane import (
    EvictionTier,
    create_eviction_policy,
    EvictionPolicy,
)
from vllm.v1.core.block_pool_optane import OptaneBlockPool
from vllm.distributed.kv_events_optane import OptaneEventQueue

logger = init_logger(__name__)


@dataclass
class OptaneStats:
    """Statistics for Optane tier operations."""

    blocks_promoted: int = 0
    """Total blocks promoted to Optane"""

    blocks_demoted: int = 0
    """Total blocks demoted from Optane"""

    blocks_evicted: int = 0
    """Total blocks evicted from Optane to SSD/Remote"""

    total_promote_time_us: float = 0.0
    """Cumulative promotion latency in microseconds"""

    total_demote_time_us: float = 0.0
    """Cumulative demotion latency in microseconds"""

    promotion_failures: int = 0
    """Number of failed promotion attempts (Optane full)"""

    demotion_failures: int = 0
    """Number of failed demotion attempts"""

    def get_avg_promote_latency_us(self) -> float:
        """Get average promotion latency."""
        if self.blocks_promoted > 0:
            return self.total_promote_time_us / self.blocks_promoted
        return 0.0

    def get_avg_demote_latency_us(self) -> float:
        """Get average demotion latency."""
        if self.blocks_demoted > 0:
            return self.total_demote_time_us / self.blocks_demoted
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for reporting."""
        return {
            "blocks_promoted": self.blocks_promoted,
            "blocks_demoted": self.blocks_demoted,
            "blocks_evicted": self.blocks_evicted,
            "avg_promote_latency_us": self.get_avg_promote_latency_us(),
            "avg_demote_latency_us": self.get_avg_demote_latency_us(),
            "promotion_failures": self.promotion_failures,
            "demotion_failures": self.demotion_failures,
        }


class OptaneManager:
    """Manages Optane persistent memory tier for KV cache blocks.

    Responsibilities:
    1. Track Optane block allocation and usage
    2. Handle block promotion (GPU/CPU → Optane) and demotion (Optane → GPU/CPU)
    3. Emit KV-events for observability
    4. Manage eviction policy integration
    5. Report metrics and statistics

    Args:
        optane_config: Optane configuration parameters
        block_size: Size of each KV cache block in tokens
        max_blocks: Maximum number of blocks to store in Optane
        enable_events: Whether to enable KV-event emission
        enable_metrics: Whether to enable metrics collection
    """

    def __init__(
        self,
        optane_config: OptaneConfig,
        block_size: int,
        max_blocks: int,
        enable_events: bool = False,
        enable_metrics: bool = False,
    ) -> None:
        """Initialize OptaneManager.

        Args:
            optane_config: Optane configuration
            block_size: KV cache block size in tokens
            max_blocks: Maximum Optane blocks
            enable_events: Enable KV-event emission
            enable_metrics: Enable metrics collection
        """
        self.optane_config = optane_config
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.enable_events = enable_events
        self.enable_metrics = enable_metrics

        # Initialize Optane block pool
        self.pool = OptaneBlockPool(
            config=optane_config,
            block_size=block_size,
            num_blocks=max_blocks,
        )

        # Initialize eviction policy
        policy_name = optane_config.optane_eviction_policy or "cascade"
        self.eviction_policy = create_eviction_policy(policy_name)

        # Initialize event queue
        self.event_queue = OptaneEventQueue() if enable_events else None

        # Track block locations: block_id -> (gpu_tensor, promotion_time)
        self.promoted_blocks: Dict[int, tuple] = {}

        # Statistics
        self.stats = OptaneStats()

        logger.info(
            f"Initialized OptaneManager: {max_blocks} blocks, "
            f"{optane_config.get_optane_bytes() / (1024**3):.2f} GiB, "
            f"policy={policy_name}, backend={optane_config.optane_backend}"
        )

    def promote_block_to_optane(
        self,
        block_id: int,
        kv_data: torch.Tensor,
        parent_hash: Optional[str] = None,
        group_idx: int = 0,
    ) -> bool:
        """Promote a KV cache block from GPU/CPU to Optane persistent memory.

        Args:
            block_id: Unique identifier for the block
            kv_data: KV tensor data (shape: [2, num_tokens, num_heads, head_dim])
            parent_hash: Optional hash of parent block for prefix caching
            group_idx: KV cache group index

        Returns:
            True if promotion succeeded, False if Optane capacity insufficient
        """
        start_time = time.time()

        try:
            # Check if Optane has capacity
            if self.pool.get_num_free_blocks() == 0:
                logger.debug(
                    f"Cannot promote block {block_id}: Optane at capacity "
                    f"({self.pool.get_num_allocated_blocks()}/{self.max_blocks})"
                )
                self.stats.promotion_failures += 1
                return False

            # Allocate block in Optane
            if not self.pool.allocate_block(block_id):
                logger.warning(f"Failed to allocate block {block_id} in Optane")
                self.stats.promotion_failures += 1
                return False

            # Store KV data
            if not self.pool.store_kv_block(block_id, kv_data, parent_hash):
                logger.warning(f"Failed to store block {block_id} data in Optane")
                self.pool.evict_block(block_id)
                self.stats.promotion_failures += 1
                return False

            # Track promoted block
            self.promoted_blocks[block_id] = (kv_data, time.time())

            # Update statistics
            elapsed_us = (time.time() - start_time) * 1_000_000
            self.stats.blocks_promoted += 1
            self.stats.total_promote_time_us += elapsed_us

            # Emit event if enabled
            if self.event_queue is not None:
                self.event_queue.append_block_stored(
                    block_ids=[block_id],
                    parent_block_hash=parent_hash,
                    token_ids=list(range(kv_data.shape[1])),
                    block_size=self.block_size,
                    group_idx=group_idx,
                )

            logger.debug(
                f"Promoted block {block_id} to Optane ({elapsed_us:.2f} µs). "
                f"Optane: {self.pool.get_num_allocated_blocks()}/{self.max_blocks}"
            )
            return True

        except Exception as e:
            logger.error(f"Error promoting block {block_id} to Optane: {e}")
            self.stats.promotion_failures += 1
            return False

    def demote_block_from_optane(
        self,
        block_id: int,
        device: torch.device = torch.device("cpu"),
        group_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        """Demote a KV cache block from Optane back to GPU/CPU.

        Args:
            block_id: Block identifier
            device: Target device (GPU or CPU)
            group_idx: KV cache group index

        Returns:
            Retrieved KV tensor or None if retrieval failed
        """
        start_time = time.time()

        try:
            # Retrieve from Optane
            kv_data = self.pool.retrieve_kv_block(block_id, device)

            if kv_data is None:
                logger.warning(f"Failed to retrieve block {block_id} from Optane")
                self.stats.demotion_failures += 1
                return None

            # Update statistics
            elapsed_us = (time.time() - start_time) * 1_000_000
            self.stats.blocks_demoted += 1
            self.stats.total_demote_time_us += elapsed_us

            # Remove from tracking
            self.promoted_blocks.pop(block_id, None)

            # Emit event if enabled
            if self.event_queue is not None:
                self.event_queue.append_block_accessed(
                    block_ids=[block_id],
                    access_type="read",
                    latency_us=elapsed_us / 1000.0,  # Convert to ms
                    group_idx=group_idx,
                )

            logger.debug(
                f"Demoted block {block_id} from Optane ({elapsed_us:.2f} µs). "
                f"Optane: {self.pool.get_num_allocated_blocks()}/{self.max_blocks}"
            )
            return kv_data

        except Exception as e:
            logger.error(f"Error demoting block {block_id} from Optane: {e}")
            self.stats.demotion_failures += 1
            return None

    def should_promote_to_optane(
        self,
        gpu_usage_fraction: float,
        optane_usage_fraction: float,
        block_access_frequency: float = 0.0,
    ) -> bool:
        """Determine if blocks should be promoted to Optane.

        Decision logic based on:
        - GPU memory pressure (usage > threshold)
        - Optane availability
        - Block access patterns

        Args:
            gpu_usage_fraction: Current GPU memory usage (0.0 to 1.0)
            optane_usage_fraction: Current Optane usage (0.0 to 1.0)
            block_access_frequency: Access frequency of candidate block (optional)

        Returns:
            True if promotion is recommended, False otherwise
        """
        promotion_threshold = self.optane_config.optane_promotion_threshold or 0.85

        # Don't promote if GPU not under pressure
        if gpu_usage_fraction < promotion_threshold:
            return False

        # Don't promote if Optane is full
        if optane_usage_fraction >= 1.0:
            return False

        # Promote if GPU is under pressure and Optane has space
        return True

    def get_eviction_candidates(
        self,
        num_candidates: int,
    ) -> List[int]:
        """Get candidate blocks from Optane for eviction to SSD/Remote.

        Uses the configured eviction policy (LRU, LFU, cascade, or parallel).

        Args:
            num_candidates: Number of eviction candidates to return

        Returns:
            List of block IDs to evict from Optane
        """
        if not self.promoted_blocks:
            return []

        candidates = []
        allocated_blocks = list(self.promoted_blocks.keys())

        # Use eviction policy to rank candidates
        policy_candidates = self.eviction_policy.select_eviction_candidates(
            num_candidates, EvictionTier.OPTANE
        )

        # Filter to only blocks we're actually tracking
        for block_id in policy_candidates:
            if block_id in self.promoted_blocks and len(candidates) < num_candidates:
                candidates.append(block_id)

        # Fallback: if policy doesn't return enough, use FIFO on remaining
        if len(candidates) < num_candidates:
            for block_id in allocated_blocks:
                if block_id not in candidates and len(candidates) < num_candidates:
                    candidates.append(block_id)

        return candidates

    def evict_blocks_from_optane(
        self,
        block_ids: List[int],
        group_idx: int = 0,
    ) -> int:
        """Evict blocks from Optane to SSD/Remote storage.

        Args:
            block_ids: List of block IDs to evict
            group_idx: KV cache group index

        Returns:
            Number of blocks successfully evicted
        """
        evicted = 0

        for block_id in block_ids:
            if self.pool.evict_block(block_id):
                self.promoted_blocks.pop(block_id, None)
                self.stats.blocks_evicted += 1
                evicted += 1

                # Emit event if enabled
                if self.event_queue is not None:
                    self.event_queue.append_block_removed(
                        block_ids=[block_id],
                        reason="eviction",
                        group_idx=group_idx,
                    )

                logger.debug(
                    f"Evicted block {block_id} from Optane to SSD. "
                    f"Optane: {self.pool.get_num_allocated_blocks()}/{self.max_blocks}"
                )

        return evicted

    def get_num_free_blocks(self) -> int:
        """Get number of free blocks in Optane.

        Returns:
            Number of available blocks for promotion
        """
        return self.pool.get_num_free_blocks()

    def get_num_allocated_blocks(self) -> int:
        """Get number of allocated blocks in Optane.

        Returns:
            Number of blocks currently stored in Optane
        """
        return self.pool.get_num_allocated_blocks()

    def get_usage(self) -> float:
        """Get Optane tier usage as a fraction.

        Returns:
            Usage fraction (0.0 to 1.0)
        """
        return self.pool.get_usage()

    def get_promoted_block_ids(self) -> List[int]:
        """Get list of all blocks currently in Optane.

        Returns:
            List of block IDs
        """
        return list(self.promoted_blocks.keys())

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for Optane tier.

        Returns:
            Dictionary containing:
            - Optane pool metrics
            - Promotion/demotion statistics
            - Event queue status
        """
        metrics = {
            "optane": self.pool.get_metrics(),
            "stats": self.stats.to_dict(),
        }

        if self.event_queue is not None:
            metrics["event_queue_size"] = len(self.event_queue)
            metrics["events_dropped"] = self.event_queue.dropped_events

        return metrics

    def get_events(self) -> List[Any]:
        """Get and clear all pending KV-events.

        Returns:
            List of KV-events
        """
        if self.event_queue is None:
            return []
        return self.event_queue.take_events()

    def reset(self) -> None:
        """Reset all Optane blocks and metrics."""
        self.pool.reset()
        self.promoted_blocks.clear()
        self.stats = OptaneStats()
        if self.event_queue is not None:
            self.event_queue.clear()
        logger.info("OptaneManager reset")

    def shutdown(self) -> None:
        """Cleanly shutdown OptaneManager and release resources."""
        self.reset()
        logger.info("OptaneManager shutdown")
