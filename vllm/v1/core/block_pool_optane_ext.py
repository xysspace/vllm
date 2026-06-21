# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""BlockPool extensions for Optane tier integration.

Extends the standard KV cache block pool with methods for promoting/demoting
blocks to/from Optane persistent memory tier.
"""

from typing import List, Optional, Dict, Any, Tuple
import logging

import torch

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)


class OptaneTierInfo:
    """Information about a block's location across memory tiers."""

    def __init__(self, block_id: int):
        self.block_id = block_id
        # Tier tracking: which tiers contain this block
        # Values: 'gpu', 'cpu', 'optane', 'ssd'
        self.current_tier: str = 'gpu'
        self.is_in_optane: bool = False
        self.is_in_ssd: bool = False
        self.promotion_time: Optional[float] = None
        self.last_access_time: Optional[float] = None
        self.access_count: int = 0


class BlockPoolExtensions:
    """Extensions to KV cache block pool for Optane tier management.

    Provides methods for:
    - Tracking block location across tiers
    - Promoting blocks to Optane
    - Demoting blocks from Optane
    - Computing tier-aware eviction candidates
    - Reporting tier usage statistics
    """

    def __init__(self, base_block_pool: Any):
        """Initialize block pool extensions.

        Args:
            base_block_pool: The base KVCacheBlockPool to extend
        """
        self.base_pool = base_block_pool
        
        # Track Optane-specific information for each block
        self.block_tier_info: Dict[int, OptaneTierInfo] = {}
        
        # Blocks currently in Optane
        self.optane_blocks: set = set()
        
        # Blocks currently in SSD
        self.ssd_blocks: set = set()
        
        logger.info("Initialized BlockPoolExtensions for Optane tier support")

    def get_or_create_tier_info(self, block_id: int) -> OptaneTierInfo:
        """Get or create tier info for a block.

        Args:
            block_id: The block ID

        Returns:
            OptaneTierInfo for the block
        """
        if block_id not in self.block_tier_info:
            self.block_tier_info[block_id] = OptaneTierInfo(block_id)
        return self.block_tier_info[block_id]

    def mark_promoted_to_optane(self, block_id: int) -> None:
        """Mark a block as promoted to Optane.

        Args:
            block_id: The block ID
        """
        info = self.get_or_create_tier_info(block_id)
        info.current_tier = 'optane'
        info.is_in_optane = True
        info.promotion_time = __import__('time').time()
        self.optane_blocks.add(block_id)

    def mark_demoted_from_optane(self, block_id: int, target_tier: str = 'gpu') -> None:
        """Mark a block as demoted from Optane.

        Args:
            block_id: The block ID
            target_tier: Target tier after demotion (default: 'gpu')
        """
        info = self.get_or_create_tier_info(block_id)
        info.current_tier = target_tier
        info.is_in_optane = False
        info.promotion_time = None
        self.optane_blocks.discard(block_id)

    def mark_promoted_to_ssd(self, block_id: int) -> None:
        """Mark a block as promoted to SSD.

        Args:
            block_id: The block ID
        """
        info = self.get_or_create_tier_info(block_id)
        info.current_tier = 'ssd'
        info.is_in_ssd = True
        self.ssd_blocks.add(block_id)

    def mark_demoted_from_ssd(self, block_id: int) -> None:
        """Mark a block as demoted from SSD.

        Args:
            block_id: The block ID
        """
        info = self.get_or_create_tier_info(block_id)
        info.current_tier = 'gpu'
        info.is_in_ssd = False
        self.ssd_blocks.discard(block_id)

    def record_block_access(self, block_id: int) -> None:
        """Record a block access for tier statistics.

        Args:
            block_id: The block ID
        """
        import time
        info = self.get_or_create_tier_info(block_id)
        info.last_access_time = time.time()
        info.access_count += 1

    def get_tier_info(self, block_id: int) -> Optional[OptaneTierInfo]:
        """Get tier info for a block.

        Args:
            block_id: The block ID

        Returns:
            OptaneTierInfo or None if not tracked
        """
        return self.block_tier_info.get(block_id)

    def get_blocks_in_optane(self) -> List[int]:
        """Get list of all blocks currently in Optane.

        Returns:
            List of block IDs in Optane
        """
        return list(self.optane_blocks)

    def get_blocks_in_ssd(self) -> List[int]:
        """Get list of all blocks currently in SSD.

        Returns:
            List of block IDs in SSD
        """
        return list(self.ssd_blocks)

    def get_num_blocks_in_optane(self) -> int:
        """Get count of blocks in Optane.

        Returns:
            Number of blocks in Optane
        """
        return len(self.optane_blocks)

    def get_num_blocks_in_ssd(self) -> int:
        """Get count of blocks in SSD.

        Returns:
            Number of blocks in SSD
        """
        return len(self.ssd_blocks)

    def get_tier_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for each tier.

        Returns:
            Dictionary with tier usage information
        """
        total_blocks = len(self.block_tier_info)
        gpu_blocks = total_blocks - len(self.optane_blocks) - len(self.ssd_blocks)
        
        return {
            'gpu_blocks': gpu_blocks,
            'optane_blocks': len(self.optane_blocks),
            'ssd_blocks': len(self.ssd_blocks),
            'total_blocks': total_blocks,
            'gpu_fraction': gpu_blocks / total_blocks if total_blocks > 0 else 0.0,
            'optane_fraction': len(self.optane_blocks) / total_blocks if total_blocks > 0 else 0.0,
            'ssd_fraction': len(self.ssd_blocks) / total_blocks if total_blocks > 0 else 0.0,
        }

    def get_eviction_candidates_by_tier(
        self,
        tier: str,
        num_candidates: int,
        policy: str = 'lru'
    ) -> List[int]:
        """Get eviction candidates from a specific tier.

        Candidates are selected based on LRU or LFU policy.

        Args:
            tier: Tier to select from ('optane' or 'ssd')
            num_candidates: Number of candidates to return
            policy: Eviction policy ('lru' or 'lfu')

        Returns:
            List of block IDs to evict
        """
        if tier == 'optane':
            blocks = self.get_blocks_in_optane()
        elif tier == 'ssd':
            blocks = self.get_blocks_in_ssd()
        else:
            logger.warning(f"Unknown tier for eviction: {tier}")
            return []

        if not blocks:
            return []

        # Sort by policy
        if policy == 'lru':
            # Sort by last access time (oldest first)
            sorted_blocks = sorted(
                blocks,
                key=lambda bid: (
                    self.block_tier_info[bid].last_access_time or 0.0
                )
            )
        elif policy == 'lfu':
            # Sort by access count (least frequently used first)
            sorted_blocks = sorted(
                blocks,
                key=lambda bid: self.block_tier_info[bid].access_count
            )
        else:
            logger.warning(f"Unknown eviction policy: {policy}")
            sorted_blocks = blocks

        return sorted_blocks[:num_candidates]

    def get_promotion_candidates(
        self,
        gpu_blocks: List[int],
        num_candidates: int,
        policy: str = 'lfu'
    ) -> List[int]:
        """Get GPU blocks that are good candidates for promotion to Optane.

        Selects frequently accessed blocks that would benefit from Optane's
        lower latency compared to SSD.

        Args:
            gpu_blocks: List of blocks currently on GPU
            num_candidates: Number of candidates to return
            policy: Selection policy ('lfu' or 'lru')

        Returns:
            List of block IDs to promote
        """
        if not gpu_blocks:
            return []

        # Sort by access frequency or recency
        if policy == 'lfu':
            # Promote frequently accessed blocks
            sorted_blocks = sorted(
                gpu_blocks,
                key=lambda bid: -self.block_tier_info[bid].access_count
            )
        else:  # 'lru' - promote recently accessed
            sorted_blocks = sorted(
                gpu_blocks,
                key=lambda bid: -(self.block_tier_info[bid].last_access_time or 0.0)
            )

        return sorted_blocks[:num_candidates]

    def cleanup_tier_info(self, block_id: int) -> None:
        """Clean up tier info for a freed block.

        Args:
            block_id: The block ID to clean up
        """
        self.block_tier_info.pop(block_id, None)
        self.optane_blocks.discard(block_id)
        self.ssd_blocks.discard(block_id)

    def reset(self) -> None:
        """Reset all tier tracking information."""
        self.block_tier_info.clear()
        self.optane_blocks.clear()
        self.ssd_blocks.clear()
        logger.info("BlockPoolExtensions reset")


class TierAwareBlockPool:
    """A block pool wrapper that integrates tier awareness.

    Combines the base block pool with Optane tier extensions to provide
    a unified interface for managing blocks across GPU, Optane, and SSD.
    """

    def __init__(self, base_block_pool: Any):
        """Initialize tier-aware block pool.

        Args:
            base_block_pool: The base KVCacheBlockPool to wrap
        """
        self.base_pool = base_block_pool
        self.extensions = BlockPoolExtensions(base_block_pool)

    def promote_to_optane(
        self,
        block_ids: List[int],
        optane_manager: Any,
        block_data_map: Dict[int, torch.Tensor]
    ) -> Tuple[List[int], int]:
        """Promote blocks to Optane tier.

        Args:
            block_ids: Block IDs to promote
            optane_manager: OptaneManager instance
            block_data_map: Mapping of block_id -> KV tensor data

        Returns:
            Tuple of (successfully_promoted_ids, num_promoted)
        """
        promoted = []
        for block_id in block_ids:
            if block_id not in block_data_map:
                logger.warning(f"Block {block_id} not in data map, skipping promotion")
                continue

            kv_data = block_data_map[block_id]
            if optane_manager.promote_block_to_optane(block_id, kv_data):
                self.extensions.mark_promoted_to_optane(block_id)
                promoted.append(block_id)

        return promoted, len(promoted)

    def demote_from_optane(
        self,
        block_ids: List[int],
        optane_manager: Any,
        target_tier: str = 'gpu'
    ) -> Tuple[Dict[int, torch.Tensor], int]:
        """Demote blocks from Optane tier.

        Args:
            block_ids: Block IDs to demote
            optane_manager: OptaneManager instance
            target_tier: Target tier after demotion ('gpu' or 'cpu')

        Returns:
            Tuple of (block_id_to_tensor_map, num_demoted)
        """
        demoted_data = {}
        device = torch.device('cuda' if target_tier == 'gpu' else 'cpu')

        for block_id in block_ids:
            kv_data = optane_manager.demote_block_from_optane(
                block_id,
                device=device
            )
            if kv_data is not None:
                demoted_data[block_id] = kv_data
                self.extensions.mark_demoted_from_optane(block_id, target_tier)

        return demoted_data, len(demoted_data)

    def get_tier_stats(self) -> Dict[str, Any]:
        """Get comprehensive tier statistics.

        Returns:
            Dictionary with tier usage and performance stats
        """
        stats = {
            'tier_usage': self.extensions.get_tier_usage_stats(),
            'base_pool': self.base_pool.get_metrics() if hasattr(self.base_pool, 'get_metrics') else {},
        }
        return stats

    def get_promotion_recommendation(
        self,
        gpu_memory_usage: float,
        optane_usage: float,
        num_candidates: int = 10
    ) -> List[int]:
        """Get recommended blocks for promotion to Optane.

        Args:
            gpu_memory_usage: Current GPU memory usage fraction (0.0-1.0)
            optane_usage: Current Optane usage fraction (0.0-1.0)
            num_candidates: Number of recommendations to return

        Returns:
            List of block IDs recommended for promotion
        """
        # Only recommend promotion if GPU is under pressure and Optane has space
        if gpu_memory_usage < 0.8 or optane_usage > 0.9:
            return []

        # Get frequently accessed GPU blocks
        gpu_blocks = [
            bid for bid in self.extensions.block_tier_info.keys()
            if self.extensions.block_tier_info[bid].current_tier == 'gpu'
        ]

        return self.extensions.get_promotion_candidates(
            gpu_blocks,
            num_candidates,
            policy='lfu'
        )

    def reset(self) -> None:
        """Reset all tier tracking."""
        self.extensions.reset()
