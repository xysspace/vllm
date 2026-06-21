# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block lifecycle management across memory tiers."""

from typing import Dict, Optional
import time

from vllm.logger import init_logger
from vllm.v1.core.optane_types import (
    EvictionTier,
    BlockState,
)

logger = init_logger(__name__)


class BlockLifecycleManager:
    """Manages block state transitions across memory tiers.
    
    Tracks complete lifecycle: allocation → promotion/demotion → freedom
    """

    def __init__(self):
        """Initialize block lifecycle manager."""
        self.blocks: Dict[int, BlockState] = {}
        self.tier_counts: Dict[EvictionTier, int] = {tier: 0 for tier in EvictionTier}

    def allocate(
        self,
        block_id: int,
        size_bytes: int,
        initial_tier: EvictionTier = EvictionTier.GPU,
    ) -> BlockState:
        """Allocate a new block.
        
        Args:
            block_id: Unique block identifier
            size_bytes: Block size
            initial_tier: Initial tier (default: GPU)
            
        Returns:
            BlockState for the new block
        """
        if block_id in self.blocks:
            logger.warning(f"Block {block_id} already allocated, reallocating")
            self.free(block_id)

        state = BlockState(
            block_id=block_id,
            current_tier=initial_tier,
            ref_cnt=1,
            access_time=time.time(),
            access_freq=0,
            compressed=False,
        )

        self.blocks[block_id] = state
        self.tier_counts[initial_tier] += 1

        logger.debug(f"Allocated block {block_id} in {initial_tier.name} tier")
        return state

    def promote(
        self,
        block_id: int,
        target_tier: EvictionTier,
    ) -> bool:
        """Move block to higher-tier memory (faster).
        
        Args:
            block_id: Block to promote
            target_tier: Target tier
            
        Returns:
            True if successful
        """
        if block_id not in self.blocks:
            logger.warning(f"Cannot promote block {block_id}: not found")
            return False

        state = self.blocks[block_id]
        old_tier = state.current_tier

        # Verify promotion direction
        tier_order = [EvictionTier.SSD, EvictionTier.OPTANE, EvictionTier.CPU, EvictionTier.GPU]
        if tier_order.index(target_tier) > tier_order.index(old_tier):
            logger.warning(
                f"Invalid promotion: {old_tier.name} -> {target_tier.name} "
                f"(should go towards GPU)"
            )
            return False

        # Update tier counts
        self.tier_counts[old_tier] -= 1
        self.tier_counts[target_tier] += 1

        # Update state
        state.current_tier = target_tier
        state.access_time = time.time()

        logger.debug(f"Promoted block {block_id}: {old_tier.name} -> {target_tier.name}")
        return True

    def demote(
        self,
        block_id: int,
        target_tier: EvictionTier,
    ) -> bool:
        """Move block to lower-tier memory (more storage).
        
        Args:
            block_id: Block to demote
            target_tier: Target tier
            
        Returns:
            True if successful
        """
        if block_id not in self.blocks:
            logger.warning(f"Cannot demote block {block_id}: not found")
            return False

        state = self.blocks[block_id]
        old_tier = state.current_tier

        # Verify demotion direction
        tier_order = [EvictionTier.SSD, EvictionTier.OPTANE, EvictionTier.CPU, EvictionTier.GPU]
        if tier_order.index(target_tier) < tier_order.index(old_tier):
            logger.warning(
                f"Invalid demotion: {old_tier.name} -> {target_tier.name} "
                f"(should go towards SSD)"
            )
            return False

        # Update tier counts
        self.tier_counts[old_tier] -= 1
        self.tier_counts[target_tier] += 1

        # Update state
        state.current_tier = target_tier
        state.access_time = time.time()

        logger.debug(f"Demoted block {block_id}: {old_tier.name} -> {target_tier.name}")
        return True

    def free(self, block_id: int) -> bool:
        """Free a block.
        
        Args:
            block_id: Block to free
            
        Returns:
            True if successful
        """
        if block_id not in self.blocks:
            return False

        state = self.blocks[block_id]
        self.tier_counts[state.current_tier] -= 1
        del self.blocks[block_id]

        logger.debug(f"Freed block {block_id}")
        return True

    def touch(self, block_id: int) -> bool:
        """Mark block as recently accessed.
        
        Args:
            block_id: Block to touch
            
        Returns:
            True if successful
        """
        if block_id not in self.blocks:
            return False

        state = self.blocks[block_id]
        state.access_time = time.time()
        state.access_freq += 1
        return True

    def get_state(self, block_id: int) -> Optional[BlockState]:
        """Get current state of a block.
        
        Args:
            block_id: Block to query
            
        Returns:
            BlockState if found, None otherwise
        """
        return self.blocks.get(block_id)

    def get_blocks_in_tier(self, tier: EvictionTier) -> list[int]:
        """Get all blocks in a tier.
        
        Args:
            tier: Tier to query
            
        Returns:
            List of block IDs in that tier
        """
        return [bid for bid, state in self.blocks.items() if state.current_tier == tier]

    def get_tier_usage(self) -> Dict[EvictionTier, int]:
        """Get block count by tier.
        
        Returns:
            Dictionary with block counts per tier
        """
        return self.tier_counts.copy()

    def get_statistics(self) -> dict:
        """Get lifecycle statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_blocks": len(self.blocks),
            "blocks_by_tier": self.tier_counts.copy(),
        }
