# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction policies for multi-tier KV cache management."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import time
import heapq
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.core.optane_types import EvictionTier, OptaneBlockInfo

logger = init_logger(__name__)


class EvictionPolicy(ABC):
    """Abstract base class for eviction policies.
    
    Determines which blocks to evict when memory pressure occurs.
    """

    def __init__(self, name: str):
        """Initialize policy.
        
        Args:
            name: Policy name
        """
        self.name = name
        self.decision_count = 0

    @abstractmethod
    def get_eviction_candidates(
        self,
        blocks: Dict[int, OptaneBlockInfo],
        num_candidates: int,
    ) -> List[int]:
        """Get candidate blocks for eviction.
        
        Args:
            blocks: All blocks to consider
            num_candidates: Number of candidates to return
            
        Returns:
            List of block IDs to evict (ordered by priority)
        """
        pass

    @abstractmethod
    def record_access(self, block_id: int) -> None:
        """Record block access for policy tracking.
        
        Args:
            block_id: Block that was accessed
        """
        pass

    def on_block_allocated(self, block_id: int) -> None:
        """Called when a block is allocated.
        
        Args:
            block_id: Block that was allocated
        """
        pass

    def on_block_freed(self, block_id: int) -> None:
        """Called when a block is freed.
        
        Args:
            block_id: Block that was freed
        """
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy.
    
    Evicts blocks with oldest access timestamp.
    Best for streaming inference with sequential access patterns.
    """

    def __init__(self):
        """Initialize LRU policy."""
        super().__init__("lru")
        self.access_times: Dict[int, float] = {}

    def get_eviction_candidates(
        self,
        blocks: Dict[int, OptaneBlockInfo],
        num_candidates: int,
    ) -> List[int]:
        """Get least recently used blocks.
        
        Args:
            blocks: All blocks to consider
            num_candidates: Number of candidates
            
        Returns:
            Block IDs sorted by access time (oldest first)
        """
        self.decision_count += 1

        # Sort by access_time (older is earlier)
        candidates = sorted(
            blocks.items(),
            key=lambda x: x[1].access_time,
        )[:num_candidates]

        return [block_id for block_id, _ in candidates]

    def record_access(self, block_id: int) -> None:
        """Update access timestamp.
        
        Args:
            block_id: Block that was accessed
        """
        self.access_times[block_id] = time.time()


class LFUEvictionPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy.
    
    Evicts blocks with lowest access frequency.
    Best for workloads with varying block importance.
    """

    def __init__(self):
        """Initialize LFU policy."""
        super().__init__("lfu")
        self.access_counts: Dict[int, int] = {}

    def get_eviction_candidates(
        self,
        blocks: Dict[int, OptaneBlockInfo],
        num_candidates: int,
    ) -> List[int]:
        """Get least frequently used blocks.
        
        Args:
            blocks: All blocks to consider
            num_candidates: Number of candidates
            
        Returns:
            Block IDs sorted by frequency (least first)
        """
        self.decision_count += 1

        # Sort by access_count (lower is earlier)
        candidates = sorted(
            blocks.items(),
            key=lambda x: x[1].access_count,
        )[:num_candidates]

        return [block_id for block_id, _ in candidates]

    def record_access(self, block_id: int) -> None:
        """Increment access count.
        
        Args:
            block_id: Block that was accessed
        """
        if block_id not in self.access_counts:
            self.access_counts[block_id] = 0
        self.access_counts[block_id] += 1

    def on_block_allocated(self, block_id: int) -> None:
        """Initialize counter for new block.
        
        Args:
            block_id: Block that was allocated
        """
        self.access_counts[block_id] = 0

    def on_block_freed(self, block_id: int) -> None:
        """Clean up counter for freed block.
        
        Args:
            block_id: Block that was freed
        """
        self.access_counts.pop(block_id, None)


class CascadeEvictionPolicy(EvictionPolicy):
    """Cascade/Strict Hierarchy eviction policy (default).
    
    Enforces tier boundaries: GPU → CPU → Optane → SSD
    Best for predictable production deployments.
    """

    def __init__(self):
        """Initialize Cascade policy."""
        super().__init__("cascade")
        # Track which tier blocks are in
        self.blocks_by_tier: Dict[EvictionTier, set] = {
            tier: set() for tier in EvictionTier
        }

    def get_eviction_candidates(
        self,
        blocks: Dict[int, OptaneBlockInfo],
        num_candidates: int,
    ) -> List[int]:
        """Get blocks to demote to next tier.
        
        Uses LRU within each tier for selection.
        
        Args:
            blocks: All blocks to consider
            num_candidates: Number of candidates
            
        Returns:
            Block IDs to demote (oldest first per tier)
        """
        self.decision_count += 1

        candidates = []

        # Try each tier from hottest to coldest
        tier_order = [EvictionTier.GPU, EvictionTier.CPU, EvictionTier.OPTANE, EvictionTier.SSD]

        for tier in tier_order:
            if len(candidates) >= num_candidates:
                break

            # Get blocks in this tier
            tier_blocks = {bid: b for bid, b in blocks.items() if b.current_tier == tier}

            if not tier_blocks:
                continue

            # Sort by access_time (oldest first)
            sorted_blocks = sorted(
                tier_blocks.items(),
                key=lambda x: x[1].access_time,
            )

            # Add to candidates
            for block_id, _ in sorted_blocks:
                if len(candidates) < num_candidates:
                    candidates.append(block_id)

        return candidates

    def record_access(self, block_id: int) -> None:
        """Update block access time.
        
        Args:
            block_id: Block that was accessed
        """
        pass  # Cascade policy relies on block metadata

    def on_block_allocated(self, block_id: int) -> None:
        """Track new block in tier.
        
        Args:
            block_id: Block that was allocated
        """
        self.blocks_by_tier[EvictionTier.GPU].add(block_id)

    def on_block_freed(self, block_id: int) -> None:
        """Remove freed block from tracking.
        
        Args:
            block_id: Block that was freed
        """
        for tier_blocks in self.blocks_by_tier.values():
            tier_blocks.discard(block_id)


class ParallelEvictionPolicy(EvictionPolicy):
    """Parallel/Adaptive eviction policy.
    
    Routes blocks to best available tier dynamically.
    Best for mixed workloads with variable access patterns.
    """

    def __init__(self, target_utilizations: Optional[Dict[EvictionTier, float]] = None):
        """Initialize Parallel policy.
        
        Args:
            target_utilizations: Target utilization per tier (default: 0.8 for all)
        """
        super().__init__("parallel")
        self.target_utilizations = target_utilizations or {
            tier: 0.8 for tier in EvictionTier
        }
        self.access_counts: Dict[int, int] = {}

    def get_eviction_candidates(
        self,
        blocks: Dict[int, OptaneBlockInfo],
        num_candidates: int,
    ) -> List[int]:
        """Get blocks and suggest target tiers adaptively.
        
        Selects blocks with lowest access frequency for eviction.
        
        Args:
            blocks: All blocks to consider
            num_candidates: Number of candidates
            
        Returns:
            Block IDs with lowest frequency
        """
        self.decision_count += 1

        # Combine frequency and recency: lower is better
        scored_blocks = []
        for block_id, block_info in blocks.items():
            freq = self.access_counts.get(block_id, 0)
            age = time.time() - block_info.access_time
            # Score: favor low frequency and old age
            score = (freq + 1) * (age + 1)
            scored_blocks.append((score, block_id))

        # Return lowest-scoring blocks
        scored_blocks.sort()
        return [bid for _, bid in scored_blocks[:num_candidates]]

    def record_access(self, block_id: int) -> None:
        """Increment access count.
        
        Args:
            block_id: Block that was accessed
        """
        if block_id not in self.access_counts:
            self.access_counts[block_id] = 0
        self.access_counts[block_id] += 1

    def on_block_allocated(self, block_id: int) -> None:
        """Initialize counter for new block.
        
        Args:
            block_id: Block that was allocated
        """
        self.access_counts[block_id] = 0

    def on_block_freed(self, block_id: int) -> None:
        """Clean up counter for freed block.
        
        Args:
            block_id: Block that was freed
        """
        self.access_counts.pop(block_id, None)


def create_eviction_policy(policy_name: str) -> EvictionPolicy:
    """Factory function to create eviction policies.
    
    Args:
        policy_name: Name of policy ("lru", "lfu", "cascade", "parallel")
        
    Returns:
        EvictionPolicy instance
        
    Raises:
        ValueError: If policy name is unknown
    """
    policies = {
        "lru": LRUEvictionPolicy,
        "lfu": LFUEvictionPolicy,
        "cascade": CascadeEvictionPolicy,
        "parallel": ParallelEvictionPolicy,
    }

    if policy_name not in policies:
        raise ValueError(
            f"Unknown eviction policy: {policy_name}. "
            f"Valid options: {list(policies.keys())}"
        )

    return policies[policy_name]()
