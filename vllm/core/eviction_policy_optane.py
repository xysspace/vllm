# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from vllm.logger import init_logger
from vllm.config.optane import OptaneEvictionPolicy

logger = init_logger(__name__)


class EvictionTier(Enum):
    """Memory tier in the hierarchy."""
    GPU = 0
    CPU = 1
    OPTANE = 2
    SSD = 3
    REMOTE = 4  # e.g., Ceph


@dataclass
class BlockTierInfo:
    """Information about a block's tier location."""
    block_id: int
    tier: EvictionTier
    last_access_time: float
    access_count: int
    size_bytes: int
    timestamp: float  # When block was stored in this tier


class EvictionPolicy(ABC):
    """Base class for KV cache eviction policies."""

    @abstractmethod
    def select_eviction_candidates(
        self, num_candidates: int, tier: EvictionTier
    ) -> List[int]:
        """Select blocks to evict from the given tier.
        
        Args:
            num_candidates: Number of candidates to return
            tier: The tier to evict from
            
        Returns:
            List of block IDs to evict
        """
        pass

    @abstractmethod
    def on_block_accessed(self, block_id: int, tier: EvictionTier) -> None:
        """Called when a block is accessed."""
        pass

    @abstractmethod
    def on_block_stored(self, block_id: int, tier: EvictionTier) -> None:
        """Called when a block is stored in a tier."""
        pass

    @abstractmethod
    def on_block_evicted(self, block_id: int, tier: EvictionTier) -> None:
        """Called when a block is evicted from a tier."""
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used (LRU) eviction policy.
    
    Evicts the least recently accessed block from each tier.
    """

    def __init__(self):
        self.block_info: Dict[int, BlockTierInfo] = {}
        self.tier_blocks: Dict[EvictionTier, List[int]] = {
            tier: [] for tier in EvictionTier
        }

    def select_eviction_candidates(
        self, num_candidates: int, tier: EvictionTier
    ) -> List[int]:
        """Select least recently used blocks."""
        blocks_in_tier = self.tier_blocks.get(tier, [])
        if not blocks_in_tier:
            return []

        # Sort by access time (oldest first)
        sorted_blocks = sorted(
            blocks_in_tier,
            key=lambda b: self.block_info[b].last_access_time,
        )
        return sorted_blocks[:num_candidates]

    def on_block_accessed(self, block_id: int, tier: EvictionTier) -> None:
        """Update access time on block access."""
        import time

        if block_id in self.block_info:
            self.block_info[block_id].last_access_time = time.time()
            self.block_info[block_id].access_count += 1

    def on_block_stored(self, block_id: int, tier: EvictionTier) -> None:
        """Track when block is stored in a tier."""
        import time

        self.block_info[block_id] = BlockTierInfo(
            block_id=block_id,
            tier=tier,
            last_access_time=time.time(),
            access_count=0,
            size_bytes=0,
            timestamp=time.time(),
        )
        if tier not in self.tier_blocks:
            self.tier_blocks[tier] = []
        self.tier_blocks[tier].append(block_id)

    def on_block_evicted(self, block_id: int, tier: EvictionTier) -> None:
        """Remove block from tracking on eviction."""
        if block_id in self.block_info:
            del self.block_info[block_id]
        if tier in self.tier_blocks and block_id in self.tier_blocks[tier]:
            self.tier_blocks[tier].remove(block_id)


class LFUEvictionPolicy(EvictionPolicy):
    """Least Frequently Used (LFU) eviction policy.
    
    Evicts the least frequently accessed block from each tier.
    """

    def __init__(self):
        self.block_info: Dict[int, BlockTierInfo] = {}
        self.tier_blocks: Dict[EvictionTier, List[int]] = {
            tier: [] for tier in EvictionTier
        }

    def select_eviction_candidates(
        self, num_candidates: int, tier: EvictionTier
    ) -> List[int]:
        """Select least frequently used blocks."""
        blocks_in_tier = self.tier_blocks.get(tier, [])
        if not blocks_in_tier:
            return []

        # Sort by access count (lowest first)
        sorted_blocks = sorted(
            blocks_in_tier,
            key=lambda b: self.block_info[b].access_count,
        )
        return sorted_blocks[:num_candidates]

    def on_block_accessed(self, block_id: int, tier: EvictionTier) -> None:
        """Increment access count on block access."""
        if block_id in self.block_info:
            self.block_info[block_id].access_count += 1

    def on_block_stored(self, block_id: int, tier: EvictionTier) -> None:
        """Track when block is stored in a tier."""
        import time

        self.block_info[block_id] = BlockTierInfo(
            block_id=block_id,
            tier=tier,
            last_access_time=time.time(),
            access_count=0,
            size_bytes=0,
            timestamp=time.time(),
        )
        if tier not in self.tier_blocks:
            self.tier_blocks[tier] = []
        self.tier_blocks[tier].append(block_id)

    def on_block_evicted(self, block_id: int, tier: EvictionTier) -> None:
        """Remove block from tracking on eviction."""
        if block_id in self.block_info:
            del self.block_info[block_id]
        if tier in self.tier_blocks and block_id in self.tier_blocks[tier]:
            self.tier_blocks[tier].remove(block_id)


class CascadeEvictionPolicy(EvictionPolicy):
    """Cascade eviction policy: GPU → CPU → Optane → SSD → Remote.
    
    Enforces strict tier hierarchy. When a tier is full, blocks are
    evicted to the next lower tier.
    """

    def __init__(self):
        self.block_info: Dict[int, BlockTierInfo] = {}
        self.tier_blocks: Dict[EvictionTier, List[int]] = {
            tier: [] for tier in EvictionTier
        }
        # Cascade order
        self.tier_order = [
            EvictionTier.GPU,
            EvictionTier.CPU,
            EvictionTier.OPTANE,
            EvictionTier.SSD,
            EvictionTier.REMOTE,
        ]

    def select_eviction_candidates(
        self, num_candidates: int, tier: EvictionTier
    ) -> List[int]:
        """Select oldest blocks in tier for eviction to next tier."""
        blocks_in_tier = self.tier_blocks.get(tier, [])
        if not blocks_in_tier:
            return []

        # Sort by timestamp (oldest first - cascade them down)
        sorted_blocks = sorted(
            blocks_in_tier,
            key=lambda b: self.block_info[b].timestamp,
        )
        return sorted_blocks[:num_candidates]

    def on_block_accessed(self, block_id: int, tier: EvictionTier) -> None:
        """Update access tracking."""
        import time

        if block_id in self.block_info:
            self.block_info[block_id].last_access_time = time.time()
            self.block_info[block_id].access_count += 1

    def on_block_stored(self, block_id: int, tier: EvictionTier) -> None:
        """Track block storage in tier."""
        import time

        self.block_info[block_id] = BlockTierInfo(
            block_id=block_id,
            tier=tier,
            last_access_time=time.time(),
            access_count=0,
            size_bytes=0,
            timestamp=time.time(),
        )
        if tier not in self.tier_blocks:
            self.tier_blocks[tier] = []
        self.tier_blocks[tier].append(block_id)

    def on_block_evicted(self, block_id: int, tier: EvictionTier) -> None:
        """Remove from tier tracking on eviction."""
        if block_id in self.block_info:
            del self.block_info[block_id]
        if tier in self.tier_blocks and block_id in self.tier_blocks[tier]:
            self.tier_blocks[tier].remove(block_id)

    def get_next_tier(self, current_tier: EvictionTier) -> Optional[EvictionTier]:
        """Get the next tier in cascade order."""
        try:
            idx = self.tier_order.index(current_tier)
            if idx + 1 < len(self.tier_order):
                return self.tier_order[idx + 1]
        except ValueError:
            pass
        return None


class ParallelEvictionPolicy(EvictionPolicy):
    """Parallel eviction policy: route blocks to best available tier.
    
    Dynamically routes blocks to minimize latency based on current tier utilization
    and block characteristics (recency, importance).
    """

    def __init__(self):
        self.block_info: Dict[int, BlockTierInfo] = {}
        self.tier_blocks: Dict[EvictionTier, List[int]] = {
            tier: [] for tier in EvictionTier
        }
        self.tier_latencies = {
            EvictionTier.GPU: 0.0001,  # ~100 ns
            EvictionTier.CPU: 0.0005,  # ~500 ns
            EvictionTier.OPTANE: 0.0003,  # ~300 ns
            EvictionTier.SSD: 0.005,  # ~5 µs
            EvictionTier.REMOTE: 0.05,  # ~50 µs
        }

    def select_eviction_candidates(
        self, num_candidates: int, tier: EvictionTier
    ) -> List[int]:
        """Select blocks based on parallel policy scoring."""
        blocks_in_tier = self.tier_blocks.get(tier, [])
        if not blocks_in_tier:
            return []

        # Score blocks: lower score = higher priority for eviction
        def score_block(block_id: int) -> float:
            info = self.block_info[block_id]
            # Combine recency and access frequency
            import time

            age = time.time() - info.timestamp
            score = age / (1 + info.access_count)  # Older and less used = higher
            return score

        sorted_blocks = sorted(blocks_in_tier, key=score_block, reverse=True)
        return sorted_blocks[:num_candidates]

    def on_block_accessed(self, block_id: int, tier: EvictionTier) -> None:
        """Update access tracking."""
        import time

        if block_id in self.block_info:
            self.block_info[block_id].last_access_time = time.time()
            self.block_info[block_id].access_count += 1

    def on_block_stored(self, block_id: int, tier: EvictionTier) -> None:
        """Track block storage in tier."""
        import time

        self.block_info[block_id] = BlockTierInfo(
            block_id=block_id,
            tier=tier,
            last_access_time=time.time(),
            access_count=0,
            size_bytes=0,
            timestamp=time.time(),
        )
        if tier not in self.tier_blocks:
            self.tier_blocks[tier] = []
        self.tier_blocks[tier].append(block_id)

    def on_block_evicted(self, block_id: int, tier: EvictionTier) -> None:
        """Remove from tier tracking on eviction."""
        if block_id in self.block_info:
            del self.block_info[block_id]
        if tier in self.tier_blocks and block_id in self.tier_blocks[tier]:
            self.tier_blocks[tier].remove(block_id)

    def get_best_tier(
        self, tier_utilizations: Dict[EvictionTier, float]
    ) -> EvictionTier:
        """Select best tier based on utilization and latency.
        
        Args:
            tier_utilizations: Dict mapping tier to utilization (0.0-1.0)
            
        Returns:
            Best tier for block storage
        """
        scores = {}
        for tier, util in tier_utilizations.items():
            # Lower latency and lower utilization = better
            score = self.tier_latencies[tier] + util
            scores[tier] = score

        return min(scores, key=scores.get)


def create_eviction_policy(policy_name: str) -> EvictionPolicy:
    """Factory function to create eviction policy.
    
    Args:
        policy_name: Name of the policy ('lru', 'lfu', 'cascade', 'parallel')
        
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
            f"Valid policies: {list(policies.keys())}"
        )

    return policies[policy_name]()
