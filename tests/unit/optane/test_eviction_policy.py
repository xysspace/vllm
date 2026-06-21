# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import time

from vllm.core.eviction_policy_optane import (
    EvictionTier,
    LRUEvictionPolicy,
    LFUEvictionPolicy,
    CascadeEvictionPolicy,
    ParallelEvictionPolicy,
    create_eviction_policy,
)


class TestLRUEvictionPolicy:
    """Tests for LRU eviction policy."""

    def test_lru_selection(self):
        """Test LRU candidate selection."""
        policy = LRUEvictionPolicy()

        # Store blocks
        for i in range(3):
            policy.on_block_stored(i, EvictionTier.OPTANE)
            time.sleep(0.01)  # Small delay to differentiate timestamps

        # Get candidates
        candidates = policy.select_eviction_candidates(1, EvictionTier.OPTANE)
        assert len(candidates) == 1
        assert candidates[0] == 0  # Oldest block

    def test_lru_access_update(self):
        """Test LRU update on block access."""
        policy = LRUEvictionPolicy()

        policy.on_block_stored(0, EvictionTier.OPTANE)
        time.sleep(0.01)
        policy.on_block_stored(1, EvictionTier.OPTANE)

        # Access block 0 (update its timestamp)
        policy.on_block_accessed(0, EvictionTier.OPTANE)

        # Block 1 should now be oldest
        candidates = policy.select_eviction_candidates(1, EvictionTier.OPTANE)
        assert candidates[0] == 1


class TestLFUEvictionPolicy:
    """Tests for LFU eviction policy."""

    def test_lfu_selection(self):
        """Test LFU candidate selection."""
        policy = LFUEvictionPolicy()

        policy.on_block_stored(0, EvictionTier.OPTANE)
        policy.on_block_stored(1, EvictionTier.OPTANE)
        policy.on_block_stored(2, EvictionTier.OPTANE)

        # Access block 2 multiple times
        policy.on_block_accessed(2, EvictionTier.OPTANE)
        policy.on_block_accessed(2, EvictionTier.OPTANE)

        # Block 0 should be least frequently used
        candidates = policy.select_eviction_candidates(1, EvictionTier.OPTANE)
        assert candidates[0] == 0


class TestCascadeEvictionPolicy:
    """Tests for cascade eviction policy."""

    def test_cascade_selection(self):
        """Test cascade policy candidate selection."""
        policy = CascadeEvictionPolicy()

        # Store blocks in same tier
        for i in range(3):
            policy.on_block_stored(i, EvictionTier.OPTANE)
            time.sleep(0.01)

        # Get candidates (should be oldest first)
        candidates = policy.select_eviction_candidates(1, EvictionTier.OPTANE)
        assert candidates[0] == 0

    def test_cascade_tier_order(self):
        """Test cascade tier hierarchy."""
        policy = CascadeEvictionPolicy()

        expected_order = [
            EvictionTier.GPU,
            EvictionTier.CPU,
            EvictionTier.OPTANE,
            EvictionTier.SSD,
            EvictionTier.REMOTE,
        ]
        assert policy.tier_order == expected_order

        # Test next tier
        next_tier = policy.get_next_tier(EvictionTier.OPTANE)
        assert next_tier == EvictionTier.SSD

        next_tier = policy.get_next_tier(EvictionTier.REMOTE)
        assert next_tier is None


class TestParallelEvictionPolicy:
    """Tests for parallel eviction policy."""

    def test_parallel_tier_selection(self):
        """Test parallel policy tier selection."""
        policy = ParallelEvictionPolicy()

        # All tiers equally utilized
        utilizations = {
            EvictionTier.GPU: 0.5,
            EvictionTier.CPU: 0.5,
            EvictionTier.OPTANE: 0.5,
            EvictionTier.SSD: 0.5,
        }
        # Should prefer lower latency (GPU)
        best = policy.get_best_tier(utilizations)
        assert best == EvictionTier.GPU

    def test_parallel_utilization_bias(self):
        """Test parallel policy considers utilization."""
        policy = ParallelEvictionPolicy()

        # GPU fully utilized, CPU low
        utilizations = {
            EvictionTier.GPU: 0.9,
            EvictionTier.CPU: 0.1,
            EvictionTier.OPTANE: 0.1,
        }
        # Should prefer CPU (lower latency than Optane, but also lower util than GPU)
        best = policy.get_best_tier(utilizations)
        assert best == EvictionTier.CPU


class TestEvictionPolicyFactory:
    """Tests for eviction policy factory function."""

    def test_create_lru_policy(self):
        """Test creating LRU policy."""
        policy = create_eviction_policy("lru")
        assert isinstance(policy, LRUEvictionPolicy)

    def test_create_lfu_policy(self):
        """Test creating LFU policy."""
        policy = create_eviction_policy("lfu")
        assert isinstance(policy, LFUEvictionPolicy)

    def test_create_cascade_policy(self):
        """Test creating cascade policy."""
        policy = create_eviction_policy("cascade")
        assert isinstance(policy, CascadeEvictionPolicy)

    def test_create_parallel_policy(self):
        """Test creating parallel policy."""
        policy = create_eviction_policy("parallel")
        assert isinstance(policy, ParallelEvictionPolicy)

    def test_invalid_policy(self):
        """Test invalid policy name raises error."""
        with pytest.raises(ValueError):
            create_eviction_policy("invalid_policy")
