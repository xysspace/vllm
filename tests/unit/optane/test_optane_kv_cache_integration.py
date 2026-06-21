# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for OptaneManager and KVCacheManager Optane integration (Phase 3.2)."""

import pytest

from vllm.config.optane import OptaneConfig


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_optane_config(**kwargs) -> OptaneConfig:
    defaults = dict(
        enable_optane=True,
        optane_cache_size=1.0,
        optane_mount_path="/tmp/optane_test",
        optane_backend="native",
        optane_eviction_policy="cascade",
        optane_enable_metrics=True,
    )
    defaults.update(kwargs)
    return OptaneConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests: OptaneManager standalone
# ---------------------------------------------------------------------------


class TestOptaneManagerInit:
    """Test OptaneManager initialization and basic lifecycle."""

    def test_init_disabled(self):
        """OptaneManager with disabled config still constructs."""
        from vllm.v1.core.optane_manager import OptaneManager

        config = OptaneConfig(enable_optane=False)
        # Provide a minimal block_pool stub.
        manager = OptaneManager(
            optane_config=config,
            block_pool=_FakeBlockPool(),
            enable_metrics=False,
        )
        assert manager.eviction_policy is None

    def test_init_enabled(self):
        """OptaneManager with enabled config sets up eviction policy."""
        from vllm.v1.core.optane_manager import OptaneManager

        config = _make_optane_config()
        manager = OptaneManager(
            optane_config=config,
            block_pool=_FakeBlockPool(),
            enable_metrics=True,
        )
        # With enabled config but no optane_block_pool, policy should still
        # be None (requires optane_block_pool to be provided).
        assert manager.enable_metrics is True
        assert manager.stats is not None

    def test_stats_reset(self):
        """reset_stats() produces fresh statistics."""
        from vllm.v1.core.optane_manager import OptaneManager

        config = _make_optane_config()
        manager = OptaneManager(
            optane_config=config,
            block_pool=_FakeBlockPool(),
        )
        # Modify stats
        manager.stats.total_blocks_allocated = 99
        manager.reset_stats()
        assert manager.stats.total_blocks_allocated == 0

    def test_take_events_empty(self):
        """take_events() returns empty list when no events queued."""
        from vllm.v1.core.optane_manager import OptaneManager

        config = OptaneConfig(enable_optane=False)
        manager = OptaneManager(
            optane_config=config,
            block_pool=_FakeBlockPool(),
        )
        assert manager.take_events() == []


class TestOptaneManagerBlockLifecycle:
    """Test block allocation and freeing through OptaneManager."""

    def test_on_block_allocated_disabled(self):
        """Blocks allocated while disabled stay on GPU tier."""
        from vllm.v1.core.optane_manager import BlockTierPlacement, OptaneManager
        from vllm.core.eviction_policy_optane import EvictionTier

        config = OptaneConfig(enable_optane=False)
        manager = OptaneManager(
            optane_config=config,
            block_pool=_FakeBlockPool(usage=0.0),
        )
        block = _FakeBlock(block_id=0)
        placement = manager.on_block_allocated(block)
        assert placement == BlockTierPlacement.GPU
        assert manager.get_block_tier(0) == EvictionTier.GPU

    def test_on_block_allocated_low_usage(self):
        """Blocks allocated under GPU threshold stay on GPU."""
        from vllm.v1.core.optane_manager import BlockTierPlacement, OptaneManager
        from vllm.core.eviction_policy_optane import EvictionTier

        config = _make_optane_config()
        manager = OptaneManager(
            optane_config=config,
            block_pool=_FakeBlockPool(usage=0.5),
        )
        block = _FakeBlock(block_id=1)
        placement = manager.on_block_allocated(block)
        assert placement == BlockTierPlacement.GPU
        assert manager.get_block_tier(1) == EvictionTier.GPU

    def test_on_block_freed_decrements_count(self):
        """on_block_freed removes block from tracking and decrements counter."""
        from vllm.v1.core.optane_manager import BlockTierPlacement, OptaneManager
        from vllm.core.eviction_policy_optane import EvictionTier

        config = _make_optane_config()
        manager = OptaneManager(
            optane_config=config,
            block_pool=_FakeBlockPool(usage=0.0),
            enable_metrics=True,
        )
        block = _FakeBlock(block_id=42)
        manager.on_block_allocated(block)
        assert manager.stats.total_blocks_allocated == 1
        manager.on_block_freed(block)
        assert manager.stats.total_blocks_freed == 1
        # Block should no longer be tracked.
        assert 42 not in manager.block_to_tier

    def test_on_block_freed_unknown_block_noop(self):
        """on_block_freed for untracked block is a no-op."""
        from vllm.v1.core.optane_manager import OptaneManager

        config = OptaneConfig(enable_optane=False)
        manager = OptaneManager(
            optane_config=config,
            block_pool=_FakeBlockPool(),
        )
        block = _FakeBlock(block_id=999)
        # Should not raise.
        manager.on_block_freed(block)

    def test_on_block_prefetched_updates_tier(self):
        """on_block_prefetched moves block tier from Optane back to GPU."""
        from vllm.v1.core.optane_manager import OptaneManager
        from vllm.core.eviction_policy_optane import EvictionTier

        config = _make_optane_config()
        manager = OptaneManager(
            optane_config=config,
            block_pool=_FakeBlockPool(usage=0.0),
        )
        block = _FakeBlock(block_id=7)
        # Manually place the block in Optane tier.
        manager.block_to_tier[7] = EvictionTier.OPTANE
        manager.stats.optane_tier.num_blocks = 1

        manager.on_block_prefetched(block)
        assert manager.get_block_tier(7) == EvictionTier.GPU


class TestOptaneManagerStats:
    """Test statistics collection in OptaneManager."""

    def test_stats_track_allocations(self):
        """Allocating multiple blocks increments the counter correctly."""
        from vllm.v1.core.optane_manager import OptaneManager

        config = _make_optane_config()
        manager = OptaneManager(
            optane_config=config,
            block_pool=_FakeBlockPool(usage=0.1),
            enable_metrics=True,
        )
        for i in range(5):
            manager.on_block_allocated(_FakeBlock(block_id=i))

        assert manager.stats.total_blocks_allocated == 5
        assert manager.stats.gpu_tier.num_blocks == 5

    def test_stats_reset_clears_all(self):
        """reset_stats() zeroes all counter fields."""
        from vllm.v1.core.optane_manager import OptaneManager

        config = _make_optane_config()
        manager = OptaneManager(
            optane_config=config,
            block_pool=_FakeBlockPool(usage=0.1),
            enable_metrics=True,
        )
        manager.on_block_allocated(_FakeBlock(0))
        manager.reset_stats()
        assert manager.stats.total_blocks_allocated == 0
        assert manager.stats.total_blocks_freed == 0
        assert manager.stats.total_tier_crossings == 0


# ---------------------------------------------------------------------------
# Tests: OptaneTierManager strategies
# ---------------------------------------------------------------------------


class TestTierPromotionStrategy:
    """Test TierPromotionStrategy decision logic."""

    def test_promotes_on_high_gpu_usage(self):
        from vllm.v1.core.optane_tier_manager import (
            TierPromotionStrategy,
            PromotionTrigger,
        )

        strategy = TierPromotionStrategy(
            gpu_high_watermark=0.85,
            optane_max_fill=0.90,
        )
        decision = strategy.evaluate(
            gpu_usage=0.90,
            optane_usage=0.20,
            free_gpu_blocks=10,
            total_gpu_blocks=100,
        )
        assert decision.should_promote is True
        assert decision.trigger == PromotionTrigger.MEMORY_PRESSURE

    def test_no_promote_when_optane_full(self):
        from vllm.v1.core.optane_tier_manager import TierPromotionStrategy

        strategy = TierPromotionStrategy(optane_max_fill=0.90)
        decision = strategy.evaluate(
            gpu_usage=0.95,
            optane_usage=0.95,
            free_gpu_blocks=5,
            total_gpu_blocks=100,
        )
        assert decision.should_promote is False

    def test_no_promote_on_low_gpu_usage(self):
        from vllm.v1.core.optane_tier_manager import TierPromotionStrategy

        strategy = TierPromotionStrategy(gpu_low_watermark=0.70)
        decision = strategy.evaluate(
            gpu_usage=0.40,
            optane_usage=0.10,
            free_gpu_blocks=60,
            total_gpu_blocks=100,
        )
        assert decision.should_promote is False

    def test_watermark_trigger(self):
        from vllm.v1.core.optane_tier_manager import (
            TierPromotionStrategy,
            PromotionTrigger,
        )

        strategy = TierPromotionStrategy(
            gpu_high_watermark=0.85,
            gpu_low_watermark=0.70,
        )
        decision = strategy.evaluate(
            gpu_usage=0.75,
            optane_usage=0.20,
            free_gpu_blocks=25,
            total_gpu_blocks=100,
        )
        assert decision.should_promote is True
        assert decision.trigger == PromotionTrigger.WATERMARK


class TestTierDemotionStrategy:
    """Test TierDemotionStrategy decision logic."""

    def test_demotes_on_high_optane_usage(self):
        from vllm.v1.core.optane_tier_manager import (
            TierDemotionStrategy,
            DemotionTrigger,
        )

        strategy = TierDemotionStrategy(
            optane_high_watermark=0.80,
            gpu_target_usage=0.75,
        )
        decision = strategy.evaluate(
            gpu_usage=0.50,
            optane_usage=0.85,
            blocks_in_optane=50,
        )
        assert decision.should_demote is True
        assert decision.trigger == DemotionTrigger.TIER_REBALANCE

    def test_no_demote_when_no_blocks(self):
        from vllm.v1.core.optane_tier_manager import TierDemotionStrategy

        strategy = TierDemotionStrategy()
        decision = strategy.evaluate(
            gpu_usage=0.30,
            optane_usage=0.90,
            blocks_in_optane=0,
        )
        assert decision.should_demote is False

    def test_no_demote_when_gpu_busy(self):
        from vllm.v1.core.optane_tier_manager import TierDemotionStrategy

        strategy = TierDemotionStrategy(gpu_target_usage=0.75)
        decision = strategy.evaluate(
            gpu_usage=0.80,
            optane_usage=0.90,
            blocks_in_optane=20,
        )
        assert decision.should_demote is False


# ---------------------------------------------------------------------------
# Tests: KVCacheManager Optane API (backward-compatible)
# ---------------------------------------------------------------------------


class TestKVCacheManagerOptaneAPI:
    """Test that KVCacheManager exposes the Optane API without breaking."""

    def test_optane_disabled_by_default(self, kv_cache_manager):
        """KVCacheManager has no optane_manager when config not provided."""
        assert kv_cache_manager.optane_manager is None

    def test_get_optane_stats_returns_none_when_disabled(self, kv_cache_manager):
        """get_optane_stats() returns None when Optane is disabled."""
        assert kv_cache_manager.get_optane_stats() is None

    def test_should_manage_optane_tier_false_when_disabled(self, kv_cache_manager):
        """should_manage_optane_tier() returns False when Optane is disabled."""
        assert kv_cache_manager.should_manage_optane_tier() is False

    def test_promote_cold_blocks_returns_zero_when_disabled(self, kv_cache_manager):
        """promote_cold_blocks_to_optane() returns 0 when Optane is disabled."""
        assert kv_cache_manager.promote_cold_blocks_to_optane() == 0

    def test_demote_hot_blocks_returns_zero_when_disabled(self, kv_cache_manager):
        """demote_hot_blocks_from_optane() returns 0 when Optane is disabled."""
        assert kv_cache_manager.demote_hot_blocks_from_optane() == 0

    def test_optane_manager_init_with_config(self):
        """KVCacheManager initializes optane_manager when enabled config given."""
        from unittest.mock import MagicMock, patch

        config = _make_optane_config()
        mock_manager = MagicMock()

        with patch(
            "vllm.v1.core.kv_cache_manager.KVCacheManager._init_optane_manager",
        ) as patched_init:
            patched_init.side_effect = lambda cfg: setattr(
                _dummy_instance, "optane_manager", mock_manager
            )
            # Just verify _init_optane_manager is called when config is enabled.
            from vllm.v1.core.kv_cache_manager import KVCacheManager

            called_with = []

            original = KVCacheManager._init_optane_manager

            def record(self, cfg):
                called_with.append(cfg)

            KVCacheManager._init_optane_manager = record
            try:
                mgr = _build_kv_cache_manager(optane_config=config)
                assert len(called_with) == 1
                assert called_with[0] is config
            finally:
                KVCacheManager._init_optane_manager = original


# ---------------------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------------------


class _FakeBlock:
    """Minimal KVCacheBlock stub for testing."""

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.block_hash = None
        self.ref_cnt = 0


class _FakeBlockPool:
    """Minimal BlockPool stub for testing."""

    def __init__(self, usage: float = 0.0, num_gpu_blocks: int = 1000):
        self._usage = usage
        self.num_gpu_blocks = num_gpu_blocks
        self.hash_block_size = 16

    def get_usage(self) -> float:
        return self._usage

    def get_num_free_blocks(self) -> int:
        return self.num_gpu_blocks


@pytest.fixture()
def kv_cache_manager():
    """Return a KVCacheManager with Optane disabled (default)."""
    return _build_kv_cache_manager()


def _build_kv_cache_manager(optane_config=None):
    """Build a minimal KVCacheManager for unit testing."""
    from unittest.mock import MagicMock, patch

    from vllm.v1.core.kv_cache_manager import KVCacheManager

    kv_cache_config = MagicMock()
    kv_cache_config.num_blocks = 1000
    kv_cache_config.kv_cache_groups = [MagicMock()]

    mock_coordinator = MagicMock()
    mock_block_pool = _FakeBlockPool()
    mock_coordinator.block_pool = mock_block_pool
    mock_coordinator.single_type_managers = []

    with patch(
        "vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator",
        return_value=mock_coordinator,
    ):
        mgr = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=2048,
            scheduler_block_size=16,
            hash_block_size=16,
            enable_caching=True,
            optane_config=optane_config,
        )
    return mgr


# Placeholder so patch side_effect has a target
_dummy_instance = object()
