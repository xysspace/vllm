# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm.config.optane import OptaneConfig, OptaneEvictionPolicy, OptaneBackend


class TestOptaneConfig:
    """Tests for OptaneConfig."""

    def test_default_config(self):
        """Test default OptaneConfig creation."""
        config = OptaneConfig()
        assert config.enable_optane is False
        assert config.optane_cache_size is None
        assert config.optane_mount_path == "/mnt/optane"
        assert config.optane_backend == "native"
        assert config.optane_eviction_policy == "cascade"

    def test_enabled_config(self):
        """Test enabled OptaneConfig."""
        config = OptaneConfig(
            enable_optane=True,
            optane_cache_size=512.0,
        )
        assert config.is_enabled() is True
        assert config.get_optane_bytes() == 512 * 1024 * 1024 * 1024

    def test_disabled_config(self):
        """Test disabled OptaneConfig."""
        config = OptaneConfig(enable_optane=False)
        assert config.is_enabled() is False
        assert config.get_optane_bytes() == 0

    def test_eviction_policies(self):
        """Test all eviction policy options."""
        for policy in ["lru", "lfu", "cascade", "parallel"]:
            config = OptaneConfig(optane_eviction_policy=policy)
            assert config.optane_eviction_policy == policy

    def test_backends(self):
        """Test all backend options."""
        for backend in ["native", "pmdk"]:
            config = OptaneConfig(optane_backend=backend)
            assert config.optane_backend == backend

    def test_compute_hash(self):
        """Test hash computation for config."""
        config1 = OptaneConfig(
            enable_optane=True,
            optane_cache_size=512.0,
        )
        config2 = OptaneConfig(
            enable_optane=True,
            optane_cache_size=512.0,
        )
        # Same configs should have same hash
        assert config1.compute_hash() == config2.compute_hash()

        config3 = OptaneConfig(
            enable_optane=True,
            optane_cache_size=256.0,  # Different size
        )
        # Different configs should have different hash
        assert config1.compute_hash() != config3.compute_hash()

    def test_compression_settings(self):
        """Test compression configuration."""
        config = OptaneConfig(
            enable_optane=True,
            optane_cache_size=512.0,
            optane_enable_compression=True,
            optane_compression_level=6,
        )
        assert config.optane_enable_compression is True
        assert config.optane_compression_level == 6

    def test_prefetch_settings(self):
        """Test prefetch configuration."""
        config = OptaneConfig(
            enable_optane=True,
            optane_cache_size=512.0,
            optane_prefetch_enabled=True,
            optane_prefetch_window_size=8,
        )
        assert config.optane_prefetch_enabled is True
        assert config.optane_prefetch_window_size == 8
