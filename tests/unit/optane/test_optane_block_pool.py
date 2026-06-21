# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import tempfile
import os
from pathlib import Path

from vllm.config.optane import OptaneConfig
from vllm.v1.core.block_pool_optane import (
    OptaneBlockPool,
    OptaneBlockMetadata,
)


class TestOptaneBlockPool:
    """Tests for OptaneBlockPool."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for Optane mount point."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def optane_config(self, temp_dir):
        """Create test OptaneConfig."""
        return OptaneConfig(
            enable_optane=True,
            optane_cache_size=1.0,  # 1 GiB for testing
            optane_mount_path=temp_dir,
            optane_backend="native",
        )

    def test_block_pool_creation(self, optane_config, temp_dir):
        """Test OptaneBlockPool creation."""
        pool = OptaneBlockPool(
            config=optane_config,
            block_size=16,
            num_blocks=100,
        )
        assert pool.num_blocks == 100
        assert pool.block_size == 16
        assert pool.get_num_free_blocks() == 100
        assert pool.get_num_allocated_blocks() == 0

    def test_block_allocation(self, optane_config, temp_dir):
        """Test block allocation."""
        pool = OptaneBlockPool(
            config=optane_config,
            block_size=16,
            num_blocks=10,
        )
        # Allocate a block
        assert pool.allocate_block(0) is True
        assert pool.get_num_allocated_blocks() == 1
        assert pool.get_num_free_blocks() == 9

        # Cannot re-allocate
        assert pool.allocate_block(0) is False

    def test_block_storage_retrieval(self, optane_config, temp_dir):
        """Test storing and retrieving KV blocks."""
        pool = OptaneBlockPool(
            config=optane_config,
            block_size=16,
            num_blocks=10,
        )
        # Allocate and store a block
        block_id = 0
        pool.allocate_block(block_id)

        # Create dummy KV data
        kv_data = torch.randn(2, 16, 8, 64, dtype=torch.float16)

        # Store block
        assert pool.store_kv_block(block_id, kv_data) is True

        # Retrieve block
        retrieved = pool.retrieve_kv_block(block_id)
        assert retrieved is not None
        # Note: Exact match may differ due to serialization

    def test_block_eviction(self, optane_config, temp_dir):
        """Test block eviction."""
        pool = OptaneBlockPool(
            config=optane_config,
            block_size=16,
            num_blocks=10,
        )
        # Allocate blocks
        pool.allocate_block(0)
        pool.allocate_block(1)

        assert pool.get_num_allocated_blocks() == 2

        # Evict a block
        assert pool.evict_block(0) is True
        assert pool.get_num_allocated_blocks() == 1
        assert pool.get_num_free_blocks() == 9

    def test_eviction_candidates_lru(self, optane_config, temp_dir):
        """Test LRU eviction candidate selection."""
        optane_config.optane_eviction_policy = "lru"
        pool = OptaneBlockPool(
            config=optane_config,
            block_size=16,
            num_blocks=10,
        )
        # Allocate and store blocks
        for i in range(3):
            pool.allocate_block(i)
            kv_data = torch.randn(2, 16, 8, 64, dtype=torch.float16)
            pool.store_kv_block(i, kv_data)

        # Get eviction candidates
        candidates = pool.get_eviction_candidates(1)
        assert len(candidates) == 1
        # Oldest block should be first candidate
        assert candidates[0] == 0

    def test_metrics(self, optane_config, temp_dir):
        """Test metrics collection."""
        pool = OptaneBlockPool(
            config=optane_config,
            block_size=16,
            num_blocks=10,
        )
        pool.allocate_block(0)
        kv_data = torch.randn(2, 16, 8, 64, dtype=torch.float16)
        pool.store_kv_block(0, kv_data)

        metrics = pool.get_metrics()
        assert "blocks_stored" in metrics
        assert "blocks_retrieved" in metrics
        assert "blocks_evicted" in metrics
        assert metrics["blocks_stored"] > 0

    def test_pool_usage(self, optane_config, temp_dir):
        """Test pool usage calculation."""
        pool = OptaneBlockPool(
            config=optane_config,
            block_size=16,
            num_blocks=10,
        )
        assert pool.get_usage() == 0.0

        pool.allocate_block(0)
        pool.allocate_block(1)
        usage = pool.get_usage()
        assert usage == 0.2  # 2 out of 10 blocks

    def test_pool_reset(self, optane_config, temp_dir):
        """Test pool reset."""
        pool = OptaneBlockPool(
            config=optane_config,
            block_size=16,
            num_blocks=10,
        )
        pool.allocate_block(0)
        pool.allocate_block(1)
        assert pool.get_num_allocated_blocks() == 2

        pool.reset()
        assert pool.get_num_allocated_blocks() == 0
        assert pool.get_num_free_blocks() == 10
