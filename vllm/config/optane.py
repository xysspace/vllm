# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import Literal

from pydantic import Field

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

OptaneEvictionPolicy = Literal["lru", "lfu", "cascade", "parallel"]
OptaneBackend = Literal["native", "pmdk"]


@config
class OptaneConfig:
    """Configuration for Optane persistent memory tier in KV cache.
    
    Optane is Intel's persistent memory technology that provides a new memory tier
    between DRAM and SSD. This config enables vLLM to use Optane for extended
    KV cache storage, improving throughput and latency compared to SSD-based
    offloading.
    
    The Optane tier integrates with vLLM's existing tiering (GPU → CPU → Optane → SSD)
    and supports cascaded or parallel eviction policies.
    """

    enable_optane: bool = False
    """Whether to enable Optane tier for KV cache offloading."""

    optane_cache_size: float | None = None
    """Size of Optane cache in GiB. If None, Optane tier is disabled.
    This is the per-rank Optane allocation in distributed setups."""

    optane_mount_path: str = "/mnt/optane"
    """Mount path for DAX-enabled Optane persistent memory device.
    Must be a valid path with read/write permissions.
    Typical: /mnt/optane, /dev/dax0.0, etc."""

    optane_backend: OptaneBackend = "native"
    """Backend for Optane allocation:
    - 'native': Direct mmap-based allocation (simpler, lower overhead)
    - 'pmdk': Intel PMDK library (transactional, crash-safe, higher overhead)
    """

    optane_eviction_policy: OptaneEvictionPolicy = "cascade"
    """Eviction policy for multi-tier memory hierarchy:
    - 'lru': LRU within Optane tier (no cascade)
    - 'lfu': LFU within Optane tier (no cascade)
    - 'cascade': GPU → CPU → Optane → SSD (strict hierarchy)
    - 'parallel': Route blocks to best available tier (adaptive)
    """

    optane_pmdk_pool_path: str | None = None
    """PMDK pool file path for persistent memory allocation.
    Only used when optane_backend='pmdk'. If None, uses in-memory pool."""

    optane_pmdk_pool_size: int = 1024  # MB
    """PMDK pool size in MB. Only used when optane_backend='pmdk'."""

    optane_enable_compression: bool = False
    """Whether to compress KV blocks before storing in Optane.
    Can reduce Optane usage but increases CPU overhead."""

    optane_compression_level: int = 3
    """Compression level (1-9) for block compression.
    Higher = better compression but slower. Only used if compression enabled."""

    optane_block_transfer_batch_size: int = 16
    """Number of blocks to transfer in a single batch operation.
    Larger batches reduce syscall overhead but increase latency variance."""

    optane_prefetch_enabled: bool = True
    """Whether to prefetch blocks from Optane to GPU in background.
    Can improve throughput for streaming inference."""

    optane_prefetch_window_size: int = 4
    """Number of future blocks to prefetch from Optane."""

    optane_enable_metrics: bool = True
    """Whether to collect detailed metrics about Optane usage and performance."""

    optane_metrics_sample_rate: float = 0.1
    """Fraction of block operations to sample for metrics (0.0-1.0).
    Lower values reduce overhead."""

    def compute_hash(self) -> str:
        """Compute a hash that uniquely identifies this config.
        
        Used to detect configuration changes between engine runs.
        Includes all fields that affect Optane tier behavior.
        """
        from vllm.config.utils import hash_factors

        # Factors that affect Optane behavior
        factors = {
            "optane_cache_size": self.optane_cache_size,
            "optane_backend": self.optane_backend,
            "optane_eviction_policy": self.optane_eviction_policy,
            "optane_enable_compression": self.optane_enable_compression,
            "optane_compression_level": self.optane_compression_level,
            "optane_prefetch_enabled": self.optane_prefetch_enabled,
        }
        return hash_factors(factors)

    def is_enabled(self) -> bool:
        """Check if Optane tier is enabled."""
        return self.enable_optane and self.optane_cache_size is not None

    def get_optane_bytes(self) -> int:
        """Get Optane cache size in bytes."""
        if self.optane_cache_size is None:
            return 0
        return int(self.optane_cache_size * 1024 * 1024 * 1024)  # GiB to bytes
