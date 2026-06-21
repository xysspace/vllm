# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Type definitions for Optane tier management."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class EvictionTier(Enum):
    """Memory tier hierarchy for KV cache."""
    GPU = auto()          # NVIDIA/AMD VRAM - Fastest, smallest
    CPU = auto()          # System RAM - Medium speed, medium size
    OPTANE = auto()        # Intel Optane - Slower than RAM, larger capacity
    SSD = auto()           # NVMe SSD - Slow, very large capacity
    REMOTE = auto()        # Remote storage/network - Slowest, unlimited

    @property
    def is_volatile(self) -> bool:
        """Return True if tier contents lost on power loss."""
        return self in (EvictionTier.GPU, EvictionTier.CPU)

    @property
    def bandwidth_gbps(self) -> float:
        """Approximate bandwidth in GB/s for each tier."""
        bandwidths = {
            EvictionTier.GPU: 900.0,      # H100: ~900 GB/s
            EvictionTier.CPU: 100.0,      # DDR5: ~100 GB/s
            EvictionTier.OPTANE: 30.0,    # Optane DC: ~30 GB/s
            EvictionTier.SSD: 3.0,        # NVMe: ~3 GB/s
            EvictionTier.REMOTE: 0.1,     # Network: ~100 MB/s
        }
        return bandwidths.get(self, 0.1)


@dataclass
class OptaneBlockInfo:
    """Metadata for a block stored in Optane tier."""
    block_id: int
    physical_offset: int        # Offset in Optane memory
    size_bytes: int             # Uncompressed size
    compressed_size_bytes: int  # Actual storage size
    current_tier: EvictionTier  # Where block physically resides
    compressed: bool            # Whether data is compressed
    compression_ratio: float    # original_size / compressed_size
    access_time: float          # Last access timestamp
    access_count: int           # Total accesses
    ref_cnt: int                # Reference count
    owner_requests: set = field(default_factory=set)  # Request IDs

    @property
    def is_in_optane(self) -> bool:
        """Check if block is physically in Optane tier."""
        return self.current_tier == EvictionTier.OPTANE

    @property
    def space_saved_bytes(self) -> int:
        """Bytes saved due to compression."""
        if self.compressed:
            return self.size_bytes - self.compressed_size_bytes
        return 0


@dataclass
class BlockState:
    """Complete state for a KV cache block across tiers."""
    block_id: int
    current_tier: EvictionTier
    ref_cnt: int
    access_time: float
    access_freq: int
    compressed: bool
    compression_ratio: float = 1.0
    data_location: Optional[str] = None  # Backend-specific location
    owner_requests: set = field(default_factory=set)


@dataclass
class CompressedBlock:
    """Compressed block with metadata."""
    original_size: int
    compressed_size: int
    compression_time_us: float
    decompression_time_us: float
    data: bytes                 # Compressed data
    compression_algorithm: str = "zstd"

    @property
    def compression_ratio(self) -> float:
        """Compression ratio (original / compressed)."""
        if self.compressed_size > 0:
            return self.original_size / self.compressed_size
        return 1.0

    @property
    def space_saved_bytes(self) -> int:
        """Bytes saved due to compression."""
        return self.original_size - self.compressed_size


@dataclass
class CompressionStats:
    """Aggregate compression statistics."""
    total_blocks_compressed: int = 0
    total_bytes_original: int = 0
    total_bytes_compressed: int = 0
    total_compression_time_us: float = 0.0
    total_decompression_time_us: float = 0.0
    max_compression_ratio: float = 0.0
    min_compression_ratio: float = float('inf')

    @property
    def avg_compression_ratio(self) -> float:
        """Average compression ratio across all blocks."""
        if self.total_blocks_compressed == 0:
            return 1.0
        if self.total_bytes_compressed == 0:
            return 1.0
        return self.total_bytes_original / self.total_bytes_compressed

    @property
    def total_bytes_saved(self) -> int:
        """Total bytes saved due to compression."""
        return self.total_bytes_original - self.total_bytes_compressed

    @property
    def avg_compression_time_us(self) -> float:
        """Average compression time per block."""
        if self.total_blocks_compressed == 0:
            return 0.0
        return self.total_compression_time_us / self.total_blocks_compressed

    @property
    def avg_decompression_time_us(self) -> float:
        """Average decompression time per block."""
        if self.total_blocks_compressed == 0:
            return 0.0
        return self.total_decompression_time_us / self.total_blocks_compressed


@dataclass
class LatencyStats:
    """Latency statistics for operations."""
    count: int = 0
    total_us: float = 0.0
    min_us: float = float('inf')
    max_us: float = 0.0
    p50_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0

    @property
    def avg_us(self) -> float:
        """Average latency."""
        if self.count == 0:
            return 0.0
        return self.total_us / self.count


@dataclass
class OptaneMetricsSummary:
    """Summary of all Optane tier metrics."""
    # Allocation metrics
    total_blocks_allocated: int = 0
    total_blocks_freed: int = 0
    current_blocks_allocated: int = 0
    current_allocated_bytes: int = 0

    # Tier metrics
    blocks_per_tier: dict = field(default_factory=dict)  # EvictionTier -> count
    bytes_per_tier: dict = field(default_factory=dict)   # EvictionTier -> bytes

    # Access metrics
    total_accesses: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    latency_by_tier: dict = field(default_factory=dict)  # EvictionTier -> LatencyStats

    # Eviction metrics
    total_evictions: int = 0
    evictions_by_policy: dict = field(default_factory=dict)  # policy_name -> count
    eviction_latency_p50: float = 0.0
    eviction_latency_p99: float = 0.0

    # Compression metrics
    compression_stats: CompressionStats = field(default_factory=CompressionStats)

    # Performance metrics
    throughput_blocks_per_sec: float = 0.0
    memory_utilization: float = 0.0  # fraction 0.0-1.0
