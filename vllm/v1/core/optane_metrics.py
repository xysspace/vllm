# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metrics collection for Optane tier operations."""

from typing import Dict, List
import time
import threading

from vllm.logger import init_logger
from vllm.v1.core.optane_types import (
    EvictionTier,
    LatencyStats,
    OptaneMetricsSummary,
    CompressionStats,
)

logger = init_logger(__name__)


class OptaneMetricsCollector:
    """Collects metrics about Optane tier operations.
    
    Thread-safe metric collection with optional sampling to reduce overhead.
    """

    def __init__(self, sample_rate: float = 0.1):
        """Initialize metrics collector.
        
        Args:
            sample_rate: Fraction of operations to sample (0.0-1.0)
        """
        self.sample_rate = max(0.0, min(sample_rate, 1.0))
        self.lock = threading.RLock()

        # Allocation metrics
        self.total_blocks_allocated = 0
        self.total_blocks_freed = 0
        self.current_blocks_allocated = 0
        self.current_allocated_bytes = 0

        # Tier metrics
        self.blocks_per_tier: Dict[EvictionTier, int] = {tier: 0 for tier in EvictionTier}
        self.bytes_per_tier: Dict[EvictionTier, int] = {tier: 0 for tier in EvictionTier}

        # Access metrics
        self.total_accesses = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.latency_by_tier: Dict[EvictionTier, LatencyStats] = {
            tier: LatencyStats() for tier in EvictionTier
        }

        # Eviction metrics
        self.total_evictions = 0
        self.evictions_by_policy: Dict[str, int] = {}
        self.eviction_latencies: List[float] = []

        # Compression metrics
        self.compression_stats = CompressionStats()

        # Throughput
        self.blocks_processed = 0
        self.start_time = time.time()

    def _should_sample(self) -> bool:
        """Check if this operation should be sampled.
        
        Returns:
            True if should record metrics
        """
        if self.sample_rate >= 1.0:
            return True
        import random
        return random.random() < self.sample_rate

    def record_block_allocated(
        self,
        block_id: int,
        size_bytes: int,
        tier: EvictionTier,
    ) -> None:
        """Record block allocation.
        
        Args:
            block_id: Block ID
            size_bytes: Block size
            tier: Tier allocated in
        """
        if not self._should_sample():
            return

        with self.lock:
            self.total_blocks_allocated += 1
            self.current_blocks_allocated += 1
            self.current_allocated_bytes += size_bytes
            self.blocks_per_tier[tier] += 1
            self.bytes_per_tier[tier] += size_bytes

    def record_block_freed(self, block_id: int, size_bytes: int, tier: EvictionTier) -> None:
        """Record block deallocation.
        
        Args:
            block_id: Block ID
            size_bytes: Block size
            tier: Tier freed from
        """
        if not self._should_sample():
            return

        with self.lock:
            self.total_blocks_freed += 1
            self.current_blocks_allocated = max(0, self.current_blocks_allocated - 1)
            self.current_allocated_bytes = max(0, self.current_allocated_bytes - size_bytes)
            self.blocks_per_tier[tier] = max(0, self.blocks_per_tier[tier] - 1)
            self.bytes_per_tier[tier] = max(0, self.bytes_per_tier[tier] - size_bytes)

    def record_tier_transition(
        self,
        block_id: int,
        from_tier: EvictionTier,
        to_tier: EvictionTier,
        size_bytes: int,
        latency_us: float,
    ) -> None:
        """Record block movement between tiers.
        
        Args:
            block_id: Block ID
            from_tier: Source tier
            to_tier: Target tier
            size_bytes: Block size
            latency_us: Operation latency in microseconds
        """
        if not self._should_sample():
            return

        with self.lock:
            self.blocks_per_tier[from_tier] = max(0, self.blocks_per_tier[from_tier] - 1)
            self.bytes_per_tier[from_tier] = max(0, self.bytes_per_tier[from_tier] - size_bytes)
            self.blocks_per_tier[to_tier] += 1
            self.bytes_per_tier[to_tier] += size_bytes

            # Record latency
            stats = self.latency_by_tier[to_tier]
            stats.count += 1
            stats.total_us += latency_us
            stats.min_us = min(stats.min_us, latency_us)
            stats.max_us = max(stats.max_us, latency_us)

    def record_access(self, block_id: int, latency_us: float, cache_hit: bool = True) -> None:
        """Record block access and latency.
        
        Args:
            block_id: Block ID
            latency_us: Access latency in microseconds
            cache_hit: Whether this was a cache hit
        """
        if not self._should_sample():
            return

        with self.lock:
            self.total_accesses += 1
            if cache_hit:
                self.cache_hit_count += 1
            else:
                self.cache_miss_count += 1

    def record_eviction(
        self,
        block_id: int,
        policy_name: str,
        latency_us: float,
    ) -> None:
        """Record eviction decision.
        
        Args:
            block_id: Block ID
            policy_name: Eviction policy name
            latency_us: Eviction latency
        """
        if not self._should_sample():
            return

        with self.lock:
            self.total_evictions += 1
            self.evictions_by_policy[policy_name] = (
                self.evictions_by_policy.get(policy_name, 0) + 1
            )
            self.eviction_latencies.append(latency_us)

    def record_compression(
        self,
        original_size: int,
        compressed_size: int,
        compression_time_us: float,
    ) -> None:
        """Record compression operation.
        
        Args:
            original_size: Original data size
            compressed_size: Compressed data size
            compression_time_us: Compression time
        """
        if not self._should_sample():
            return

        with self.lock:
            self.compression_stats.total_blocks_compressed += 1
            self.compression_stats.total_bytes_original += original_size
            self.compression_stats.total_bytes_compressed += compressed_size
            self.compression_stats.total_compression_time_us += compression_time_us

    def get_summary(self) -> OptaneMetricsSummary:
        """Get comprehensive metrics summary.
        
        Returns:
            OptaneMetricsSummary with all collected metrics
        """
        with self.lock:
            # Calculate eviction latency percentiles
            eviction_latencies = sorted(self.eviction_latencies)
            eviction_p50 = 0.0
            eviction_p99 = 0.0
            if eviction_latencies:
                eviction_p50 = eviction_latencies[int(len(eviction_latencies) * 0.5)]
                eviction_p99 = eviction_latencies[int(len(eviction_latencies) * 0.99)]

            # Calculate throughput
            elapsed_time = time.time() - self.start_time
            throughput = self.blocks_processed / elapsed_time if elapsed_time > 0 else 0.0

            # Calculate utilization
            total_blocks = self.current_blocks_allocated
            utilization = 0.0  # Could compute from capacity if known

            return OptaneMetricsSummary(
                total_blocks_allocated=self.total_blocks_allocated,
                total_blocks_freed=self.total_blocks_freed,
                current_blocks_allocated=self.current_blocks_allocated,
                current_allocated_bytes=self.current_allocated_bytes,
                blocks_per_tier=self.blocks_per_tier.copy(),
                bytes_per_tier=self.bytes_per_tier.copy(),
                total_accesses=self.total_accesses,
                cache_hit_count=self.cache_hit_count,
                cache_miss_count=self.cache_miss_count,
                latency_by_tier={
                    tier: LatencyStats(
                        count=stats.count,
                        total_us=stats.total_us,
                        min_us=stats.min_us,
                        max_us=stats.max_us,
                    )
                    for tier, stats in self.latency_by_tier.items()
                },
                total_evictions=self.total_evictions,
                evictions_by_policy=self.evictions_by_policy.copy(),
                eviction_latency_p50=eviction_p50,
                eviction_latency_p99=eviction_p99,
                compression_stats=self.compression_stats,
                throughput_blocks_per_sec=throughput,
                memory_utilization=utilization,
            )

    def reset(self) -> None:
        """Reset all metrics."""
        with self.lock:
            self.total_blocks_allocated = 0
            self.total_blocks_freed = 0
            self.current_blocks_allocated = 0
            self.current_allocated_bytes = 0
            self.blocks_per_tier = {tier: 0 for tier in EvictionTier}
            self.bytes_per_tier = {tier: 0 for tier in EvictionTier}
            self.total_accesses = 0
            self.cache_hit_count = 0
            self.cache_miss_count = 0
            self.latency_by_tier = {tier: LatencyStats() for tier in EvictionTier}
            self.total_evictions = 0
            self.evictions_by_policy = {}
            self.eviction_latencies = []
            self.compression_stats = CompressionStats()
            self.blocks_processed = 0
            self.start_time = time.time()
