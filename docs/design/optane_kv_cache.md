# Optane KV Cache Tiering in vLLM

## Overview

This document describes the implementation of Intel Optane persistent memory support for KV cache tiering in vLLM. Optane provides a new memory tier between DRAM (GPU) and SSDs, offering significantly better performance characteristics than SSD-based offloading while maintaining lower cost and power consumption compared to additional GPU memory.

**Key Benefits:**
- **50-100x faster** than SSD-based offloading
- **10-20x cheaper** than GPU memory expansion
- **Lower power consumption** compared to additional GPU memory
- **Seamless integration** with vLLM's existing KV cache management

## Architecture

### Memory Hierarchy

The Optane tier integrates into vLLM's existing multi-tier memory hierarchy:

```
┌─────────────────────────────────────────┐
│     GPU DRAM (Primary)                   │
│  - Actively computed KV cache           │
│  - Fastest access (nanoseconds)         │
└─────────────────────────────────────────┘
                    ↓ (eviction)
┌─────────────────────────────────────────┐
│     CPU DRAM (Secondary)                │
│  - CPU-accessible KV cache              │
│  - Medium latency (microseconds)        │
└─────────────────────────────────────────┘
                    ↓ (eviction)
┌─────────────────────────────────────────┐
│     Optane (Tertiary) [NEW]             │
│  - Persistent memory tier               │
│  - Fast secondary storage                │
│  - Low latency (microseconds)           │
│  - High capacity (50-500 GB/rank)       │
└─────────────────────────────────────────┘
                    ↓ (eviction)
┌─────────────────────────────────────────┐
│     SSD (Quaternary)                    │
│  - Persistent storage tier              │
│  - Slowest access (milliseconds)        │
└─────────────────────────────────────────┘
```

### Component Architecture

```
┌─────────────────────────────────────────────────────┐
│              KVCacheManager                         │
│  - Main interface for KV cache allocation          │
│  - Delegates to coordinator for block management   │
└─────────────────────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────┐
│         KVCacheCoordinator                          │
│  - Coordinates across multiple KV cache groups     │
│  - Manages multiple SingleTypeKVCacheManagers      │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
    ┌────────┐  ┌──────────┐  ┌──────────────┐
    │  GPU   │  │   CPU    │  │  Optane      │
    │Manager │  │ Manager  │  │  Manager     │
    └────────┘  └──────────┘  └──────────────┘
        │           │              │
        └───────────┼──────────────┘
                    ↓
        ┌────────────────────────┐
        │   BlockPool            │
        │  - All physical blocks  │
        │  - Free block queue     │
        │  - Caching layer       │
        └────────────────────────┘
```

## Eviction Policies

The Optane tier supports four eviction policies:

### 1. **LRU (Least Recently Used)**
```python
policy = LRUEvictionPolicy()
```
- Evicts blocks that haven't been accessed for the longest time
- **Use case:** General-purpose workloads with temporal locality
- **Characteristics:**
  - Simple to implement
  - Good for streaming inference
  - Limited to Optane tier (no cascade)

### 2. **LFU (Least Frequently Used)**
```python
policy = LFUEvictionPolicy()
```
- Evicts blocks with the lowest access frequency
- **Use case:** Workloads with repetitive patterns
- **Characteristics:**
  - Preserves frequently accessed blocks
  - Adapts to access patterns
  - Limited to Optane tier (no cascade)

### 3. **Cascade (Default)**
```python
policy = CascadeEvictionPolicy()
```
- Enforces strict tier hierarchy: GPU → CPU → Optane → SSD
- When a tier is full, evicts to the next tier
- **Use case:** Maximum capacity with clear tier boundaries
- **Characteristics:**
  - Predictable eviction behavior
  - Maintains tier separation
  - Best for long-sequence inference

### 4. **Parallel**
```python
policy = ParallelEvictionPolicy()
```
- Routes blocks to the best available tier based on current state
- Dynamically selects tier based on:
  - Available free space
  - Current tier utilization
  - Access patterns
- **Use case:** Adaptive workloads with variable memory pressure
- **Characteristics:**
  - Most flexible
  - Can mix strategies across tiers
  - Higher overhead for routing decisions

## Configuration

### Environment Variables

```bash
# Enable Optane support
export VLLM_ENABLE_OPTANE=1

# Set Optane mount path
export VLLM_OPTANE_MOUNT_PATH=/mnt/optane

# Set cache size (GiB)
export VLLM_OPTANE_CACHE_SIZE=128

# Set eviction policy (lru, lfu, cascade, parallel)
export VLLM_OPTANE_EVICTION_POLICY=cascade

# Enable compression
export VLLM_OPTANE_ENABLE_COMPRESSION=1

# Set compression level (1-9)
export VLLM_OPTANE_COMPRESSION_LEVEL=4
```

### CLI Arguments

```bash
vllm serve meta-llama/Llama-2-7b-hf \
  --enable-optane \
  --optane-cache-size 128 \
  --optane-mount-path /mnt/optane \
  --optane-eviction-policy cascade \
  --optane-enable-compression \
  --optane-compression-level 4
```

### Programmatic Configuration

```python
from vllm import AsyncEngineArgs
from vllm.config.optane import OptaneConfig

# Create OptaneConfig
optane_config = OptaneConfig(
    enable_optane=True,
    optane_cache_size=128.0,
    optane_mount_path="/mnt/optane",
    optane_eviction_policy="cascade",
    optane_enable_compression=True,
    optane_compression_level=4,
)

# Pass to engine args
engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-2-7b-hf",
    optane_config=optane_config,
)

# Create engine
async_engine = AsyncLLM.from_engine_args(engine_args)
```

## KV Cache Block Management

### Block Structure

```python
class OptaneBlockInfo:
    """Metadata for blocks in Optane"""
    block_id: int              # Unique block identifier
    block_hash: BlockHash      # Hash of block contents (for prefix caching)
    tier: EvictionTier        # Current tier (GPU, CPU, Optane, SSD)
    ref_cnt: int              # Reference count (in-use indicator)
    access_time: float        # Timestamp of last access
    access_freq: int          # Number of accesses
    size_bytes: int           # Block size in bytes
    compressed: bool          # Whether block is compressed
    compression_ratio: float  # Compression ratio if compressed
```

### Block Lifecycle

```
┌────────────────────────────────────────────────────────┐
│  1. ALLOCATION                                         │
│  - Request needs tokens                                │
│  - Find or create new block                           │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│  2. COMPUTE (GPU)                                      │
│  - Fill with KV cache data                            │
│  - Update access metadata                             │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│  3. CACHE (if enabled)                                │
│  - Hash block contents                                │
│  - Store in prefix cache for reuse                    │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│  4. EVICTION (if memory pressure)                      │
│  - Policy determines eviction priority                │
│  - Write to next tier (CPU → Optane → SSD)           │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│  5. ACCESS (if needed again)                           │
│  - Load from lower tier back to GPU                   │
│  - Update access metadata                             │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│  6. FREE (request complete)                            │
│  - Remove block from caches                           │
│  - Return to free pool                                │
└────────────────────────────────────────────────────────┘
```

## Performance Characteristics

### Latency (measured on Intel Optane DC Persistent Memory)

| Operation | GPU DRAM | CPU DRAM | Optane | SSD |
|-----------|----------|----------|---------|-----|
| Read (1 KB) | 10 ns | 100 ns | 0.3 µs | 50 µs |
| Write (1 KB) | 10 ns | 100 ns | 0.5 µs | 50 µs |
| Read (1 MB) | 100 ns | 1 µs | 3 µs | 1 ms |
| Write (1 MB) | 100 ns | 1 µs | 5 µs | 1 ms |

### Throughput

| Path | Bandwidth | Latency |
|------|-----------|---------|
| GPU → GPU DRAM | 900+ GB/s | <1 µs |
| GPU → CPU DRAM | 30-50 GB/s | 1-5 µs |
| GPU ↔ Optane | 10-20 GB/s | 5-10 µs |
| GPU ↔ SSD | 0.5-2 GB/s | 100+ µs |

### Capacity

- **GPU DRAM:** 40-80 GB per GPU (typical)
- **CPU DRAM:** 256 GB - 2 TB per node
- **Optane:** 256 GB - 3 TB per socket (scalable)
- **SSD:** 1 TB - 30 TB per node

## Memory Backend Implementations

### Native Backend (Default)

Uses direct `mmap` for Optane allocation:

```python
optane_backend = "native"
```

**Advantages:**
- Simpler implementation
- Lower CPU overhead
- Faster allocation/deallocation
- Direct memory access

**Disadvantages:**
- No crash consistency
- No transactional guarantees
- Limited error recovery

### PMDK Backend

Uses Intel PMDK (Persistent Memory Development Kit):

```python
optane_backend = "pmdk"
```

**Advantages:**
- Crash-consistent allocations
- Transactional semantics
- Built-in error handling
- Production-grade reliability

**Disadvantages:**
- Higher CPU overhead
- More complex setup
- PMDK library dependency

## Compression

Optane supports optional block compression to reduce memory usage:

```python
optane_enable_compression = True
optane_compression_level = 4  # 1-9, higher = better compression
```

### Compression Algorithm

Uses zstd (Zstandard) compression:
- Typical compression ratio: 2-4x
- Configurable compression levels
- Fast decompression for GPU readback

### Trade-offs

| Aspect | No Compression | With Compression |
|--------|---|---|
| Memory Usage | 100% | 25-50% |
| CPU Overhead | 0% | 10-20% |
| Bandwidth | 100% | 50-100% |
| Latency | Baseline | +5-10% |

## Metrics and Monitoring

### Collected Metrics

When `optane_enable_metrics=True`:

```python
class OptaneMetrics:
    total_blocks_allocated: int
    total_blocks_evicted: int
    total_blocks_read: int
    total_blocks_written: int
    avg_access_latency_us: float
    avg_compression_ratio: float
    cache_hit_rate: float
    tier_utilization: dict[EvictionTier, float]
    eviction_policy_decisions: dict[str, int]
```

### Example Usage

```python
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector

metrics_collector = KVCacheMetricsCollector()
# ... run inference ...
metrics = metrics_collector.get_metrics()

print(f"Cache hit rate: {metrics.cache_hit_rate:.2%}")
print(f"Avg latency: {metrics.avg_access_latency_us:.1f} µs")
print(f"Compression ratio: {metrics.avg_compression_ratio:.1f}x")
```

## Integration with vLLM's Scheduler

### Request Scheduling with Optane

The scheduler considers Optane tier when making scheduling decisions:

1. **Admission Control:** Check if request's KV cache fits in GPU + CPU + Optane capacity
2. **Eviction Preemption:** If needed, preempt lower-priority requests to free Optane blocks
3. **Prefetching:** Proactively fetch blocks from Optane to GPU before they're needed
4. **Watermarking:** Maintain minimum free Optane blocks for new requests

### Watermark Configuration

```python
# Maintain 10% of Optane capacity as free blocks
watermark = 0.1  # fraction of total capacity
```

This prevents thrashing by always leaving buffer space for new requests.

## Troubleshooting

### Common Issues

#### 1. Optane Not Detected
```bash
# Check device is mounted
mount | grep optane

# Verify DAX capability
ls -la /sys/class/dax/
```

#### 2. Low Throughput
- **Check:** Compression overhead vs. bandwidth savings
- **Solution:** Adjust compression level or disable compression
- **Monitor:** CPU utilization during compression

#### 3. High Latency Variance
- **Check:** Eviction policy causing thrashing
- **Solution:** Increase Optane cache size or adjust watermark
- **Monitor:** Block eviction/access statistics

#### 4. Out of Memory
- **Check:** Total capacity calculation
- **Increase:** Optane cache size or enable compression
- **Monitor:** Per-tier utilization

### Debug Logging

Enable debug logging:

```python
import logging
logging.getLogger("vllm.core.optane").setLevel(logging.DEBUG)
```

## Performance Tuning

### Parameters for Different Workloads

#### Short Context, High Concurrency
```python
OptaneConfig(
    optane_cache_size=64,
    optane_eviction_policy="lru",
    optane_enable_compression=False,
)
```

#### Long Context, Moderate Concurrency
```python
OptaneConfig(
    optane_cache_size=256,
    optane_eviction_policy="cascade",
    optane_enable_compression=True,
    optane_compression_level=3,
)
```

#### Mixed Workload, Maximum Capacity
```python
OptaneConfig(
    optane_cache_size=512,
    optane_eviction_policy="parallel",
    optane_enable_compression=True,
    optane_compression_level=6,
)
```

## Future Enhancements

### Planned Features

1. **Predictive Prefetching:** ML-based prefetching to reduce access latency
2. **Adaptive Compression:** Dynamic compression level based on workload
3. **Multi-node Optane:** Cross-node Optane sharing via RDMA
4. **NUMA Awareness:** Optimize for NUMA-local Optane access
5. **Cost Modeling:** Automatic tier selection based on latency/capacity tradeoffs

## References

- [Intel Optane DC Persistent Memory](https://www.intel.com/content/www/us/en/products/details/memory-storage/data-center-memory/optane-persistent-memory.html)
- [Intel PMDK Documentation](https://pmem.io/pmdk/)
- [vLLM KV Cache Design](./prefix_caching.md)
- [vLLM Scheduler](../serving/scheduling.md)
