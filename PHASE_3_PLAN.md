# Phase 3 Plan: Core Implementation

**Status:** Planning Document  
**Builds On:** Phase 2 (Implementation of OptaneConfig and design documentation)  
**Target:** Full runtime integration of Optane into vLLM's KV cache management system

---

## Phase 2 Completion Summary

### ✅ **Phase 2 Delivered**

| Component | Location | Status |
|-----------|----------|--------|
| OptaneConfig | `vllm/config/optane.py` | ✅ Complete |
| Environment Variables | Multiple files | ✅ Complete |
| CLI Arguments | `vllm/entrypoints/cli/serve.py` | ✅ Complete |
| Design Documentation | `docs/design/optane_kv_cache.md` | ✅ Complete |
| Architecture Diagrams | Design docs | ✅ Complete |
| Eviction Policies Spec | Design docs | ✅ Complete |
| Block Lifecycle Design | Design docs | ✅ Complete |

Phase 2 successfully established the configuration layer and comprehensive design documentation for Optane integration. All foundation components are in place and ready for Phase 3 implementation.

---

## Phase 3 Overview

Phase 3 focuses on **implementing the operational core** that enables Optane tier management during inference. This is the critical bridge between configuration (Phase 2) and production deployment (Phase 4+).

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ vLLM Inference Engine                                       │
├─────────────────────────────────────────────────────────────┤
│  Scheduler                                                  │
│    ↓ allocate_slots()                                       │
├─────────────────────────────────────────────────────────────┤
│  KVCacheManager (Main Coordinator) [MODIFIED]              │
│    ├─ BlockPool (GPU memory management)                     │
│    ├─ KVCacheCoordinator (Multi-group coordination)         │
│    ├─⭐ OptaneManager (NEW - Persistent memory)              │
│    ├─⭐ BlockLifecycleManager (NEW)                          │
│    ├─⭐ EvictionPolicyExecutor (NEW)                         │
│    └─⭐ CompressionManager (NEW)                             │
├─────────────────────────────────────────────────────────────┤
│  Memory Hierarchy                                           │
│    GPU (40-80 GB)   ← Hot tier                              │
│    CPU (256 GB)     ← Warm tier                             │
│    ⭐ OPTANE         ← Extended tier (100-1500 GB)           │
│    SSD/Remote       ← Cold tier                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 3 Component Breakdown

### 3.1 OptaneManager - Core Bridge Component

**Location:** `vllm/v1/core/optane_manager.py` (~600 lines)

**Purpose:** 
- Bridge between KVCacheManager and Optane persistent memory backend
- Manage block allocation/deallocation in Optane tier
- Track block metadata and access patterns
- Coordinate tier transitions

**Key Responsibilities:**
- Physical block management in Optane memory
- Tier transition orchestration
- Metadata tracking and updates
- Space utilization monitoring

**Key Methods:**

```python
class OptaneManager:
    """Manages Optane persistent memory tier for KV cache blocks."""
    
    def __init__(self, config: OptaneConfig):
        """Initialize OptaneManager with configuration."""
        
    def allocate_block(self, size_bytes: int) -> OptaneBlockInfo:
        """Allocate a block in Optane memory."""
        
    def free_block(self, block_id: int) -> None:
        """Free an allocated Optane block."""
        
    def evict_to_optane(self, block_id: int, data: bytes) -> None:
        """Evict block data from GPU to Optane."""
        
    def load_from_optane(self, block_id: int) -> bytes:
        """Load block data from Optane to GPU."""
        
    def get_available_space(self) -> int:
        """Get currently available Optane space."""
        
    def update_block_metadata(self, block_id: int, **kwargs) -> None:
        """Update metadata for a block."""
        
    def promote_block(self, block_id: int, target_tier: EvictionTier) -> None:
        """Move block to higher tier (faster memory)."""
        
    def demote_block(self, block_id: int, target_tier: EvictionTier) -> None:
        """Move block to lower tier (more storage)."""
```

**Data Structures:**

```python
@dataclass
class OptaneBlockInfo:
    """Metadata for a block in Optane tier."""
    block_id: int
    physical_offset: int  # Offset in Optane memory
    size_bytes: int
    current_tier: EvictionTier
    compressed: bool
    compression_ratio: float
    access_time: float
    access_count: int
    ref_cnt: int
```

---

### 3.2 EvictionPolicyExecutor - Multi-Strategy Support

**Location:** `vllm/v1/core/eviction_policy_executor.py` (~700 lines)

**Purpose:**
- Implement all four configurable eviction policies
- Make eviction decisions based on workload characteristics
- Track access patterns for intelligent eviction

**Eviction Policies Implemented:**

#### 3.2.1 **LRUEvictionPolicy** - Least Recently Used
```python
class LRUEvictionPolicy(EvictionPolicy):
    """Evict blocks with oldest access timestamp.
    
    Best for:
    - Streaming inference with sequential access
    - Time-based working set patterns
    """
    def get_eviction_candidates(self, num_blocks: int) -> list[BlockId]:
        # Return blocks with oldest access_time
        
    def record_access(self, block_id: int) -> None:
        # Update access_time to current time
```

#### 3.2.2 **LFUEvictionPolicy** - Least Frequently Used
```python
class LFUEvictionPolicy(EvictionPolicy):
    """Evict blocks with lowest access frequency.
    
    Best for:
    - Repetitive inference patterns
    - Working sets with varying importance
    """
    def get_eviction_candidates(self, num_blocks: int) -> list[BlockId]:
        # Return blocks with lowest access_count
        
    def record_access(self, block_id: int) -> None:
        # Increment access_count
```

#### 3.2.3 **CascadeEvictionPolicy** - Strict Hierarchy (Default)
```python
class CascadeEvictionPolicy(EvictionPolicy):
    """Enforce strict tier boundaries: GPU → CPU → Optane → SSD.
    
    Best for:
    - Predictable memory pressure scenarios
    - Production deployments requiring determinism
    """
    def get_eviction_candidates(self, num_blocks: int) -> list[BlockId]:
        # Return blocks for next tier demotion
        
    def should_cascade_to_next_tier(self) -> bool:
        # Determine if current tier is full
```

#### 3.2.4 **ParallelEvictionPolicy** - Adaptive Routing
```python
class ParallelEvictionPolicy(EvictionPolicy):
    """Route blocks to best available tier dynamically.
    
    Best for:
    - Mixed workloads with variable access patterns
    - Maximizing utilization across all tiers
    """
    def get_eviction_candidates(self, num_blocks: int) -> list[BlockId]:
        # Select blocks and target tiers adaptively
        
    def select_best_tier(self, block_id: int) -> EvictionTier:
        # Choose tier based on current utilization
```

**Common Interface:**

```python
class EvictionPolicy(ABC):
    """Abstract base for all eviction policies."""
    
    @abstractmethod
    def get_eviction_candidates(self, num_blocks: int) -> list[BlockId]:
        """Get block IDs to evict."""
        
    @abstractmethod
    def record_access(self, block_id: int) -> None:
        """Record block access for policy tracking."""
        
    def on_block_allocated(self, block_id: int) -> None:
        """Called when a block is allocated."""
        
    def on_block_freed(self, block_id: int) -> None:
        """Called when a block is freed."""
```

---

### 3.3 KVCacheManager Integration

**Location:** Modifications to `vllm/v1/core/kv_cache_manager.py` (~300 lines)

**Integration Points:**

1. **Initialization:**
   ```python
   class KVCacheManager:
       def __init__(self, 
                    kv_cache_config: KVCacheConfig,
                    optane_config: OptaneConfig | None = None,
                    ...):
           self.optane_manager = (
               OptaneManager(optane_config) 
               if optane_config and optane_config.enable_optane 
               else None
           )
           self.eviction_policy = self._create_eviction_policy(
               optane_config.optane_eviction_policy 
               if optane_config else "cascade"
           )
   ```

2. **Slot Allocation:**
   ```python
   def allocate_slots(self, request: Request, ...) -> KVCacheBlocks | None:
       # Consider Optane capacity in allocation decision
       if self.optane_manager:
           available = (
               self.block_pool.get_num_free_blocks() + 
               self.optane_manager.get_available_space() / block_size
           )
       else:
           available = self.block_pool.get_num_free_blocks()
       
       # If GPU full but Optane available, allow allocation
       # (blocks will be evicted to Optane)
       if available > required_blocks:
           # Proceed with allocation
           return self._allocate_with_tier_awareness(request, ...)
       return None
   ```

3. **Eviction Triggering:**
   ```python
   def _trigger_eviction_to_optane(self, num_blocks: int) -> None:
       """Evict blocks from GPU to Optane when needed."""
       candidates = self.eviction_policy.get_eviction_candidates(
           num_blocks
       )
       for block_id in candidates:
           block = self.block_pool.get_block(block_id)
           self.optane_manager.evict_to_optane(block_id, block.data)
           self.block_pool.mark_evicted(block_id)
   ```

4. **Block Retrieval:**
   ```python
   def get_blocks(self, request_id: str) -> KVCacheBlocks:
       """Get blocks, loading from Optane if needed."""
       blocks = self.coordinator.get_blocks(request_id)
       
       # Check if any blocks are in Optane
       for block in blocks:
           if block.current_tier == EvictionTier.OPTANE:
               # Load back to GPU
               self.optane_manager.load_from_optane(block.block_id)
               self.eviction_policy.record_access(block.block_id)
       
       return self.create_kv_cache_blocks(blocks)
   ```

---

### 3.4 Block Lifecycle Management

**Location:** `vllm/v1/core/optane_block_lifecycle.py` (~400 lines)

**Purpose:**
- Manage state transitions for all blocks
- Track block ownership and tier placement
- Coordinate lifecycle across all memory tiers

**Block State Machine:**

```
UNALLOCATED
    ↓
ALLOCATED_GPU (HOT)
    ↓ (optional - on free/eviction)
ALLOCATED_CPU (WARM)
    ↓ (optional - on pressure/eviction)
ALLOCATED_OPTANE (COOL)
    ↓ (optional - on high pressure)
ALLOCATED_SSD (COLD)
    ↓
FREED
```

**State Tracking Structure:**

```python
@dataclass
class BlockState:
    """Complete state for a KV cache block."""
    block_id: int
    current_tier: EvictionTier  # Where the block physically resides
    ref_cnt: int  # Number of requests using this block
    access_time: float  # Last access timestamp
    access_freq: int  # Total access count
    compressed: bool  # Whether data is compressed
    compression_ratio: float  # Bytes saved / original bytes
    data_location: BlockLocation  # Physical location metadata
    owner_requests: set[str]  # Which requests own this block
```

**Key Methods:**

```python
class BlockLifecycleManager:
    """Manages block state transitions across memory tiers."""
    
    def allocate(self, size_bytes: int) -> BlockId:
        """Allocate a new block in GPU memory."""
        
    def promote(self, block_id: int, target_tier: EvictionTier) -> None:
        """Move block to higher-tier memory (faster)."""
        # GPU → keeps in GPU
        # CPU → move to GPU
        # Optane → move to GPU or CPU
        
    def demote(self, block_id: int, target_tier: EvictionTier) -> None:
        """Move block to lower-tier memory (more storage)."""
        # GPU → move to CPU
        # CPU → move to Optane
        # Optane → move to SSD
        
    def free(self, block_id: int) -> None:
        """Free block and reclaim memory at all tiers."""
        
    def touch(self, block_id: int) -> None:
        """Mark block as recently accessed."""
        
    def get_state(self, block_id: int) -> BlockState:
        """Get current state of a block."""
        
    def transition_tier(self, block_id: int, new_tier: EvictionTier) -> None:
        """Execute tier transition with data movement."""
```

---

### 3.5 Compression System

**Location:** `vllm/v1/core/optane_compression.py` (~300 lines)

**Purpose:**
- Compress blocks for Optane storage
- Manage compression/decompression pipeline
- Track compression efficiency

**Compression Strategy:**

```python
class CompressionManager:
    """Manages compression/decompression of KV cache blocks."""
    
    def __init__(self, 
                 enable_compression: bool = True,
                 compression_level: int = 3,
                 compression_algorithm: str = "zstd"):
        
    def compress(self, data: bytes, level: int | None = None) -> CompressedBlock:
        """Compress block data with zstd.
        
        Args:
            data: Uncompressed block data
            level: Compression level (1-9, default: 3)
            
        Returns:
            CompressedBlock with compressed data and metadata
        """
        
    def decompress(self, compressed: CompressedBlock) -> bytes:
        """Decompress block data.
        
        Args:
            compressed: CompressedBlock to decompress
            
        Returns:
            Original uncompressed data
        """
        
    def should_compress(self, block_size: int) -> bool:
        """Determine if block should be compressed.
        
        Heuristic:
        - Always compress blocks > 1MB
        - Never compress blocks < 100KB
        - For blocks in between, compress if compression_ratio > 1.2x
        """
        
    def get_compression_stats(self) -> CompressionStats:
        """Get aggregate compression statistics."""
```

**Data Structures:**

```python
@dataclass
class CompressedBlock:
    """Compressed block with metadata."""
    original_size: int
    compressed_size: int
    compression_time_us: float
    decompression_time_us: float
    data: bytes  # Compressed data
    
    @property
    def compression_ratio(self) -> float:
        """Return compression ratio (original / compressed)."""
        return self.original_size / self.compressed_size if self.compressed_size > 0 else 1.0

@dataclass
class CompressionStats:
    """Aggregate compression statistics."""
    total_blocks_compressed: int
    total_bytes_original: int
    total_bytes_compressed: int
    avg_compression_ratio: float
    avg_compression_time_us: float
    avg_decompression_time_us: float
```

**Compression Benefits:**

| Block Size | Compression Ratio | Space Saved | Use Case |
|-----------|------------------|------------|----------|
| 1 KB (token) | 4-8x | 75-87% | Attention patterns |
| 1 MB (block) | 2-4x | 50-75% | Typical KV cache |
| 10 MB (batch) | 1.5-3x | 33-67% | Large sequences |

---

### 3.6 Metrics Collection Framework

**Location:** `vllm/v1/core/optane_metrics.py` (~250 lines)

**Purpose:**
- Collect comprehensive metrics about Optane operations
- Enable performance monitoring and debugging
- Support production observability

**Metrics Collected:**

```python
class OptaneMetricsCollector:
    """Collects metrics about Optane tier operations."""
    
    def record_block_allocated(self, block_id: int, tier: EvictionTier) -> None:
        """Record block allocation."""
        
    def record_block_freed(self, block_id: int) -> None:
        """Record block deallocation."""
        
    def record_tier_transition(self, 
                               block_id: int, 
                               from_tier: EvictionTier,
                               to_tier: EvictionTier,
                               latency_us: float) -> None:
        """Record block movement between tiers."""
        
    def record_access(self, block_id: int, latency_us: float) -> None:
        """Record block access and latency."""
        
    def record_compression(self, 
                          original_size: int,
                          compressed_size: int,
                          time_us: float) -> None:
        """Record compression operation."""
        
    def record_eviction_decision(self, 
                                policy_name: str,
                                num_candidates: int) -> None:
        """Record eviction policy decision."""
        
    def get_summary(self) -> MetricsSummary:
        """Get comprehensive metrics summary."""
```

**Metrics Types:**

```python
@dataclass
class OptaneMetricsSummary:
    """Summary of all Optane metrics."""
    
    # Allocation metrics
    total_blocks_allocated: int
    total_blocks_freed: int
    current_blocks_allocated: int
    current_allocated_bytes: int
    
    # Tier metrics
    blocks_per_tier: dict[EvictionTier, int]
    bytes_per_tier: dict[EvictionTier, int]
    
    # Access metrics
    total_accesses: int
    cache_hit_count: int
    cache_miss_count: int
    latency_by_tier: dict[EvictionTier, LatencyStats]
    
    # Eviction metrics
    total_evictions: int
    evictions_by_policy: dict[str, int]
    eviction_latency_p50: float
    eviction_latency_p99: float
    
    # Compression metrics
    total_blocks_compressed: int
    total_compression_bytes_saved: int
    avg_compression_ratio: float
    compression_time_total_us: float
    decompression_time_total_us: float
    
    # Performance metrics
    throughput_blocks_per_sec: float
    memory_utilization: float
```

**Sampling Strategy:**

```python
# Reduce overhead with configurable sampling
# Default: sample 10% of operations
if random.random() < self.sample_rate:
    self.metrics.record_access(block_id, latency_us)
```

---

## Phase 3 Integration Points

### Flow Diagram: Request Execution with Optane

```
Request arrives
    ↓
Scheduler calls KVCacheManager.allocate_slots()
    ↓
Check GPU memory available
    ├─ YES → Allocate in GPU
    │         return immediately
    └─ NO → Check Optane available
            ├─ YES → Trigger eviction from GPU to Optane
            │        Allocate in GPU using freed space
            │        EvictionPolicy decides which blocks to evict
            │        OptaneManager.evict_to_optane()
            │        Return allocation
            └─ NO → Check next tier (SSD)
                    └─ If available → similar cascade
                    └─ Reject allocation (return None)
    ↓
Request executes (access blocks in GPU)
    ↓
Upon block access:
    ├─ Block in GPU tier
    │   └─ EvictionPolicy.record_access()
    │      MetricsCollector.record_access()
    ├─ Block in Optane tier (evicted previously)
    │   └─ OptaneManager.load_from_optane()
    │      EvictionPolicy.promote_block() or access from Optane
    │      Update access metrics
    └─ Block in SSD tier
        └─ Similar flow (fetch to CPU or GPU)
    ↓
Request completes
    ↓
Free blocks:
    KVCacheManager.free()
    └─ OptaneManager.free_block() for any Optane blocks
    └─ BlockLifecycleManager.free() to clean up
```

---

## Phase 3 Testing Strategy

### Unit Tests (~500 lines)

```
tests/unit/v1/core/test_optane_manager.py
├─ test_allocate_block()
├─ test_free_block()
├─ test_evict_to_optane()
├─ test_load_from_optane()
└─ test_available_space()

tests/unit/v1/core/test_eviction_policies.py
├─ TestLRUPolicy
│  ├─ test_evict_oldest_accessed()
│  └─ test_update_on_access()
├─ TestLFUPolicy
│  ├─ test_evict_least_used()
│  └─ test_frequency_tracking()
├─ TestCascadePolicy
│  ├─ test_tier_boundaries()
│  └─ test_cascade_to_next_tier()
└─ TestParallelPolicy
   ├─ test_adaptive_tier_selection()
   └─ test_utilization_tracking()

tests/unit/v1/core/test_block_lifecycle.py
├─ test_allocate_state_transition()
├─ test_promote_block()
├─ test_demote_block()
└─ test_free_block()

tests/unit/v1/core/test_optane_compression.py
├─ test_compress_decompress()
├─ test_compression_ratio()
├─ test_compression_threshold()
└─ test_compression_stats()

tests/unit/v1/core/test_optane_metrics.py
├─ test_record_allocation()
├─ test_record_access_latency()
├─ test_metrics_summary()
└─ test_sampling()
```

### Integration Tests (~400 lines)

```
tests/integration/v1/test_kv_cache_optane.py
├─ test_allocate_with_optane_enabled()
├─ test_allocate_without_optane()
├─ test_gpu_memory_pressure_triggers_eviction()
├─ test_multi_tier_block_access()
├─ test_block_lifecycle_across_tiers()
└─ test_compression_with_eviction()

tests/integration/v1/test_optane_end_to_end.py
├─ test_simple_inference_with_optane()
├─ test_batch_inference_with_optane()
├─ test_high_memory_pressure_scenario()
└─ test_mixed_eviction_policies()
```

### Benchmarks (~200 lines)

```
benchmarks/v1/bench_optane_throughput.py
├─ Measure requests/sec with Optane
├─ Compare against baseline (no Optane)
└─ Report memory utilization

benchmarks/v1/bench_optane_latency.py
├─ Latency distribution by block tier
├─ P50, P95, P99 latencies
└─ Eviction overhead

benchmarks/v1/bench_optane_compression.py
├─ Compression ratio by block type
├─ Compression/decompression throughput
└─ CPU overhead
```

---

## Phase 3 Deliverables Checklist

### Core Components
- [ ] OptaneManager implementation (600 lines)
- [ ] EvictionPolicyExecutor with 4 policies (700 lines)
- [ ] BlockLifecycleManager (400 lines)
- [ ] CompressionManager (300 lines)
- [ ] OptaneMetricsCollector (250 lines)

### Integration
- [ ] KVCacheManager modifications (300 lines)
- [ ] Request scheduler integration
- [ ] Block pool integration
- [ ] Config initialization code

### Testing
- [ ] Unit tests (500 lines)
- [ ] Integration tests (400 lines)
- [ ] Benchmark scripts (200 lines)
- [ ] Test data generators

### Documentation
- [ ] Developer guide for Phase 3 components
- [ ] API documentation for all classes
- [ ] Testing guide
- [ ] Debugging guide

---

## Phase 3 Success Criteria

### Functional Requirements
- ✓ Optane tier transparently manages KV cache blocks
- ✓ All four eviction policies functional and working
- ✓ Blocks can be evicted to and restored from Optane
- ✓ Compression/decompression working correctly
- ✓ Metrics accurately track all operations
- ✓ Integration with KVCacheManager seamless

### Performance Requirements
- ✓ < 5% performance overhead for tier transitions
- ✓ Compression achieves 2-4x reduction
- ✓ Compression overhead < 10% CPU utilization
- ✓ Eviction decision latency < 1ms
- ✓ No regression in throughput without Optane

### Quality Requirements
- ✓ 90%+ unit test coverage
- ✓ All benchmarks passing
- ✓ Production-ready error handling
- ✓ Comprehensive logging/metrics
- ✓ No memory leaks (verified with valgrind)

### Code Quality
- ✓ Clean code (follows vLLM style guide)
- ✓ Complete docstrings
- ✓ Type hints throughout
- ✓ Error handling for all edge cases

---

## Phase 3 Timeline Estimate

| Component | Effort | Timeline |
|-----------|--------|----------|
| OptaneManager | 600 lines | 1.5 weeks |
| Eviction Policies | 700 lines | 2 weeks |
| Block Lifecycle | 400 lines | 1 week |
| Compression | 300 lines | 1 week |
| Metrics | 250 lines | 1 week |
| KVCacheManager Integration | 300 lines | 1 week |
| Testing | 1100 lines | 2 weeks |
| Documentation | - | 1 week |
| **Total** | **~3650 lines** | **~11 weeks** |

---

## Phase 3 Dependencies

### Input from Phase 2
- ✓ OptaneConfig complete and working
- ✓ Design documentation comprehensive
- ✓ Architecture validated

### External Dependencies
- vllm v0.6+ with V1 engine
- Python 3.9+
- zstd compression library
- Intel Optane hardware (for production testing)

### Blocking Issues
None identified. Phase 3 can proceed immediately.

---

## Next Steps After Phase 3

Once Phase 3 is complete, proceed to:

### Phase 4: Production Hardening
- Distributed/multi-node support
- Advanced error recovery
- Production monitoring/alerting
- Performance tuning for specific hardware

### Phase 5: Advanced Features
- Monitoring dashboard
- Cost optimization modes
- Advanced scheduling algorithms
- Multi-tenant support

---

## Summary

Phase 3 represents the **core implementation** of Optane integration into vLLM. By implementing OptaneManager, EvictionPolicyExecutor, and related components, vLLM will gain the capability to transparently manage KV cache across multiple memory tiers, significantly extending the effective cache size and enabling longer-sequence inference on more constrained hardware.

The modular design allows for:
- Easy testing and validation of each component
- Flexible eviction policy selection at runtime
- Production-grade metrics and monitoring
- Smooth integration with existing vLLM infrastructure

Phase 3 is estimated to take **~11 weeks** for a single developer or **~6 weeks** for a team of 2-3 engineers.
