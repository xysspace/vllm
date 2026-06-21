# Phase 2 Plan: KV Cache Manager Integration for Optane

**Status:** Planning Document  
**Builds On:** Phase 1 (commit `bac9da9b`)  
**Target:** Full Optane integration into vLLM's KV cache management system

---

## Overview

Phase 2 integrates the Phase 1 Optane components into vLLM's existing KV cache management system. The goal is to enable transparent Optane tier management within the inference pipeline, allowing automatic block promotion/demotion based on memory pressure and access patterns.

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ vLLM Inference Engine                                       │
├─────────────────────────────────────────────────────────────┤
│  Scheduler                                                  │
│    ↓ allocate_slots()                                       │
├─────────────────────────────────────────────────────────────┤
│  KVCacheManager (Main Coordinator)                          │
│    ├─ BlockPool (GPU memory management)                     │
│    ├─ KVCacheCoordinator (Multi-group coordination)         │
│    └─⭐ OptaneManager (NEW - Persistent memory)              │
├─────────────────────────────────────────────────────────────┤
│  Memory Hierarchy                                           │
│    GPU (40-80 GB)   ← Hot tier                              │
│    CPU (256 GB)     ← Warm tier                             │
│    ⭐ OPTANE         ← Extended tier                          │
│    SSD/Remote       ← Cold tier                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 2 Component Breakdown

### 2.1 OptaneManager - New Component

**Location:** `vllm/v1/core/optane_manager.py` (new file, ~500 lines)

**Purpose:** Bridge between KVCacheManager and Phase 1 OptaneBlockPool

```python
class OptaneManager:
    """
    Manages Optane persistent memory tier for KV cache blocks.
    
    Responsibilities:
    1. Track Optane block allocation and usage
    2. Handle block promotion (GPU → Optane) and demotion (Optane → GPU)
    3. Emit KV-events for observability
    4. Manage eviction policy integration
    5. Report metrics and statistics
    """
    
    def __init__(
        self,
        optane_config: OptaneConfig,
        block_size: int,
        max_blocks: int,
        enable_events: bool = False
    ) -> None:
        self.pool = OptaneBlockPool(config, block_size, max_blocks)
        self.eviction_policy = create_eviction_policy(
            optane_config.optane_eviction_policy
        )
        self.enable_events = enable_events
        self.event_queue = OptaneEventQueue() if enable_events else None
    
    def promote_block_to_optane(
        self,
        block_id: int,
        kv_data: torch.Tensor,
        parent_hash: Optional[str] = None
    ) -> bool:
        """
        Move block from GPU/CPU to Optane persistent memory.
        Returns: True if successful, False if capacity insufficient
        """
        
    def demote_block_from_optane(
        self,
        block_id: int,
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Retrieve block from Optane back to GPU/CPU.
        Returns: KV tensor or None if failed
        """
        
    def should_promote_to_optane(
        self,
        gpu_usage_fraction: float,
        optane_usage_fraction: float,
        access_frequency: float
    ) -> bool:
        """
        Determine if blocks should be promoted to Optane.
        Decision logic based on:
        - GPU memory pressure (usage > threshold)
        - Optane availability
        - Block access patterns
        """
        
    def get_eviction_candidates(
        self,
        num_candidates: int
    ) -> List[int]:
        """
        Get candidate blocks from Optane for eviction to SSD.
        Uses configured eviction policy.
        """
```

### 2.2 BlockPool Extensions

**File:** `vllm/v1/core/block_pool.py` (modify existing)

**Changes:**
1. Add OptaneManager reference
2. Extend `_maybe_evict_cached_block()` to consider Optane tier
3. Add `promote_to_optane()` method
4. Add metrics tracking for tier movements

```python
class BlockPool:
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        # NEW PARAMETERS
        optane_config: Optional[OptaneConfig] = None,
        optane_manager: Optional[OptaneManager] = None,
        enable_tier_promotion: bool = False,
        # EXISTING PARAMETERS...
    ):
        self.optane_manager = optane_manager
        self.enable_tier_promotion = enable_tier_promotion
        self.promote_to_optane_threshold = 0.85  # Promote when GPU > 85%
        
    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        MODIFIED: Try to promote to Optane before full eviction.
        
        Promotion logic:
        1. If Optane enabled and has capacity → promote block
        2. Else if Optane full → evict from Optane to SSD
        3. Else → evict from GPU cache normally
        """
        
    def promote_blocks_to_optane(
        self,
        blocks: List[KVCacheBlock],
        tier_usage: float
    ) -> int:
        """
        Promote cold/aged blocks to Optane when GPU pressure high.
        Returns: Number of blocks successfully promoted
        """
```

### 2.3 KVCacheManager Extensions

**File:** `vllm/v1/core/kv_cache_manager.py` (modify existing)

**Changes:**
1. Add OptaneManager initialization
2. Integrate Optane tier checks in `allocate_slots()`
3. Add automatic promotion trigger
4. Expose Optane metrics

```python
class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        # ... existing params ...
        # NEW PARAMETERS
        optane_config: Optional[OptaneConfig] = None,
        enable_optane: bool = False,
    ):
        self.optane_config = optane_config
        if enable_optane and optane_config:
            self.optane_manager = OptaneManager(
                optane_config=optane_config,
                block_size=hash_block_size,
                max_blocks=num_gpu_blocks // 2,  # Conservative estimate
                enable_events=enable_kv_cache_events
            )
        
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        # ... existing params ...
    ) -> Optional[KVCacheBlocks]:
        """
        MODIFIED: Integrate Optane promotion logic.
        
        Before: GPU allocation → evict on pressure
        After:  GPU allocation → try promote to Optane → evict on pressure
        """
        # 1. Try normal GPU allocation
        blocks = self._allocate_gpu_blocks(request, num_new_tokens)
        
        # 2. NEW: If GPU usage high, try Optane promotion
        if self.optane_manager and self.block_pool.usage > 0.85:
            self._try_promote_cold_blocks_to_optane()
        
        # 3. If still no space, evict as normal
        if blocks is None:
            # ... existing eviction logic ...
        
        return blocks
        
    def get_optane_metrics(self) -> Dict[str, Any]:
        """
        Expose Optane tier metrics.
        Returns:
        {
            'blocks_stored': int,
            'blocks_retrieved': int,
            'blocks_evicted': int,
            'avg_latency_us': float,
            'usage_fraction': float,
            'compression_ratio': float
        }
        """
        if self.optane_manager:
            return self.optane_manager.pool.get_metrics()
        return {}
```

### 2.4 Configuration Integration

**File:** `vllm/engine/arg_utils.py` (modify EngineArgs)

**New CLI Arguments:**
```python
class EngineArgs:
    # OPTANE PARAMETERS
    
    @dataclass
    class OptaneArgs:
        """Optane persistent memory configuration."""
        
        enable_optane: bool = False
        """Whether to enable Optane persistent memory tier."""
        
        optane_cache_size_gb: int = 256
        """Size of Optane cache in GB."""
        
        optane_backend: str = "native"
        """Optane backend: 'native' (mmap) or 'pmdk'."""
        
        optane_eviction_policy: str = "cascade"
        """Eviction policy: 'lru', 'lfu', 'cascade', or 'parallel'."""
        
        optane_enable_compression: bool = True
        """Enable compression for Optane blocks."""
        
        optane_compression_level: int = 6
        """zlib compression level (0-9)."""
        
        optane_promotion_threshold: float = 0.85
        """Promote to Optane when GPU usage exceeds this fraction."""
        
        optane_enable_events: bool = False
        """Enable KV-event emission for Optane operations."""
```

---

## Phase 2 Integration Points

### Integration Point 1: Block Eviction Path

**Current Flow:**
```
GPU Full → Select Eviction Candidate → Evict to SSD/Remote
```

**New Flow:**
```
GPU Full → Check Optane Capacity
    ├─ Optane Available → Promote Block
    └─ Optane Full → Select Optane Eviction Candidate → Evict to SSD
```

**Files to Modify:**
- `vllm/v1/core/block_pool.py` → `_maybe_evict_cached_block()`
- `vllm/v1/core/optane_manager.py` → `promote_block_to_optane()`

### Integration Point 2: Block Prefetch/Access

**Current Flow:**
```
Request Needs Block → Get from Cache → Return to Inference Engine
```

**New Flow:**
```
Request Needs Block → Check Location
    ├─ GPU Cache → Return immediately
    ├─ Optane → Demote to GPU + Return
    └─ SSD → Load to GPU + Return
```

**Files to Modify:**
- `vllm/v1/core/kv_cache_manager.py` → `get_blocks()`
- `vllm/v1/core/block_pool.py` → `touch()`

### Integration Point 3: Memory Pressure Response

**Decision Tree:**
```
GPU Usage > Promotion Threshold?
    ├─ YES: Optane Has Capacity?
    │   ├─ YES: Promote Cold Blocks → Return
    │   └─ NO: Evict from Optane → Promote Cold Blocks
    └─ NO: No Action
```

**Files to Modify:**
- `vllm/v1/core/kv_cache_manager.py` → `allocate_slots()`
- `vllm/v1/core/block_pool.py` → `get_new_blocks()`

---

## Data Structures & Changes

### Block Metadata Extension

**Current:**
```python
@dataclass
class KVCacheBlock:
    block_id: int
    block_hash: BlockHash
    ref_cnt: int
    prev_free_block: Optional['KVCacheBlock']
    next_free_block: Optional['KVCacheBlock']
```

**New Fields (for Optane):**
```python
@dataclass
class KVCacheBlock:
    # ... existing fields ...
    
    # NEW FIELDS
    tier_location: MemoryTier = MemoryTier.GPU
    """Where block currently resides: GPU, CPU, OPTANE, SSD"""
    
    tier_promotion_time: float = 0.0
    """Timestamp of last promotion/demotion"""
    
    access_count_in_tier: int = 0
    """Access count since block entered current tier"""
    
    compressed_in_optane: bool = False
    """Whether block is compressed in Optane storage"""
```

**MemoryTier Enum:**
```python
class MemoryTier(Enum):
    GPU = 0
    CPU = 1
    OPTANE = 2
    SSD = 3
    REMOTE = 4
```

---

## Event Flow Example

### Scenario: Block Promotion During High GPU Pressure

```
Time    Event                           Component
────────────────────────────────────────────────────────
T0      GPU usage hits 87%              block_pool.py
T1      _try_promote_cold_blocks()      kv_cache_manager.py
T2      Select block_123 (LRU)          optane_eviction_policy.py
T3      Compress block_123              optane_manager.py
T4      Write to Optane via mmap        block_pool_optane.py
T5      OptaneBlockStoredEvent emitted  kv_events_optane.py
T6      Update block_123.tier_location  kv_cache_block metadata
T7      Mark for CUDA memory free       block_pool.py
T8      Return freed GPU memory         gpu_allocator
```

---

## Testing Strategy for Phase 2

### Unit Tests

**File:** `tests/unit/optane/test_integration_kv_manager.py` (new)
```python
class TestOptaneManagerIntegration:
    def test_promote_block_to_optane(self):
        """Verify block promotion flow"""
        
    def test_demote_block_from_optane(self):
        """Verify block demotion flow"""
        
    def test_promotion_triggered_at_threshold(self):
        """Verify promotion triggers at configured GPU threshold"""
        
    def test_optane_full_triggers_eviction(self):
        """Verify Optane eviction when full"""
        
    def test_event_emission_on_promotion(self):
        """Verify KV-events emitted during promotion"""
```

**File:** `tests/unit/optane/test_block_pool_optane_integration.py` (new)
```python
class TestBlockPoolOptaneIntegration:
    def test_eviction_path_with_optane(self):
        """Verify eviction logic routes through Optane"""
        
    def test_block_pool_usage_tracking(self):
        """Verify tier usage metrics"""
        
    def test_compressed_block_round_trip(self):
        """Verify compression during promotion/demotion"""
```

### Integration Tests

**File:** `tests/v1/optane_inference_e2e_test.py` (new)
```python
class TestOptaneInferenceE2E:
    def test_inference_with_optane_tier(self):
        """Full inference with Optane tier enabled"""
        # 1. Create engine with Optane enabled
        # 2. Run inference on LLM
        # 3. Verify blocks promoted to Optane
        # 4. Verify inference accuracy unchanged
        # 5. Verify performance metrics collected
        
    def test_cache_eviction_with_optane(self):
        """Verify cache eviction respects Optane tier"""
        # 1. Run multiple requests to fill GPU cache
        # 2. Verify blocks promoted to Optane (not SSD)
        # 3. Verify demoted blocks still produce correct results
        
    def test_prefix_cache_with_optane(self):
        """Verify prefix caching works through Optane tier"""
        # 1. Cache prefix blocks
        # 2. Promote some to Optane
        # 3. Run follow-up requests sharing prefix
        # 4. Verify cache hits counted correctly
```

---

## Implementation Sequence

### Week 1: Core Integration
1. Create `OptaneManager` class
2. Modify `BlockPool` to reference OptaneManager
3. Add promotion trigger logic to `KVCacheManager`
4. Write unit tests

### Week 2: CLI & Configuration
1. Add CLI arguments to `EngineArgs`
2. Update engine initialization
3. Add metrics collection
4. Update documentation

### Week 3: Integration Testing
1. Write E2E integration tests
2. Performance benchmarking
3. Bug fixes and refinement
4. Documentation updates

### Week 4: Production Hardening
1. Distributed/multi-GPU support
2. Edge case handling
3. Performance optimization
4. Final validation

---

## Configuration Examples

### Enable Optane with Cascade Policy
```python
from vllm import EngineArgs, LLMEngine

args = EngineArgs(
    model="meta-llama/Llama-2-7b",
    enable_optane=True,
    optane_cache_size_gb=512,
    optane_eviction_policy="cascade",
    optane_compression_level=6
)
engine = LLMEngine.from_engine_args(args)
```

### CLI Usage
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b \
    --enable-optane \
    --optane-cache-size-gb 512 \
    --optane-eviction-policy cascade \
    --optane-enable-compression \
    --optane-compression-level 6
```

---

## Success Criteria

### Functional Requirements
- ✅ Blocks successfully promoted to Optane when GPU > threshold
- ✅ Promoted blocks retrievable for inference without correctness loss
- ✅ Automatic eviction from Optane when full
- ✅ KV-events correctly emitted for all operations
- ✅ All existing vLLM tests still pass

### Performance Requirements
- ✅ < 5% inference throughput degradation vs. GPU-only
- ✅ Optane promotion latency < 1ms per block
- ✅ Optane demotion latency < 2ms per block
- ✅ Memory savings of 20%+ on long-context tasks

### Observability Requirements
- ✅ Metrics exposed via `get_optane_metrics()`
- ✅ KV-events emitted and tracked
- ✅ Integration tests demonstrating end-to-end flow

---

## Known Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| **Performance Regression** | Benchmark regularly, tune promotion threshold |
| **Memory Fragmentation** | Use PMDK backend for better fragmentation handling |
| **Distributed Coordination** | Phase 2.5: Implement cross-GPU Optane sync |
| **Correctness** | Comprehensive unit + integration tests |
| **Production Stability** | Conservative thresholds, gradual rollout |

---

## File List for Phase 2

**New Files:**
- `vllm/v1/core/optane_manager.py` (500 lines)
- `tests/unit/optane/test_integration_kv_manager.py` (250 lines)
- `tests/unit/optane/test_block_pool_optane_integration.py` (200 lines)
- `tests/v1/optane_inference_e2e_test.py` (300 lines)

**Modified Files:**
- `vllm/v1/core/block_pool.py` (+150 lines)
- `vllm/v1/core/kv_cache_manager.py` (+100 lines)
- `vllm/engine/arg_utils.py` (+80 lines)

**Documentation:**
- `docs/optane_setup.md` (new)
- `docs/optane_performance_tuning.md` (new)

**Total Phase 2 Estimate:** ~1,500 lines of new code + ~350 lines modifications

---

## Phase 2 → Phase 3 Preview

Once Phase 2 is complete, Phase 3 could add:
- **Distributed Optane** - Multi-GPU coordination
- **Predictive Prefetching** - Learn access patterns
- **AutoTuning** - Automatic threshold adjustment
- **Cost Modeling** - Latency vs. throughput tradeoffs
- **Production Hardening** - Monitoring, alerting, recovery

---

## Next Steps

1. ✅ Review Phase 2 plan with team
2. ⏳ Begin implementation of OptaneManager
3. ⏳ Create integration test framework
4. ⏳ Performance benchmark baseline
5. ⏳ Gradual rollout to production

**Estimated Timeline:** 4 weeks for full Phase 2 completion

