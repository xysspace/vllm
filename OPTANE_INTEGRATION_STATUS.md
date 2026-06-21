# Optane Integration Status Report - Phase 3.3

**Current Status**: Integration of Optane with KVCacheManager and Scheduler in progress

**Last Updated**: 2026-06-21

---

## Phase Summary

### Phase 1: Core Optane Manager ✅ COMPLETE
- `OptaneManager` base class with pluggable policy support
- LRU and ARC eviction policies implemented
- Block allocation, freeing, and promotion/demotion logic
- Event emission for KV transfer

### Phase 2: Optane Manager Bug Fixes ✅ COMPLETE
- Fixed `get_candidates()` → `select_eviction_candidates(num_candidates, tier)` method call
- Added `hasattr()` guard for `get_next_tier()` (only CascadeEvictionPolicy has it)
- Moved `stats.total_blocks_allocated += 1` from unreachable code into both allocation branches
- Removed unused `OptaneBlockMetadata` import

### Phase 3.1: KVCacheManager Integration ✅ COMPLETE
- Added optional `optane_config` parameter (fully backward compatible)
- Wrapped `OptaneManager` instantiation with try-except
- Implemented hooks:
  - `KVCacheManager.allocate_slots()`: Routes block allocation to Optane when enabled
  - `KVCacheManager.free()`: Defers block freeing to Optane manager
  - New method: `get_tier_stats()` for observability

### Phase 3.2: Scheduler Integration 🔄 IN PROGRESS
- Created new branch `phase-3.3-scheduler-integration`
- Core changes needed in `Scheduler.__init__()`:
  1. Accept optional `optane_config` parameter
  2. Pass it to `KVCacheManager` initialization
  3. Initialize stats collector if Optane enabled

### Phase 3.3: Metrics & Observability ✅ JUST COMPLETED
- Created `vllm/v1/metrics/optane_stats.py`:
  - `OptaneTierUtilization`: Track block distribution (GPU/Optane/CPU)
  - `OptaneTierTransition`: Individual migration event recording
  - `OptaneTierStats`: Per-step statistics snapshot
  - `OptaneStatsCollector`: Metrics collection, aggregation, history
  - `OptanePerformanceMetrics`: Performance impact analysis
  - `OptaneHealthReport`: System health monitoring and recommendations

---

## Current Status: Phase 3.3 Completion

### ✅ Completed Files
1. **vllm/v1/core/optane_manager.py** - Core Optane memory manager
2. **vllm/v1/core/eviction_policy_optane.py** - LRU and ARC policies
3. **vllm/v1/core/kv_cache_manager.py** - Integrated Optane hook
4. **vllm/v1/metrics/optane_stats.py** - Metrics and observability (NEW)

### 🔄 Work In Progress

#### Scheduler Integration (`vllm/v1/core/sched/scheduler.py`)
Need to modify `Scheduler.__init__()` to:
```python
def __init__(
    self,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    structured_output_manager: StructuredOutputManager,
    block_size: int,
    hash_block_size: int | None = None,
    mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    include_finished_set: bool = False,
    log_stats: bool = False,
    optane_config: OptaneConfig | None = None,  # ← NEW PARAMETER
) -> None:
    # ... existing initialization ...
    
    # NEW: Pass optane_config to KVCacheManager
    self.kv_cache_manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        # ... other args ...
        optane_config=optane_config,
    )
```

#### EngineCore Integration (`vllm/v1/engine/core.py`)
Need to:
1. Accept `optane_config` in `EngineCore.__init__()`
2. Pass it to `Scheduler.__init__()`
3. Update `step()` method to collect and report Optane stats

---

## Architecture: Tier Management Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     SCHEDULER STEP                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. scheduler.schedule()                                    │
│     ├─ For each request: kv_cache_manager.allocate_slots() │
│     │  ├─ Check Optane tier availability                   │
│     │  └─ Route block allocation through Optane if enabled │
│     │                                                       │
│     └─ Collect allocation stats                            │
│                                                              │
│  2. model_executor.execute_model(scheduler_output)         │
│     └─ Requests processed with KV cache from all tiers    │
│                                                              │
│  3. scheduler.update_from_output(model_output)             │
│     ├─ For completed requests: kv_cache_manager.free()    │
│     │  └─ Route block freeing through Optane               │
│     │                                                       │
│     ├─ Update Optane stats:                                │
│     │  ├─ Tier utilization snapshot                        │
│     │  ├─ Transition events (promotion/demotion)          │
│     │  └─ Performance metrics                              │
│     │                                                       │
│     └─ Health check & recommendations                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Points Summary

### KVCacheManager Changes ✅
```python
class KVCacheManager:
    def __init__(self, ..., optane_config: OptaneConfig | None = None):
        self.optane_manager = (
            OptaneManager(...) if optane_config else None
        )
        self.optane_stats = (
            OptaneStatsCollector() if optane_config else None
        )
    
    def allocate_slots(self, request, num_new_tokens, ...):
        # Existing logic, but now can query Optane for pre-allocated blocks
        if self.optane_manager:
            optane_blocks = self.optane_manager.promote_blocks(...)
        # ... continue with allocation ...
    
    def free(self, request):
        # Existing logic, but now defers to Optane
        blocks_to_free = self.coordinator.pop_blocks_for_free(request)
        if self.optane_manager:
            self.optane_manager.demote_blocks(blocks_to_free)
        else:
            self.block_pool.free_blocks(blocks_to_free)
    
    def get_tier_stats(self) -> dict:
        if self.optane_stats:
            return self.optane_stats.get_aggregate_stats()
        return {}
```

### Scheduler Changes (TODO)
```python
class Scheduler(SchedulerInterface):
    def __init__(
        self,
        ...,
        optane_config: OptaneConfig | None = None,
    ):
        # Pass to KVCacheManager
        self.kv_cache_manager = KVCacheManager(
            ...,
            optane_config=optane_config,
        )
```

### EngineCore Changes (TODO)
```python
class EngineCore:
    def __init__(self, ..., optane_config: OptaneConfig | None = None):
        self.scheduler = Scheduler(
            ...,
            optane_config=optane_config,
        )
    
    def step(self):
        # ... existing step logic ...
        
        # Collect stats if Optane enabled
        if self.scheduler.kv_cache_manager.optane_stats:
            stats = self.scheduler.kv_cache_manager.optane_stats.end_step()
            # Log or publish stats
```

---

## Next Steps

### Immediate (Today)
1. ✅ Create Optane stats module (`optane_stats.py`) - DONE
2. 🔄 Modify `Scheduler.__init__()` to accept and pass `optane_config`
3. 🔄 Modify `EngineCore.__init__()` to accept and pass `optane_config`
4. 🔄 Add stats collection in scheduler's `update_from_output()`

### Short-term (Phase 3.4)
1. Add Optane configuration to `VllmConfig`
2. Add CLI/API flags for Optane enablement
3. Implement block transition logging/debugging
4. Add unit tests for Optane integration
5. Add end-to-end tests with multi-tier caching

### Medium-term (Phase 4)
1. Performance benchmarking and tuning
2. Adaptive tier management based on workload patterns
3. Multi-GPU tier coordination
4. Optane tier persistence across requests
5. Integration with KV connector for async transfers

### Long-term (Phase 5+)
1. Predictive block promotion based on request patterns
2. Cost-aware tier selection (power, latency, throughput)
3. Multi-memory tier support (GPU → Optane → CPU → Disk)
4. Dynamic block size optimization per tier
5. Tier-aware request batching and scheduling

---

## Configuration Example (Planned)

```python
# Enable Optane tiering
optane_config = OptaneConfig(
    num_optane_blocks=100000,
    cache_policy="lru",  # or "arc"
    promotion_threshold_blocks=1000,
    demotion_threshold_blocks=500,
    enable_events=True,
    enable_stats=True,
)

scheduler = Scheduler(
    vllm_config=config,
    optane_config=optane_config,
    ...
)
```

---

## Key Design Decisions

1. **Backward Compatibility**: All Optane parameters are optional (default `None`)
   - Existing code paths unchanged
   - No performance overhead if Optane disabled

2. **Pluggable Policies**: Eviction policy (LRU/ARC) is configurable
   - Easy to add new policies in future
   - Policy performance can be A/B tested

3. **Event-Driven**: Optane manager emits events for block transitions
   - Decoupled from KV cache manager
   - Can be used for observability/debugging

4. **Stats Collection**: Optional comprehensive metrics
   - Per-step and aggregate statistics
   - Health monitoring with recommendations
   - Minimal overhead when enabled

---

## Testing Strategy

### Unit Tests
- OptaneManager: block allocation/freeing
- EvictionPolicies: LRU/ARC correctness
- StatsCollector: metrics aggregation
- HealthReport: warning/error detection

### Integration Tests
- KVCacheManager with OptaneManager
- Scheduler with Optane-enabled KVCacheManager
- EngineCore end-to-end with Optane

### Performance Tests
- Benchmark: GPU-only vs GPU+Optane
- Throughput, latency, memory utilization
- Tier transition overhead
- Stats collection overhead

---

## Branch Info

**Current Branch**: `phase-3.3-scheduler-integration`

**Base**: `phase-3-optane-core`

**Changes So Far**:
- ✅ Added `vllm/v1/metrics/optane_stats.py`
- 🔄 Ready to modify Scheduler and EngineCore

**To Merge**: 
1. Complete Scheduler integration
2. Complete EngineCore integration
3. Add configuration parsing
4. Pass all tests
5. Create PR for review

---

## References

- Optane Manager: `vllm/v1/core/optane_manager.py`
- Eviction Policies: `vllm/v1/core/eviction_policy_optane.py`
- KV Cache Manager: `vllm/v1/core/kv_cache_manager.py`
- Metrics: `vllm/v1/metrics/optane_stats.py` (NEW)
- Scheduler: `vllm/v1/core/sched/scheduler.py` (TODO)
- Engine Core: `vllm/v1/engine/core.py` (TODO)
