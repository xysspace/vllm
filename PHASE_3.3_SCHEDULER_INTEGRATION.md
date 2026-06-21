# Phase 3.3 Plan: Scheduler Integration with Optane Memory Tier Management

**Branch**: `phase-3.3-scheduler-integration` (from `phase-3-optane-core`)  
**Previous Phase**: Phase 3.2 (KVCacheManager Integration) ✅  
**Status**: 📋 PLANNING  
**Estimated Effort**: 3-4 weeks  

---

## 🎯 Objectives

Enable the scheduler to intelligently manage KV cache tier transitions (GPU ↔ Optane) based on:
1. **Real-time GPU memory pressure** - Trigger promotion of cold blocks to Optane when GPU ≥ 70% utilized
2. **Optane fill rate** - Trigger demotion of hot blocks back to GPU when Optane is filling up
3. **Request lifecycle** - Coordinate tier management with request scheduling phases (prefill → decode)
4. **Performance metrication** - Track tier transitions, latencies, and throughput impact

---

## 📊 Scope & Deliverables

### Phase 3.3 Deliverables

| Task | Module | Files | Lines | Status |
|------|--------|-------|-------|--------|
| **Tier management hooks in scheduler** | Scheduler | `vllm/v1/core/sched/scheduler.py` | ~50-100 | 🔄 TODO |
| **Engine core integration** | Engine | `vllm/v1/engine/core.py` | ~30-50 | 🔄 TODO |
| **Per-step Optane management** | Optane | `vllm/v1/core/optane_lifecycle.py` | ~150-200 | 🔄 TODO (NEW) |
| **Metrication & observability** | Stats | `vllm/v1/metrics/optane_stats.py` | ~100-150 | 🔄 TODO (NEW) |
| **End-to-end tests** | Tests | `tests/unit/optane/test_e2e_scheduler.py` | ~300-400 | 🔄 TODO (NEW) |
| **Performance benchmarks** | Benchmarks | `benchmarks/bench_optane_tier_mgmt.py` | ~200-300 | 🔄 TODO (NEW) |
| **Documentation** | Docs | `docs/design/optane_scheduler_integration.md` | ~200-300 | 🔄 TODO (NEW) |

**Total New Code**: ~1000-1500 LOC  
**Modified Existing Files**: 2 (scheduler, engine)  

---

## 🏗️ Architecture

### Current Flow (Phase 3.2)
```
EngineCore.step()
├─ scheduler.schedule()          [KV cache allocation decision]
├─ model_executor.execute()      [GPU compute]
└─ scheduler.update_from_output() [request finalization]
```

### Phase 3.3 Enhanced Flow
```
EngineCore.step()
├─ [NEW] optane_lifecycle.begin_step()
│   └─ GPU memory pressure check
│
├─ scheduler.schedule()
│   └─ [HOOK] kv_cache_manager.should_manage_optane_tier()
│
├─ model_executor.execute()
│   └─ [HOOK] GPU compute (blocks accessed, ref_cnt updated)
│
├─ scheduler.update_from_output()
│   └─ [HOOK] Block demotions/promotions based on hotness
│
└─ [NEW] optane_lifecycle.end_step()
    └─ Emit tier management metrics
```

### Key Integration Points

#### 1. **Scheduler.schedule()** → Promotion Decision
```python
def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
    # Existing code...
    self.kv_cache_manager.new_step_starts()
    
    # [NEW] Phase 3.3: Check if GPU pressure requires Optane migration
    if self.kv_cache_manager.should_manage_optane_tier():
        promoted_blocks = self.kv_cache_manager.promote_cold_blocks_to_optane()
        if promoted_blocks:
            logger.debug(f"Promoted {len(promoted_blocks)} blocks to Optane")
    
    # ... rest of scheduling logic ...
    return scheduler_output
```

#### 2. **update_from_output()** → Demotion Decision
```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
    # Existing code...
    
    # [NEW] Phase 3.3: Check if hot blocks should return to GPU
    optane_stats = self.kv_cache_manager.get_optane_stats()
    if optane_stats and optane_stats.optane_fill_pct > 90:
        demoted_blocks = self.kv_cache_manager.demote_hot_blocks_from_optane()
        if demoted_blocks:
            logger.debug(f"Demoted {len(demoted_blocks)} blocks from Optane")
    
    # ... rest of update logic ...
    return outputs
```

#### 3. **EngineCore** → Orchestration
```python
class EngineCore:
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        # [NEW] Begin tier management cycle
        self._optane_lifecycle.begin_step()
        
        if not self.scheduler.has_requests():
            return {}, False
            
        scheduler_output = self.scheduler.schedule(...)
        model_output = self.model_executor.execute_model(scheduler_output)
        engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
        
        # [NEW] End tier management cycle & emit metrics
        self._optane_lifecycle.end_step(engine_outputs)
        
        return engine_outputs, model_output is not None
```

---

## 📋 Detailed Implementation Plan

### Task 1: Optane Lifecycle Manager (NEW FILE)
**File**: `vllm/v1/core/optane_lifecycle.py`  
**Lines**: ~150-200  
**Purpose**: Coordinate Optane tier transitions across scheduler steps

```python
class OptaneLifecycleManager:
    """Manages Optane memory tier lifecycle during scheduling steps."""
    
    def __init__(self, kv_cache_manager: KVCacheManager):
        self.kv_cache_manager = kv_cache_manager
        self.step_metrics: OptaneStepMetrics = OptaneStepMetrics()
    
    def begin_step(self) -> None:
        """Called at start of scheduler.schedule()"""
        # Reset per-step counters
        self.step_metrics.reset()
        # Sample GPU memory pressure
        self.gpu_memory_pressure = self._sample_gpu_pressure()
    
    def check_promotion_trigger(self) -> bool:
        """Check if GPU blocks should be promoted to Optane"""
        # Watermark-based: GPU >= 70% utilization
        # Optane not at capacity
        # Not in prefill phase (avoid hotspot migrations)
    
    def check_demotion_trigger(self) -> bool:
        """Check if Optane blocks should be demoted back to GPU"""
        # Optane fill rate > 90%
        # GPU has available capacity
        # Blocks have recent access patterns
    
    def end_step(self, engine_outputs: dict) -> None:
        """Called at end of update_from_output()"""
        # Emit metrics to observability pipeline
        # Log tier transition summary
    
    @property
    def metrics(self) -> OptaneStepMetrics:
        """Per-step Optane management metrics"""
```

---

### Task 2: Scheduler Modifications
**File**: `vllm/v1/core/sched/scheduler.py`  
**Changes**: ~50-100 lines  
**Location**: `Scheduler.schedule()` method

```python
class Scheduler:
    def __init__(self, ...):
        # ... existing init code ...
        # [NEW] Phase 3.3
        self._optane_lifecycle: OptaneLifecycleManager | None = None
        if kv_cache_config.optane_config:
            self._optane_lifecycle = OptaneLifecycleManager(self.kv_cache_manager)
    
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
        # ... existing code ...
        self.kv_cache_manager.new_step_starts()
        
        # [NEW] Phase 3.3: Tier management before scheduling
        if self._optane_lifecycle:
            self._optane_lifecycle.begin_step()
            if self._optane_lifecycle.check_promotion_trigger():
                promoted = self.kv_cache_manager.promote_cold_blocks_to_optane()
                logger.debug(f"Promoted {len(promoted)} blocks to Optane")
        
        # ... rest of scheduling logic (unchanged) ...
        return scheduler_output
```

---

### Task 3: Engine Core Integration
**File**: `vllm/v1/engine/core.py`  
**Changes**: ~30-50 lines  
**Location**: `EngineCore.step()` method

```python
class EngineCore:
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        if not self.scheduler.has_requests():
            return {}, False
        
        # Existing logic...
        scheduler_output = self.scheduler.schedule(...)
        model_output = future.result()
        engine_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        
        # [NEW] Phase 3.3: Post-step Optane management
        if self.scheduler._optane_lifecycle:
            optane_stats = self.kv_cache_manager.get_optane_stats()
            if optane_stats and optane_stats.optane_fill_pct > 90:
                demoted = self.kv_cache_manager.demote_hot_blocks_from_optane()
                logger.debug(f"Demoted {len(demoted)} blocks from Optane")
            self.scheduler._optane_lifecycle.end_step(engine_outputs)
        
        return engine_outputs, model_output is not None
```

---

### Task 4: Optane Statistics & Observability (NEW FILE)
**File**: `vllm/v1/metrics/optane_stats.py`  
**Lines**: ~100-150  
**Purpose**: Track Optane performance metrics

```python
@dataclass
class OptaneStepMetrics:
    """Per-step Optane tier management metrics"""
    step_id: int
    timestamp: float
    
    # Tier transitions
    blocks_promoted_to_optane: int = 0
    blocks_demoted_from_optane: int = 0
    
    # Latencies
    promotion_latency_ms: float = 0.0
    demotion_latency_ms: float = 0.0
    
    # State
    gpu_blocks_used: int = 0
    optane_blocks_used: int = 0
    cpu_blocks_used: int = 0
    
    # Utilization
    gpu_memory_utilization_pct: float = 0.0
    optane_fill_pct: float = 0.0
    
    def reset(self) -> None:
        """Reset step counters"""
    
    @property
    def total_tier_transitions(self) -> int:
        """Sum of all tier movements"""
        return self.blocks_promoted_to_optane + self.blocks_demoted_from_optane
```

---

### Task 5: End-to-End Tests (NEW FILE)
**File**: `tests/unit/optane/test_e2e_scheduler.py`  
**Lines**: ~300-400  
**Purpose**: Integration tests for scheduler + Optane

```python
class TestSchedulerOptaneIntegration:
    """E2E tests for scheduler-driven tier management"""
    
    def test_promotion_on_gpu_pressure(self):
        """Verify blocks promoted when GPU >= 70%"""
    
    def test_demotion_on_optane_full(self):
        """Verify blocks demoted when Optane fill > 90%"""
    
    def test_promotion_respects_hotness(self):
        """Verify cold blocks promoted first"""
    
    def test_demotion_respects_access_patterns(self):
        """Verify hot blocks demoted first"""
    
    def test_no_promotion_during_prefill(self):
        """Verify migrations avoided during request prefill"""
    
    def test_metrics_accuracy(self):
        """Verify per-step metrics recorded correctly"""
    
    def test_concurrent_requests_tier_mgmt(self):
        """Verify correct tier management with multiple in-flight requests"""
    
    def test_backward_compatibility(self):
        """Verify scheduler works unchanged when Optane disabled"""
```

---

### Task 6: Performance Benchmarks (NEW FILE)
**File**: `benchmarks/bench_optane_tier_mgmt.py`  
**Lines**: ~200-300  
**Purpose**: Measure tier management overhead and throughput impact

```python
def benchmark_tier_management_overhead():
    """Measure latency of promote/demote operations"""
    # Result: < 1ms per transition with 100K blocks
    # Result: <5% throughput overhead with active tier management

def benchmark_throughput_with_optane():
    """Compare throughput with/without Optane enabled"""
    # Baseline: No Optane
    # Optane Disabled: Same as baseline (backward compat)
    # Optane Enabled (promotion only): +5-10% throughput
    # Optane Enabled (full mgmt): +15-25% throughput with 2x batch size
    
def benchmark_latency_with_optane():
    """Measure request latencies with tier management"""
    # P50, P95, P99 latencies
    # Compare GPU-only vs. GPU+Optane scheduling
```

---

### Task 7: Documentation (NEW FILE)
**File**: `docs/design/optane_scheduler_integration.md`  
**Lines**: ~200-300  
**Content**:
- Architecture overview (with diagrams)
- Tier promotion algorithm (watermark logic)
- Tier demotion algorithm (fill-based + hotness)
- Configuration guide
- Performance characteristics
- Troubleshooting guide

---

## 🔄 Integration Workflow

### Week 1-2: Core Implementation
```
Day 1-2:   Design review & spike on scheduler modifications
Day 3-4:   Implement OptaneLifecycleManager
Day 5-7:   Integrate hooks into Scheduler.schedule()
Day 8-10:  Integrate hooks into EngineCore.step()
Day 11-14: Unit tests for lifecycle manager
```

### Week 3: Testing & Refinement
```
Day 15-16: E2E tests (3-4 test cases)
Day 17-18: Performance benchmarking
Day 19-21: Bug fixes & performance tuning
```

### Week 4: Documentation & Optimization
```
Day 22-23: Comprehensive documentation
Day 24-27: Performance optimization pass
Day 28:    Final review & cleanup
```

---

## 🧪 Testing Strategy

### Unit Tests
- [ ] `OptaneLifecycleManager` initialization & lifecycle
- [ ] Promotion trigger detection (GPU pressure thresholds)
- [ ] Demotion trigger detection (Optane fill thresholds)
- [ ] Metrics collection & aggregation
- [ ] Backward compatibility (Optane disabled)

### Integration Tests
- [ ] Scheduler → OptaneLifecycleManager interaction
- [ ] EngineCore → Scheduler → OptaneLifecycleManager flow
- [ ] KVCacheManager → BlockPool → Optane tier movements
- [ ] Concurrent request handling with tier management
- [ ] KV-event emission for tier transitions

### Performance Tests
- [ ] Tier management latency (< 1ms)
- [ ] Throughput overhead (< 5%)
- [ ] Memory efficiency (GPU/Optane/CPU utilization)
- [ ] Scaling with batch size

### E2E Tests
- [ ] Full request lifecycle with tier transitions
- [ ] High concurrency (100+ requests)
- [ ] Various request profiles (prefill, decode, mixed)
- [ ] Stress tests (memory pressure spikes)

---

## 📊 Success Criteria

| Metric | Target | Validation |
|--------|--------|-----------|
| **Tier management latency** | < 1ms per transition | `bench_optane_tier_mgmt.py` |
| **Throughput overhead** | < 5% | Benchmark on 100K blocks |
| **Memory efficiency** | +15-25% effective GPU capacity | Compare batch sizes |
| **Backward compatibility** | 100% functional | All existing tests pass |
| **Code coverage** | > 85% | `pytest --cov` |
| **Documentation** | Complete with examples | Design doc + API docs |

---

## 🎯 Next Steps (Immediate)

1. **Get Phase 3.2 merged** (PR #1)
   ```bash
   gh pr ready 1
   gh pr merge 1 --squash
   ```

2. **Create Phase 3.3 branch**
   ```bash
   git checkout phase-3-optane-core
   git pull origin phase-3-optane-core
   git checkout -b phase-3.3-scheduler-integration
   ```

3. **Start implementation** with OptaneLifecycleManager (Task 1)

4. **Create tracking issues**
   - [ ] Task 1: OptaneLifecycleManager
   - [ ] Task 2: Scheduler integration
   - [ ] Task 3: EngineCore integration
   - [ ] Task 4: Metrics & stats
   - [ ] Task 5: End-to-end tests
   - [ ] Task 6: Benchmarks
   - [ ] Task 7: Documentation

---

## 📚 Dependencies & Prerequisites

- ✅ Phase 3.2 complete (KVCacheManager + Optane integration)
- ✅ OptaneManager fully functional with all bug fixes
- ✅ TierPromotionStrategy & TierDemotionStrategy ready
- ✅ KV-event integration working

---

## 🔗 Related Issues & PRs

- **Phase 3.2 PR**: #1 (KVCacheManager ↔ Optane integration)
- **Phase 1 Summary**: `PHASE_1_SUMMARY.md`
- **Phase 3.2 Requirements**: `PHASE_3.2_REQUIREMENTS.md`

---

## 📝 Notes

- All changes maintain **backward compatibility** (Optane is opt-in)
- Graceful degradation if Optane unavailable
- No changes to existing scheduler core logic (pure additions)
- All Optane-specific imports behind `TYPE_CHECKING` to avoid circular deps

---

**Created**: 2026-06-21  
**Last Updated**: 2026-06-21  
**Status**: 📋 PLANNING → 🔄 READY FOR IMPLEMENTATION
