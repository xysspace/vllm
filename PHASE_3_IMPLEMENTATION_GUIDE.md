# Phase 3 Implementation Guide

**For Developers Starting Phase 3.2: KVCacheManager Integration**

---

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Component Overview](#component-overview)
3. [Integration Points](#integration-points)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Testing Strategy](#testing-strategy)
6. [Debugging Tips](#debugging-tips)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Environment Setup

```bash
# 1. Checkout the phase-3-optane-core branch
git checkout phase-3-optane-core

# 2. Install vLLM with development dependencies
pip install -e ".[dev]"

# 3. Run basic component tests (when available)
python -m pytest tests/v1/core/test_optane_*.py -v
```

### Verify Installation

```python
# Test that all components are importable
from vllm.v1.core.optane_types import EvictionTier, OptaneBlockInfo
from vllm.v1.core.optane_manager import OptaneManager
from vllm.v1.core.eviction_policies import create_eviction_policy
from vllm.v1.core.optane_block_lifecycle import BlockLifecycleManager
from vllm.v1.core.optane_compression import CompressionManager
from vllm.v1.core.optane_metrics import OptaneMetricsCollector

print("✅ All Phase 3 components imported successfully!")
```

---

## 🔍 Component Overview

### Dependency Graph

```
┌─────────────────────────────────────────────────┐
│      OptaneMetricsCollector (Observability)     │
└─────────────────────┬───────────────────────────┘
                      │ Records metrics to
                      ↓
┌─────────────────────────────────────────────────┐
│        OptaneManager (Space Management)         │
│                                                 │
│  Uses:                                          │
│  - eviction_policies (decisions)                │
│  - optane_block_lifecycle (state)              │
│  - optane_compression (compression)            │
└──────┬──────────────────────────────────────────┘
       │
       └── Used by: KVCacheManager (NEXT PHASE)
           Modified files:
           - vllm/v1/core/kv_cache_manager.py
           - vllm/v1/core/block_pool.py
```

### Component Responsibilities

| Component | Role | Key Methods |
|-----------|------|------------|
| **optane_types** | Data structures | - (type definitions) |
| **optane_manager** | Space allocation | `allocate_block()`, `free_block()`, `promote_block_to_optane()` |
| **eviction_policies** | Eviction decisions | `get_eviction_candidates()`, `record_access()` |
| **optane_block_lifecycle** | State tracking | `allocate()`, `promote()`, `demote()`, `free()` |
| **optane_compression** | Data compression | `compress()`, `decompress()`, `should_compress()` |
| **optane_metrics** | Observability | `record_*()`, `get_summary()` |

---

## 🔗 Integration Points

### Phase 3.2 Target: KVCacheManager Integration

The KVCacheManager is the core allocation system for KV cache blocks. Integration points:

#### 1. **Initialization** (kv_cache_manager.py: `__init__`)

**Current Code:**
```python
class KVCacheManager:
    def __init__(self, ...):
        self.block_pool = BlockPool(...)
        self.eviction_policy = ...  # GPU-only currently
```

**Required Modification:**
```python
class KVCacheManager:
    def __init__(self, ..., optane_config: OptaneConfig = None):
        self.block_pool = BlockPool(...)
        
        # NEW: Initialize Optane if enabled
        if optane_config and optane_config.enable_optane:
            self.optane_manager = OptaneManager(
                config=optane_config,
                block_size_bytes=...,
                max_blocks=...
            )
            self.eviction_policy = create_eviction_policy(
                optane_config.optane_eviction_policy
            )
            self.metrics = OptaneMetricsCollector(
                sample_rate=0.1 if optane_config.optane_enable_metrics else 0.0
            )
        else:
            self.optane_manager = None
```

#### 2. **Block Allocation** (block_pool.py: `allocate()`)

**Current Code:**
```python
def allocate(self) -> int:
    # Allocate from GPU VRAM
    block_id = self._get_free_block()
    return block_id
```

**Required Modification:**
```python
def allocate(self, request_id: str) -> int:
    # 1. Try GPU allocation first
    block_id = self._get_free_gpu_block()
    
    if block_id is not None:
        # Success - block allocated in GPU
        if self.optane_manager:
            self.metrics.record_block_allocated(
                block_id, 
                self.block_size, 
                EvictionTier.GPU
            )
        return block_id
    
    # 2. GPU full - try Optane promotion
    if self.optane_manager:
        evict_block = self._select_candidate_for_optane()
        if evict_block is not None:
            self._promote_to_optane(evict_block)
            # Reuse the GPU space
            block_id = evict_block
            return block_id
    
    # 3. Both GPU and Optane full - standard eviction
    self._evict_lru_block()
    block_id = self._get_free_gpu_block()
    return block_id
```

#### 3. **Block Retrieval** (kv_cache_manager.py: `get_block()`)

**Current Code:**
```python
def get_block(self, block_id: int) -> torch.Tensor:
    # Get from GPU cache
    return self.gpu_cache[block_id]
```

**Required Modification:**
```python
def get_block(self, block_id: int) -> torch.Tensor:
    # 1. Check if in GPU
    if self._is_block_in_gpu(block_id):
        tensor = self.gpu_cache[block_id]
        if self.optane_manager:
            self.metrics.record_access(block_id, latency_us=0.1, cache_hit=True)
        return tensor
    
    # 2. Check if in Optane (needs promotion)
    if self.optane_manager and block_id in self.optane_manager.blocks:
        # Promote from Optane back to GPU
        tensor = self._promote_from_optane(block_id)
        return tensor
    
    # 3. Not found
    raise KeyError(f"Block {block_id} not found in any tier")
```

#### 4. **Block Release** (kv_cache_manager.py: `release_block()`)

**Current Code:**
```python
def release_block(self, block_id: int):
    self.block_pool.free(block_id)
```

**Required Modification:**
```python
def release_block(self, block_id: int):
    # Check if in Optane
    if self.optane_manager and block_id in self.optane_manager.blocks:
        self.optane_manager.free_block(block_id)
        self.metrics.record_block_freed(
            block_id, 
            self.block_size, 
            EvictionTier.OPTANE
        )
    
    # Release from GPU
    self.block_pool.free(block_id)
    if self.optane_manager:
        self.metrics.record_block_freed(
            block_id, 
            self.block_size, 
            EvictionTier.GPU
        )
```

---

## 📋 Step-by-Step Implementation

### Phase 3.2a: Setup & Imports

**File: `vllm/v1/core/kv_cache_manager.py`**

Step 1: Add imports at top of file
```python
from vllm.config.optane import OptaneConfig
from vllm.v1.core.optane_manager import OptaneManager
from vllm.v1.core.optane_types import EvictionTier
from vllm.v1.core.eviction_policies import create_eviction_policy
from vllm.v1.core.optane_block_lifecycle import BlockLifecycleManager
from vllm.v1.core.optane_compression import CompressionManager
from vllm.v1.core.optane_metrics import OptaneMetricsCollector
```

Step 2: Modify KVCacheManager.__init__()
```python
def __init__(
    self,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    block_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int = 0,
    optane_config: Optional[OptaneConfig] = None,  # NEW
):
    # ... existing code ...
    
    # NEW: Initialize Optane tier
    self.optane_config = optane_config
    self.optane_manager: Optional[OptaneManager] = None
    self.lifecycle_manager: Optional[BlockLifecycleManager] = None
    self.compression_manager: Optional[CompressionManager] = None
    self.metrics_collector: Optional[OptaneMetricsCollector] = None
    
    if optane_config and optane_config.enable_optane:
        self._init_optane()
```

Step 3: Add initialization method
```python
def _init_optane(self) -> None:
    """Initialize Optane tier managers."""
    logger.info("Initializing Optane tier for KV cache")
    
    # Create OptaneManager
    self.optane_manager = OptaneManager(
        config=self.optane_config,
        block_size_bytes=self.block_size,
        max_blocks=self.optane_config.max_blocks_optane,
        enable_events=True,
    )
    
    # Create BlockLifecycleManager
    self.lifecycle_manager = BlockLifecycleManager()
    
    # Create CompressionManager
    self.compression_manager = CompressionManager(
        enable_compression=self.optane_config.optane_enable_compression,
        compression_level=self.optane_config.optane_compression_level,
        min_compress_size=1024,
    )
    
    # Create Metrics
    self.metrics_collector = OptaneMetricsCollector(
        sample_rate=0.1 if self.optane_config.optane_enable_metrics else 0.0
    )
    
    logger.info(
        f"Optane initialized: "
        f"{self.optane_config.optane_cache_size:.1f}GB, "
        f"policy={self.optane_config.optane_eviction_policy}, "
        f"compression={'enabled' if self.optane_config.optane_enable_compression else 'disabled'}"
    )
```

---

### Phase 3.2b: Allocation Logic

**File: `vllm/v1/core/block_pool.py`**

```python
def allocate(self, request_id: Optional[str] = None) -> int:
    """Allocate a block, trying tiers in order."""
    
    # Try GPU first
    if self.gpu_free_blocks:
        block_id = self.gpu_free_blocks.pop()
        self.gpu_used_blocks.add(block_id)
        
        # Track in Optane if enabled
        if self.optane_manager:
            lifecycle = self.lifecycle_manager.allocate(
                block_id=block_id,
                size_bytes=self.block_size,
                initial_tier=EvictionTier.GPU
            )
            self.metrics_collector.record_block_allocated(
                block_id, self.block_size, EvictionTier.GPU
            )
        
        return block_id
    
    # GPU full - try eviction to Optane
    if self.optane_manager:
        return self._allocate_with_optane_promotion(request_id)
    
    # No Optane - use standard eviction
    return self._allocate_with_cpu_swap()

def _allocate_with_optane_promotion(self, request_id: str) -> int:
    """Allocate block by promoting one to Optane."""
    
    # Select candidate for Optane promotion
    candidates = self.eviction_policy.get_eviction_candidates(
        self.optane_manager.blocks, 
        num_candidates=1
    )
    
    if not candidates:
        # No eviction candidates - allocate from Optane directly
        available = self.optane_manager.get_available_space()
        if available >= self.block_size:
            block_id = self._allocate_from_optane()
            return block_id
        else:
            # Both tiers full - evict from Optane to SSD
            self._evict_from_optane_to_ssd()
            return self._allocate_from_optane()
    
    # Promote candidate to Optane
    candidate_id = candidates[0]
    self._promote_to_optane(candidate_id)
    
    # Reuse GPU block
    self.gpu_used_blocks.add(candidate_id)
    return candidate_id

def _promote_to_optane(self, block_id: int) -> bool:
    """Promote a block from GPU to Optane."""
    
    start_time = time.perf_counter()
    
    # Get block data from GPU
    gpu_data = self.gpu_cache[block_id].cpu().numpy()
    
    # Compress if enabled
    if self.compression_manager.should_compress(len(gpu_data)):
        compressed = self.compression_manager.compress(gpu_data)
        data_to_store = compressed.data
        is_compressed = True
    else:
        data_to_store = gpu_data
        is_compressed = False
    
    # Allocate in Optane
    if not self.optane_manager.allocate_block(
        block_id=block_id,
        size_bytes=len(data_to_store),
        compressed=is_compressed,
    ):
        logger.warning(f"Failed to allocate block {block_id} in Optane")
        return False
    
    # Store data (backend-specific)
    self._store_in_optane(block_id, data_to_store)
    
    # Update lifecycle
    self.lifecycle_manager.demote(block_id, EvictionTier.OPTANE)
    
    # Record metrics
    latency_us = (time.perf_counter() - start_time) * 1e6
    self.metrics_collector.record_tier_transition(
        block_id,
        from_tier=EvictionTier.GPU,
        to_tier=EvictionTier.OPTANE,
        size_bytes=len(data_to_store),
        latency_us=latency_us
    )
    
    logger.debug(
        f"Promoted block {block_id} to Optane "
        f"({len(data_to_store)} bytes, {latency_us:.2f}us)"
    )
    return True
```

---

### Phase 3.2c: Retrieval & Demotion

```python
def get_block_from_cache(self, block_id: int) -> torch.Tensor:
    """Get block, promoting from Optane if needed."""
    
    # Check if in GPU (fast path)
    if block_id in self.gpu_used_blocks:
        tensor = self.gpu_cache[block_id]
        if self.metrics_collector:
            self.metrics_collector.record_access(
                block_id, latency_us=0.1, cache_hit=True
            )
        return tensor
    
    # Check if in Optane (slow path)
    if self.optane_manager and block_id in self.optane_manager.blocks:
        return self._promote_from_optane(block_id)
    
    # Not found
    raise KeyError(f"Block {block_id} not in cache")

def _promote_from_optane(self, block_id: int) -> torch.Tensor:
    """Promote block from Optane back to GPU."""
    
    start_time = time.perf_counter()
    
    # Get block metadata
    block_info = self.optane_manager.get_block_info(block_id)
    if block_info is None:
        raise KeyError(f"Block {block_id} not in Optane")
    
    # Retrieve from Optane backend
    data = self._retrieve_from_optane(block_id)
    
    # Decompress if needed
    if block_info.compressed:
        data = self.compression_manager.decompress(data)
    
    # Convert back to tensor
    tensor = torch.from_numpy(data).to(self.device)
    
    # Store in GPU cache
    self.gpu_cache[block_id] = tensor
    self.gpu_used_blocks.add(block_id)
    
    # Update lifecycle
    self.lifecycle_manager.promote(block_id, EvictionTier.GPU)
    
    # Update OptaneManager
    self.optane_manager.demote_block_from_optane(block_id)\n    
    # Record metrics
    latency_us = (time.perf_counter() - start_time) * 1e6
    self.metrics_collector.record_tier_transition(
        block_id,
        from_tier=EvictionTier.OPTANE,
        to_tier=EvictionTier.GPU,
        size_bytes=block_info.size_bytes,
        latency_us=latency_us
    )
    
    logger.debug(
        f"Promoted block {block_id} from Optane to GPU "
        f"({block_info.size_bytes} bytes, {latency_us:.2f}us)"
    )
    
    return tensor
```

---

## ✅ Testing Strategy

### Unit Tests (200 lines)

**File: `tests/v1/core/test_optane_manager.py`**

```python
import pytest
from vllm.config.optane import OptaneConfig
from vllm.v1.core.optane_manager import OptaneManager
from vllm.v1.core.optane_types import EvictionTier

class TestOptaneManager:
    @pytest.fixture
    def config(self):
        return OptaneConfig(
            enable_optane=True,
            optane_cache_size=1.0,  # 1 GB
        )
    
    @pytest.fixture
    def manager(self, config):
        return OptaneManager(config, block_size_bytes=16384, max_blocks=10000)
    
    def test_allocate_block(self, manager):
        block_info = manager.allocate_block(1, 16384)
        assert block_info is not None
        assert block_info.block_id == 1
        assert manager.get_utilization() > 0
    
    def test_free_block(self, manager):
        manager.allocate_block(1, 16384)
        success = manager.free_block(1)
        assert success
        assert manager.get_utilization() == 0
    
    def test_out_of_space(self, manager):
        # Allocate many blocks until full
        block_size = 16384
        max_blocks = int(manager.capacity_bytes / block_size)
        
        for i in range(max_blocks):
            block_info = manager.allocate_block(i, block_size)
            assert block_info is not None
        
        # Next allocation should fail
        block_info = manager.allocate_block(max_blocks, block_size)
        assert block_info is None
```

### Integration Tests (200 lines)

**File: `tests/v1/core/test_optane_integration.py`**

```python
def test_kv_cache_manager_with_optane():
    """Test KVCacheManager with Optane enabled."""
    from vllm.config.optane import OptaneConfig
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    
    optane_config = OptaneConfig(
        enable_optane=True,
        optane_cache_size=10.0,
        optane_eviction_policy="cascade",
        optane_enable_compression=True,
    )
    
    manager = KVCacheManager(
        num_gpu_blocks=100,
        num_cpu_blocks=100,
        block_size=16384,
        dtype=torch.float16,
        device="cuda:0",
        optane_config=optane_config,
    )
    
    # Allocate blocks
    blocks = [manager.allocate() for _ in range(50)]
    
    # Verify all allocated
    assert len(blocks) == 50
    
    # Get metrics
    summary = manager.metrics_collector.get_summary()
    assert summary.total_blocks_allocated >= 50
```

---

## 🐛 Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("vllm.v1.core")
logger.setLevel(logging.DEBUG)
```

### Inspect Component State

```python
# Check OptaneManager state
print(f"Optane utilization: {manager.optane_manager.get_utilization():.1%}")
print(f"Available space: {manager.optane_manager.get_available_space() / 1e9:.2f}GB")
print(f"Statistics: {manager.optane_manager.get_statistics()}")

# Check BlockLifecycleManager state
print(f"Tier usage: {lifecycle_manager.get_tier_usage()}")

# Check metrics
summary = metrics_collector.get_summary()
print(f"Cache hit rate: {summary.cache_hit_count / summary.total_accesses:.1%}")
```

### Add Assertions

```python
# Verify block is in expected tier
block_state = lifecycle_manager.get_state(block_id)
assert block_state.current_tier == EvictionTier.OPTANE

# Verify metrics are updated
prev_count = metrics.current_blocks_allocated
manager.allocate_block(1, 16384)
assert metrics.current_blocks_allocated == prev_count + 1
```

---

## ⚡ Performance Optimization

### 1. Reduce Metrics Overhead

```python
# Default: 10% sampling
metrics = OptaneMetricsCollector(sample_rate=0.1)

# For critical sections: full metrics
metrics = OptaneMetricsCollector(sample_rate=1.0)

# For production: disable metrics
metrics = OptaneMetricsCollector(sample_rate=0.0)
```

### 2. Compression Level Tuning

```python
# Fast compression (level 1): ~100 MB/s
comp = CompressionManager(compression_level=1)

# Balanced (level 3): ~50 MB/s, 2-3x ratio
comp = CompressionManager(compression_level=3)

# High compression (level 11): ~10 MB/s, 4-6x ratio
comp = CompressionManager(compression_level=11)
```

### 3. Policy Selection

| Workload | Policy | Reason |
|----------|--------|--------|
| Streaming | LRU | Sequential access pattern |
| Batch repetitive | LFU | Some sequences reused |
| Mixed/unknown | Cascade | Predictable, safe |
| Variable patterns | Parallel | Adaptive routing |

---

## 🔧 Troubleshooting

### Problem: "OptaneManager allocation failed"

**Cause:** Out of Optane space

**Solution:**
```python
# Check available space
available = manager.optane_manager.get_available_space()
required = block_size
if available < required:
    # Need to evict blocks
    candidates = eviction_policy.get_eviction_candidates(
        manager.optane_manager.blocks, num_candidates=1
    )
    for block_id in candidates:
        manager.optane_manager.evict_block_from_optane(block_id)
```

### Problem: "Compression is too slow"

**Cause:** Compression level too high

**Solution:**
```python
# Lower compression level
comp = CompressionManager(compression_level=1)  # was 11

# Or increase min_compress_size
comp = CompressionManager(min_compress_size=100000)  # Skip small blocks
```

### Problem: "High cache miss rate"

**Cause:** Wrong eviction policy or low Optane capacity

**Solutions:**
```python
# Try different policy
policy = create_eviction_policy("parallel")  # More adaptive

# Or increase Optane capacity
config.optane_cache_size = 512  # Increase from 256GB
```

### Problem: "Memory fragmentation"

**Cause:** Many small allocations/deallocations

**Solution:**
```python
# Check fragmentation
manager = OptaneManager(...)
free_offsets = len(manager.free_offsets)
utilization = manager.get_utilization()
fragmentation_ratio = free_offsets / (capacity / block_size)

if fragmentation_ratio > 0.1:
    logger.warning("Optane fragmentation detected")
    # Could trigger defragmentation in Phase 3.3
```

---

## 📖 Additional Resources

- **PHASE_3_PLAN.md** - Complete Phase 3 specification
- **PHASE_3_KICKOFF.md** - Component overview and status
- **Component Docstrings** - Detailed API documentation
- **Type Hints** - Full type hints in source code

---

## ✨ Key Takeaways

1. **ModularDesign** - Each component has single responsibility
2. **Type Safety** - Comprehensive type hints prevent errors
3. **Observability** - Metrics collection for debugging/optimization
4. **Extensibility** - Factory patterns for easy policy addition
5. **Thread Safety** - Metrics collector is thread-safe; others assume single-threaded access (for now)

---

**Last Updated:** June 21, 2026  
**Next Phase:** Phase 3.2 KVCacheManager Integration  
**Estimated Duration:** 1-2 weeks
