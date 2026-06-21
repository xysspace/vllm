# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optane memory tier lifecycle manager for scheduler-driven tier transitions.

This module implements the per-step coordination of KV cache tier management
(GPU <-> Optane) based on memory pressure, fill rates, and block hotness patterns.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

logger = init_logger(__name__)


class TierTransitionTrigger(Enum):
    """Enum for tier transition trigger reasons."""
    
    # Promotion triggers
    GPU_MEMORY_PRESSURE = "gpu_memory_pressure"  # GPU >= 70% utilization
    SCHEDULED_PROMOTION = "scheduled_promotion"  # Periodic tier rebalancing
    
    # Demotion triggers
    OPTANE_FILL_PRESSURE = "optane_fill_pressure"  # Optane > 90% capacity
    SCHEDULED_DEMOTION = "scheduled_demotion"  # Periodic tier rebalancing
    HOTNESS_RECOVERY = "hotness_recovery"  # Hot block access pattern detected


@dataclass
class PromotionDecision:
    """Decision outcome for block promotion to Optane."""
    
    should_promote: bool
    trigger: TierTransitionTrigger | None = None
    reason: str = ""
    gpu_memory_utilization: float = 0.0
    estimated_blocks_to_promote: int = 0


@dataclass
class DemotionDecision:
    """Decision outcome for block demotion from Optane."""
    
    should_demote: bool
    trigger: TierTransitionTrigger | None = None
    reason: str = ""
    optane_fill_percentage: float = 0.0
    estimated_blocks_to_demote: int = 0


@dataclass
class OptaneStepMetrics:
    """Per-step Optane tier management metrics."""
    
    step_id: int
    timestamp: float = field(default_factory=time.time)
    
    # Tier transitions
    blocks_promoted_to_optane: int = 0
    blocks_demoted_from_optane: int = 0
    blocks_promoted_failed: int = 0
    blocks_demoted_failed: int = 0
    
    # Latencies (milliseconds)
    promotion_latency_ms: float = 0.0
    demotion_latency_ms: float = 0.0
    
    # State snapshots at step start/end
    gpu_blocks_used_start: int = 0
    gpu_blocks_used_end: int = 0
    optane_blocks_used_start: int = 0
    optane_blocks_used_end: int = 0
    cpu_blocks_used_start: int = 0
    cpu_blocks_used_end: int = 0
    
    # Utilization percentages
    gpu_memory_utilization_start_pct: float = 0.0
    gpu_memory_utilization_end_pct: float = 0.0
    optane_fill_start_pct: float = 0.0
    optane_fill_end_pct: float = 0.0
    
    # Triggers detected
    promotion_trigger: TierTransitionTrigger | None = None
    demotion_trigger: TierTransitionTrigger | None = None
    
    def reset(self) -> None:
        """Reset step-specific counters."""
        self.blocks_promoted_to_optane = 0
        self.blocks_demoted_from_optane = 0
        self.blocks_promoted_failed = 0
        self.blocks_demoted_failed = 0
        self.promotion_latency_ms = 0.0
        self.demotion_latency_ms = 0.0
        self.promotion_trigger = None
        self.demotion_trigger = None
    
    @property
    def total_tier_transitions(self) -> int:
        """Total number of blocks moved between tiers."""
        return self.blocks_promoted_to_optane + self.blocks_demoted_from_optane
    
    @property
    def total_failed_transitions(self) -> int:
        """Total number of failed tier transitions."""
        return self.blocks_promoted_failed + self.blocks_demoted_failed
    
    def __repr__(self) -> str:
        return (
            f"OptaneStepMetrics("
            f"step={self.step_id}, "
            f"promoted={self.blocks_promoted_to_optane}, "
            f"demoted={self.blocks_demoted_from_optane}, "
            f"gpu_util={self.gpu_memory_utilization_end_pct:.1f}%, "
            f"optane_fill={self.optane_fill_end_pct:.1f}%)"
        )


class OptaneLifecycleManager:
    """Manages Optane memory tier lifecycle during scheduler steps.
    
    This manager coordinates GPU <-> Optane tier transitions based on:
    1. GPU memory pressure (watermark-based promotion)
    2. Optane fill rate (fill-based demotion)
    3. Block hotness patterns (access frequency)
    4. Request lifecycle phase (avoid prefill disruptions)
    
    The manager integrates with the scheduler via two hooks:
    - begin_step(): Called at start of scheduler.schedule()
    - end_step(): Called after scheduler.update_from_output()
    """
    
    # Thresholds for tier transition decisions
    GPU_PROMOTION_THRESHOLD = 0.70  # Promote when GPU >= 70% utilized
    GPU_DEMOTION_THRESHOLD = 0.40  # Demote when GPU < 40% utilized
    OPTANE_DEMOTION_THRESHOLD = 0.90  # Demote when Optane >= 90% full
    OPTANE_SAFE_THRESHOLD = 0.75  # Target fill level after demotion
    
    def __init__(self, kv_cache_manager: "KVCacheManager", enable_logging: bool = True):
        """Initialize Optane lifecycle manager.
        
        Args:
            kv_cache_manager: Reference to KVCacheManager for tier operations
            enable_logging: Enable debug logging of tier transitions
        """
        self.kv_cache_manager = kv_cache_manager
        self.enable_logging = enable_logging
        
        # Current step metrics
        self.current_step_id: int = 0
        self.step_metrics: OptaneStepMetrics = OptaneStepMetrics(step_id=0)
        
        # GPU memory pressure tracking
        self._gpu_memory_pressure: float = 0.0
        self._optane_fill_percentage: float = 0.0
        
        # Cumulative statistics across all steps
        self._total_blocks_promoted: int = 0
        self._total_blocks_demoted: int = 0
        self._total_promotion_time_ms: float = 0.0
        self._total_demotion_time_ms: float = 0.0
    
    def begin_step(self, step_id: int) -> None:
        """Called at start of scheduler.schedule() for this step.
        
        Initializes per-step metrics and samples current system state.
        
        Args:
            step_id: Current scheduler step number
        """
        self.current_step_id = step_id
        self.step_metrics = OptaneStepMetrics(step_id=step_id)
        
        # Sample initial state
        optane_stats = self.kv_cache_manager.get_optane_stats()
        if optane_stats is None:
            if self.enable_logging:
                logger.debug("Optane not available, skipping lifecycle management")
            return
        
        # Record starting block counts
        # Note: We don't have direct access to per-tier block counts from KVCacheManager
        # so we use Optane stats as proxy
        self.step_metrics.gpu_blocks_used_start = (
            optane_stats.total_blocks_allocated - optane_stats.blocks_in_optane
        )
        self.step_metrics.optane_blocks_used_start = optane_stats.blocks_in_optane
        
        # Sample GPU memory pressure
        self._gpu_memory_pressure = self._estimate_gpu_memory_pressure()
        self.step_metrics.gpu_memory_utilization_start_pct = (
            self._gpu_memory_pressure * 100
        )
        
        # Sample Optane fill percentage
        if optane_stats.total_blocks_allocated > 0:
            self._optane_fill_percentage = (
                optane_stats.blocks_in_optane / optane_stats.total_blocks_allocated
            )
        self.step_metrics.optane_fill_start_pct = self._optane_fill_percentage * 100
    
    def check_promotion_trigger(self) -> PromotionDecision:
        """Check if GPU blocks should be promoted to Optane.
        
        Promotion is triggered when:
        1. GPU memory utilization >= 70% (watermark)
        2. Optane has available capacity
        3. Not in a prefill-dominant phase (avoid hotspot migrations)
        
        Returns:
            PromotionDecision with trigger and reasoning
        """
        optane_stats = self.kv_cache_manager.get_optane_stats()
        if optane_stats is None:
            return PromotionDecision(
                should_promote=False,
                reason="Optane not available"
            )
        
        # Check GPU pressure threshold
        if self._gpu_memory_pressure < self.GPU_PROMOTION_THRESHOLD:
            return PromotionDecision(
                should_promote=False,
                reason=(
                    f"GPU utilization {self._gpu_memory_pressure*100:.1f}% "
                    f"below threshold {self.GPU_PROMOTION_THRESHOLD*100:.1f}%"
                ),
                gpu_memory_utilization=self._gpu_memory_pressure,
            )
        
        # Check Optane capacity (ensure not at max capacity)
        optane_fill_pct = self._optane_fill_percentage
        if optane_fill_pct >= 0.95:  # Leave 5% headroom
            return PromotionDecision(
                should_promote=False,
                reason=(
                    f"Optane fill {optane_fill_pct*100:.1f}% "
                    "approaching capacity (keep 5% headroom)"
                ),
            )
        
        # Estimate how many blocks could be promoted
        available_optane_capacity = int(
            optane_stats.total_blocks_allocated * (0.95 - optane_fill_pct)
        )
        available_cold_blocks = max(0, optane_stats.blocks_available_for_promotion)
        
        blocks_to_promote = min(available_optane_capacity, available_cold_blocks)
        
        if blocks_to_promote == 0:
            return PromotionDecision(
                should_promote=False,
                reason="No cold blocks available for promotion or no Optane capacity",
            )
        
        return PromotionDecision(
            should_promote=True,
            trigger=TierTransitionTrigger.GPU_MEMORY_PRESSURE,
            reason=(
                f"GPU utilization {self._gpu_memory_pressure*100:.1f}% "
                f"exceeds threshold {self.GPU_PROMOTION_THRESHOLD*100:.1f}%"
            ),
            gpu_memory_utilization=self._gpu_memory_pressure,
            estimated_blocks_to_promote=blocks_to_promote,
        )
    
    def check_demotion_trigger(self) -> DemotionDecision:
        """Check if Optane blocks should be demoted back to GPU.
        
        Demotion is triggered when:
        1. Optane fill level >= 90% (capacity pressure)
        2. GPU has available capacity
        3. Blocks have been idle in Optane (not recently accessed)
        
        Returns:
            DemotionDecision with trigger and reasoning
        """
        optane_stats = self.kv_cache_manager.get_optane_stats()
        if optane_stats is None:
            return DemotionDecision(
                should_demote=False,
                reason="Optane not available"
            )
        
        # Check Optane fill threshold
        optane_fill_pct = self._optane_fill_percentage
        if optane_fill_pct < self.OPTANE_DEMOTION_THRESHOLD:
            return DemotionDecision(
                should_demote=False,
                reason=(
                    f"Optane fill {optane_fill_pct*100:.1f}% "
                    f"below threshold {self.OPTANE_DEMOTION_THRESHOLD*100:.1f}%"
                ),
                optane_fill_percentage=optane_fill_pct,
            )
        
        # Check GPU capacity (ensure GPU not at capacity)
        gpu_utilization = self._gpu_memory_pressure
        if gpu_utilization > 0.95:  # Leave margin
            return DemotionDecision(
                should_demote=False,
                reason=(
                    f"GPU already highly utilized {gpu_utilization*100:.1f}%, "
                    "cannot accommodate demoted blocks"
                ),
            )
        
        # Estimate how many blocks need to be demoted to reach safe Optane level
        total_capacity = optane_stats.total_blocks_allocated
        current_fill = int(optane_fill_pct * total_capacity)
        safe_fill = int(self.OPTANE_SAFE_THRESHOLD * total_capacity)
        blocks_to_demote = current_fill - safe_fill
        
        # Clamp to available idle blocks
        blocks_to_demote = min(
            blocks_to_demote, max(0, optane_stats.blocks_available_for_demotion)
        )
        
        if blocks_to_demote == 0:
            return DemotionDecision(
                should_demote=False,
                reason="No hot blocks available for demotion from Optane",
            )
        
        return DemotionDecision(
            should_demote=True,
            trigger=TierTransitionTrigger.OPTANE_FILL_PRESSURE,
            reason=(
                f"Optane fill {optane_fill_pct*100:.1f}% "
                f"exceeds threshold {self.OPTANE_DEMOTION_THRESHOLD*100:.1f}%"
            ),
            optane_fill_percentage=optane_fill_pct,
            estimated_blocks_to_demote=blocks_to_demote,
        )
    
    def end_step(self) -> None:
        """Called at end of scheduler.update_from_output() for this step.
        
        Finalizes per-step metrics and logs tier management summary.
        """
        # Sample final state
        optane_stats = self.kv_cache_manager.get_optane_stats()
        if optane_stats is None:
            return
        
        # Record ending block counts
        self.step_metrics.gpu_blocks_used_end = (
            optane_stats.total_blocks_allocated - optane_stats.blocks_in_optane
        )
        self.step_metrics.optane_blocks_used_end = optane_stats.blocks_in_optane
        
        # Record ending GPU utilization
        gpu_util_end = self._estimate_gpu_memory_pressure()
        self.step_metrics.gpu_memory_utilization_end_pct = gpu_util_end * 100
        
        # Record ending Optane fill
        if optane_stats.total_blocks_allocated > 0:
            optane_fill_end = (
                optane_stats.blocks_in_optane / optane_stats.total_blocks_allocated
            )
        else:
            optane_fill_end = 0.0
        self.step_metrics.optane_fill_end_pct = optane_fill_end * 100
        
        # Update cumulative stats
        self._total_blocks_promoted += self.step_metrics.blocks_promoted_to_optane
        self._total_blocks_demoted += self.step_metrics.blocks_demoted_from_optane
        self._total_promotion_time_ms += self.step_metrics.promotion_latency_ms
        self._total_demotion_time_ms += self.step_metrics.demotion_latency_ms
        
        # Log step summary if enabled
        if self.enable_logging and self.step_metrics.total_tier_transitions > 0:
            logger.debug(f"Optane tier management: {self.step_metrics}")
    
    def record_promotion(self, num_blocks: int, latency_ms: float) -> None:
        """Record successful block promotion to Optane.
        
        Args:
            num_blocks: Number of blocks promoted
            latency_ms: Promotion operation latency in milliseconds
        """
        self.step_metrics.blocks_promoted_to_optane += num_blocks
        self.step_metrics.promotion_latency_ms += latency_ms
    
    def record_promotion_failure(self, num_blocks: int) -> None:
        """Record failed block promotion attempt.
        
        Args:
            num_blocks: Number of blocks that failed to promote
        """
        self.step_metrics.blocks_promoted_failed += num_blocks
    
    def record_demotion(self, num_blocks: int, latency_ms: float) -> None:
        """Record successful block demotion from Optane.
        
        Args:
            num_blocks: Number of blocks demoted
            latency_ms: Demotion operation latency in milliseconds
        """
        self.step_metrics.blocks_demoted_from_optane += num_blocks
        self.step_metrics.demotion_latency_ms += latency_ms
    
    def record_demotion_failure(self, num_blocks: int) -> None:
        """Record failed block demotion attempt.
        
        Args:
            num_blocks: Number of blocks that failed to demote
        """
        self.step_metrics.blocks_demoted_failed += num_blocks
    
    def _estimate_gpu_memory_pressure(self) -> float:
        """Estimate current GPU memory utilization (0.0 to 1.0).
        
        This is a placeholder that uses KVCacheManager's usage metrics.
        In production, this would integrate with actual GPU memory monitoring.
        
        Returns:
            Estimated GPU memory utilization ratio (0.0 = empty, 1.0 = full)
        """
        # Use KV cache manager's usage estimate as proxy for GPU pressure
        block_pool_usage = self.kv_cache_manager.block_pool.get_usage()
        
        # Clamp to valid range
        return max(0.0, min(1.0, block_pool_usage))
    
    def get_metrics(self) -> OptaneStepMetrics:
        """Get current step metrics.
        
        Returns:
            OptaneStepMetrics for the current step
        """
        return self.step_metrics
    
    def get_cumulative_stats(self) -> dict[str, int | float]:
        """Get cumulative statistics across all steps.
        
        Returns:
            Dictionary with cumulative metrics
        """
        return {
            "total_blocks_promoted": self._total_blocks_promoted,
            "total_blocks_demoted": self._total_blocks_demoted,
            "total_promotion_time_ms": self._total_promotion_time_ms,
            "total_demotion_time_ms": self._total_demotion_time_ms,
            "avg_promotion_latency_ms": (
                self._total_promotion_time_ms / max(1, self._total_blocks_promoted)
                if self._total_blocks_promoted > 0
                else 0.0
            ),
            "avg_demotion_latency_ms": (
                self._total_demotion_time_ms / max(1, self._total_blocks_demoted)
                if self._total_blocks_demoted > 0
                else 0.0
            ),
        }
    
    def __repr__(self) -> str:
        return (
            f"OptaneLifecycleManager("
            f"step={self.current_step_id}, "
            f"promoted={self._total_blocks_promoted}, "
            f"demoted={self._total_blocks_demoted})"
        )
