# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optane memory tier statistics and observability metrics.

This module provides comprehensive metrics collection and reporting for
Optane-based KV cache tiering, including per-step metrics, cumulative
statistics, and performance characterization.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class OptaneTierUtilization:
    """Snapshot of block utilization across tiers."""
    
    gpu_blocks_used: int = 0
    gpu_blocks_total: int = 0
    optane_blocks_used: int = 0
    optane_blocks_total: int = 0
    cpu_blocks_used: int = 0
    cpu_blocks_total: int = 0
    
    @property
    def gpu_utilization_pct(self) -> float:
        """GPU tier utilization percentage."""
        if self.gpu_blocks_total == 0:
            return 0.0
        return (self.gpu_blocks_used / self.gpu_blocks_total) * 100.0
    
    @property
    def optane_utilization_pct(self) -> float:
        """Optane tier utilization percentage."""
        if self.optane_blocks_total == 0:
            return 0.0
        return (self.optane_blocks_used / self.optane_blocks_total) * 100.0
    
    @property
    def cpu_utilization_pct(self) -> float:
        """CPU tier utilization percentage."""
        if self.cpu_blocks_total == 0:
            return 0.0
        return (self.cpu_blocks_used / self.cpu_blocks_total) * 100.0
    
    @property
    def total_blocks_used(self) -> int:
        """Total blocks across all tiers."""
        return self.gpu_blocks_used + self.optane_blocks_used + self.cpu_blocks_used
    
    @property
    def total_blocks_available(self) -> int:
        """Total available blocks across all tiers."""
        return self.gpu_blocks_total + self.optane_blocks_total + self.cpu_blocks_total


@dataclass
class OptaneTierTransition:
    """Record of a single tier transition event."""
    
    timestamp: float
    step_id: int
    transition_type: str  # 'promotion' or 'demotion'
    num_blocks: int
    source_tier: str  # 'gpu', 'optane', 'cpu'
    dest_tier: str
    reason: str
    latency_ms: float
    success: bool = True


@dataclass
class OptaneTierStats:
    """Comprehensive Optane tier statistics for observability."""
    
    # Temporal information
    collection_timestamp: float = field(default_factory=time.time)
    step_id: int = 0
    
    # Tier utilization snapshot
    utilization: OptaneTierUtilization = field(default_factory=OptaneTierUtilization)
    
    # Transition history in current step
    transitions_this_step: list[OptaneTierTransition] = field(default_factory=list)
    
    # Cumulative transition statistics
    total_promotions: int = 0
    total_demotions: int = 0
    total_failed_transitions: int = 0
    
    # Latency statistics (milliseconds)
    avg_promotion_latency_ms: float = 0.0
    max_promotion_latency_ms: float = 0.0
    avg_demotion_latency_ms: float = 0.0
    max_demotion_latency_ms: float = 0.0
    
    # Performance metrics
    blocks_migrated_total: int = 0
    migration_throughput_blocks_per_sec: float = 0.0
    
    # Request-level statistics
    requests_benefiting_from_optane: int = 0
    avg_request_latency_reduction_pct: float = 0.0
    
    def add_transition(self, transition: OptaneTierTransition) -> None:
        """Record a tier transition event."""
        self.transitions_this_step.append(transition)
        
        if transition.transition_type == 'promotion':
            self.total_promotions += 1
            if transition.success:
                self.blocks_migrated_total += transition.num_blocks
                self.avg_promotion_latency_ms = (
                    (self.avg_promotion_latency_ms * (self.total_promotions - 1) +
                     transition.latency_ms) / self.total_promotions
                )
                self.max_promotion_latency_ms = max(
                    self.max_promotion_latency_ms, transition.latency_ms
                )
            else:
                self.total_failed_transitions += 1
        
        elif transition.transition_type == 'demotion':
            self.total_demotions += 1
            if transition.success:
                self.blocks_migrated_total += transition.num_blocks
                self.avg_demotion_latency_ms = (
                    (self.avg_demotion_latency_ms * (self.total_demotions - 1) +
                     transition.latency_ms) / self.total_demotions
                )
                self.max_demotion_latency_ms = max(
                    self.max_demotion_latency_ms, transition.latency_ms
                )
            else:
                self.total_failed_transitions += 1
    
    def clear_step_transitions(self) -> None:
        """Clear per-step transition history."""
        self.transitions_this_step.clear()
    
    @property
    def total_transitions(self) -> int:
        """Total number of tier transitions."""
        return self.total_promotions + self.total_demotions
    
    @property
    def transition_success_rate_pct(self) -> float:
        """Percentage of successful transitions."""
        if self.total_transitions == 0:
            return 100.0
        successful = self.total_transitions - self.total_failed_transitions
        return (successful / self.total_transitions) * 100.0
    
    def __repr__(self) -> str:
        return (
            f"OptaneTierStats("
            f"step={self.step_id}, "
            f"promotions={self.total_promotions}, "
            f"demotions={self.total_demotions}, "
            f"gpu_util={self.utilization.gpu_utilization_pct:.1f}%, "
            f"optane_util={self.utilization.optane_utilization_pct:.1f}%)"
        )


class OptaneStatsCollector:
    """Collects and aggregates Optane tier statistics."""
    
    def __init__(self):
        """Initialize the statistics collector."""
        self.current_stats: OptaneTierStats = OptaneTierStats()
        
        # Historical statistics
        self.step_history: list[OptaneTierStats] = []
        self.max_history_size: int = 1000  # Keep last 1000 steps
        
        # Aggregate metrics
        self._total_steps_observed: int = 0
        self._total_transitions_recorded: int = 0
    
    def begin_step(self, step_id: int) -> None:
        """Begin collecting metrics for a new step."""
        self.current_stats = OptaneTierStats(step_id=step_id)
    
    def record_transition(
        self,
        transition_type: str,
        num_blocks: int,
        source_tier: str,
        dest_tier: str,
        reason: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Record a tier transition event."""
        transition = OptaneTierTransition(
            timestamp=time.time(),
            step_id=self.current_stats.step_id,
            transition_type=transition_type,
            num_blocks=num_blocks,
            source_tier=source_tier,
            dest_tier=dest_tier,
            reason=reason,
            latency_ms=latency_ms,
            success=success,
        )
        self.current_stats.add_transition(transition)
        self._total_transitions_recorded += 1
    
    def update_utilization(
        self,
        gpu_used: int,
        gpu_total: int,
        optane_used: int,
        optane_total: int,
        cpu_used: int = 0,
        cpu_total: int = 0,
    ) -> None:
        """Update tier utilization snapshot."""
        self.current_stats.utilization = OptaneTierUtilization(
            gpu_blocks_used=gpu_used,
            gpu_blocks_total=gpu_total,
            optane_blocks_used=optane_used,
            optane_blocks_total=optane_total,
            cpu_blocks_used=cpu_used,
            cpu_blocks_total=cpu_total,
        )
    
    def end_step(self) -> OptaneTierStats:
        """Finalize metrics for the current step and return them."""
        self._total_steps_observed += 1
        
        # Add to history
        self.step_history.append(self.current_stats)
        if len(self.step_history) > self.max_history_size:
            self.step_history.pop(0)
        
        return self.current_stats
    
    def get_current_stats(self) -> OptaneTierStats:
        """Get current step statistics."""
        return self.current_stats
    
    def get_step_stats(self, step_id: int) -> Optional[OptaneTierStats]:
        """Get statistics for a specific step."""
        for stats in self.step_history:
            if stats.step_id == step_id:
                return stats
        return None
    
    def get_aggregate_stats(self) -> dict:
        """Get aggregate statistics across all observed steps."""
        if not self.step_history:
            return {
                "total_steps": 0,
                "total_transitions": 0,
                "avg_transitions_per_step": 0.0,
                "avg_gpu_utilization_pct": 0.0,
                "avg_optane_utilization_pct": 0.0,
                "max_gpu_utilization_pct": 0.0,
                "max_optane_utilization_pct": 0.0,
            }
        
        total_promotions = sum(s.total_promotions for s in self.step_history)
        total_demotions = sum(s.total_demotions for s in self.step_history)
        total_transitions = total_promotions + total_demotions
        
        gpu_utils = [s.utilization.gpu_utilization_pct for s in self.step_history]
        optane_utils = [s.utilization.optane_utilization_pct for s in self.step_history]
        
        return {
            "total_steps": self._total_steps_observed,
            "total_transitions": total_transitions,
            "total_promotions": total_promotions,
            "total_demotions": total_demotions,
            "avg_transitions_per_step": (
                total_transitions / len(self.step_history)
                if self.step_history
                else 0.0
            ),
            "avg_gpu_utilization_pct": (
                sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
            ),
            "avg_optane_utilization_pct": (
                sum(optane_utils) / len(optane_utils) if optane_utils else 0.0
            ),
            "max_gpu_utilization_pct": max(gpu_utils) if gpu_utils else 0.0,
            "max_optane_utilization_pct": max(optane_utils) if optane_utils else 0.0,
            "avg_promotion_latency_ms": (
                sum(s.avg_promotion_latency_ms for s in self.step_history) /
                len(self.step_history)
                if self.step_history
                else 0.0
            ),
            "avg_demotion_latency_ms": (
                sum(s.avg_demotion_latency_ms for s in self.step_history) /
                len(self.step_history)
                if self.step_history
                else 0.0
            ),
        }
    
    def get_recent_transitions(self, num_steps: int = 10) -> list[OptaneTierTransition]:
        """Get transitions from the most recent N steps."""
        transitions = []
        for stats in self.step_history[-num_steps:]:
            transitions.extend(stats.transitions_this_step)
        return transitions
    
    def clear_history(self) -> None:
        """Clear all historical statistics."""
        self.step_history.clear()
        self._total_steps_observed = 0
        self._total_transitions_recorded = 0


@dataclass
class OptanePerformanceMetrics:
    """Performance impact metrics for Optane tiering."""
    
    # Throughput metrics
    tokens_per_second_with_optane: float = 0.0
    tokens_per_second_without_optane: float = 0.0
    throughput_improvement_pct: float = 0.0
    
    # Latency metrics (milliseconds)
    avg_request_latency_with_optane_ms: float = 0.0
    avg_request_latency_without_optane_ms: float = 0.0
    latency_improvement_pct: float = 0.0
    
    # Memory efficiency
    effective_gpu_memory_multiplier: float = 1.0  # How much larger effective GPU is
    
    # Cost metrics (if applicable)
    power_consumption_ratio_with_optane: float = 1.0
    
    def calculate_improvements(self) -> None:
        """Calculate improvement percentages."""
        if self.tokens_per_second_without_optane > 0:
            self.throughput_improvement_pct = (
                (self.tokens_per_second_with_optane /
                 self.tokens_per_second_without_optane - 1) * 100.0
            )
        
        if self.avg_request_latency_without_optane_ms > 0:
            self.latency_improvement_pct = (
                (1 - self.avg_request_latency_with_optane_ms /
                 self.avg_request_latency_without_optane_ms) * 100.0
            )


@dataclass
class OptaneHealthReport:
    """Overall health report for Optane tier management system."""
    
    is_healthy: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    # System state
    optane_available: bool = False
    optane_enabled: bool = False
    
    # Performance observations
    promotion_failures: int = 0
    demotion_failures: int = 0
    slow_transitions: int = 0  # Transitions > 10ms
    
    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        self.is_healthy = False
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.is_healthy = False
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation."""
        self.recommendations.append(recommendation)
    
    def __repr__(self) -> str:
        status = "HEALTHY" if self.is_healthy else "UNHEALTHY"
        return (
            f"OptaneHealthReport("
            f"status={status}, "
            f"warnings={len(self.warnings)}, "
            f"errors={len(self.errors)}, "
            f"recommendations={len(self.recommendations)})"
        )


def create_health_report(stats: OptaneTierStats) -> OptaneHealthReport:
    """Create a health report based on current statistics.
    
    Args:
        stats: Current Optane tier statistics
    
    Returns:
        OptaneHealthReport with health status and recommendations
    """
    report = OptaneHealthReport()
    
    # Check for high failure rates
    if stats.total_transitions > 0:
        failure_rate = (stats.total_failed_transitions / stats.total_transitions) * 100
        if failure_rate > 10:
            report.add_error(
                f"Tier transition failure rate {failure_rate:.1f}% exceeds threshold"
            )
        elif failure_rate > 5:
            report.add_warning(
                f"Tier transition failure rate {failure_rate:.1f}% elevated"
            )
    
    # Check for slow transitions
    for transition in stats.transitions_this_step:
        if transition.latency_ms > 10:
            report.slow_transitions += 1
    
    if report.slow_transitions > 5:
        report.add_warning(
            f"Slow tier transitions detected ({report.slow_transitions} > 10ms)"
        )
    
    # Check utilization imbalance
    gpu_util = stats.utilization.gpu_utilization_pct
    optane_util = stats.utilization.optane_utilization_pct
    
    if gpu_util > 95:
        report.add_warning(f"GPU tier heavily utilized ({gpu_util:.1f}%)")
        report.add_recommendation("Consider allocating more GPU blocks or enabling Optane")
    
    if optane_util > 95:
        report.add_warning(f"Optane tier heavily utilized ({optane_util:.1f}%)")
        report.add_recommendation("Consider increasing Optane capacity or reducing workload")
    
    return report
