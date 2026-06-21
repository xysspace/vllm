# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tier management utilities for Optane persistent memory integration.

Provides strategy classes and enums that govern when and how KV cache
blocks are promoted from GPU to Optane or demoted from Optane back to GPU.
These utilities are consumed by OptaneManager and KVCacheManager.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


class PromotionTrigger(Enum):
    """Events that can trigger block promotion from GPU to Optane."""

    MEMORY_PRESSURE = auto()
    """GPU memory utilization exceeds the high-watermark threshold."""

    IDLE_GPU = auto()
    """GPU is underutilized; proactively move cold blocks to Optane."""

    COLD_BLOCKS = auto()
    """Explicit scan identified blocks that have not been accessed recently."""

    WATERMARK = auto()
    """Free GPU block count fell below a configured watermark level."""


class DemotionTrigger(Enum):
    """Events that can trigger block demotion from Optane back to GPU."""

    HOT_ACCESS = auto()
    """A block in Optane was accessed and should be pulled into GPU."""

    PREFETCH_NEEDED = auto()
    """Upcoming tokens require blocks currently stored in Optane."""

    TIER_REBALANCE = auto()
    """Periodic rebalancing moves frequently-used Optane blocks to GPU."""


@dataclass
class TierPromotionDecision:
    """Result of a promotion strategy evaluation."""

    should_promote: bool
    """Whether promotion should proceed."""

    trigger: Optional[PromotionTrigger] = None
    """The trigger that caused this decision."""

    num_candidates: int = 0
    """Suggested number of blocks to promote."""

    reason: str = ""
    """Human-readable reason for the decision."""


@dataclass
class TierDemotionDecision:
    """Result of a demotion strategy evaluation."""

    should_demote: bool
    """Whether demotion should proceed."""

    trigger: Optional[DemotionTrigger] = None
    """The trigger that caused this decision."""

    num_candidates: int = 0
    """Suggested number of blocks to demote."""

    reason: str = ""
    """Human-readable reason for the decision."""


class TierPromotionStrategy:
    """Decides when to promote GPU blocks to Optane.

    Evaluates current memory utilization and access patterns to determine
    whether promotion is warranted and how many blocks to move.

    Args:
        gpu_high_watermark: GPU utilization fraction that triggers promotion.
        gpu_low_watermark: GPU utilization below which promotion is skipped.
        optane_max_fill: Maximum Optane fill fraction before promotion stops.
        default_batch_size: Number of blocks to promote per trigger event.
    """

    def __init__(
        self,
        gpu_high_watermark: float = 0.85,
        gpu_low_watermark: float = 0.70,
        optane_max_fill: float = 0.90,
        default_batch_size: int = 8,
    ) -> None:
        self.gpu_high_watermark = gpu_high_watermark
        self.gpu_low_watermark = gpu_low_watermark
        self.optane_max_fill = optane_max_fill
        self.default_batch_size = default_batch_size

    def evaluate(
        self,
        gpu_usage: float,
        optane_usage: float,
        free_gpu_blocks: int,
        total_gpu_blocks: int,
    ) -> TierPromotionDecision:
        """Evaluate whether promotion should occur.

        Args:
            gpu_usage: Current GPU memory utilization (0.0 – 1.0).
            optane_usage: Current Optane fill fraction (0.0 – 1.0).
            free_gpu_blocks: Number of free GPU blocks.
            total_gpu_blocks: Total GPU block capacity.

        Returns:
            TierPromotionDecision with recommendation.
        """
        # Never promote if Optane is nearly full.
        if optane_usage >= self.optane_max_fill:
            return TierPromotionDecision(
                should_promote=False,
                reason=f"Optane fill {optane_usage:.1%} >= max {self.optane_max_fill:.1%}",
            )

        if gpu_usage >= self.gpu_high_watermark:
            return TierPromotionDecision(
                should_promote=True,
                trigger=PromotionTrigger.MEMORY_PRESSURE,
                num_candidates=self.default_batch_size,
                reason=f"GPU usage {gpu_usage:.1%} >= high watermark {self.gpu_high_watermark:.1%}",
            )

        if gpu_usage >= self.gpu_low_watermark:
            return TierPromotionDecision(
                should_promote=True,
                trigger=PromotionTrigger.WATERMARK,
                num_candidates=max(1, self.default_batch_size // 2),
                reason=f"GPU usage {gpu_usage:.1%} >= low watermark {self.gpu_low_watermark:.1%}",
            )

        return TierPromotionDecision(
            should_promote=False,
            reason=f"GPU usage {gpu_usage:.1%} below watermarks",
        )


class TierDemotionStrategy:
    """Decides when to demote Optane blocks back to GPU.

    Evaluates access patterns and prefetch hints to determine whether it
    is beneficial to move blocks from Optane into GPU memory.

    Args:
        optane_high_watermark: Optane fill fraction that triggers demotion.
        gpu_target_usage: GPU utilization target; avoid demoting above this.
        default_batch_size: Number of blocks to demote per trigger event.
    """

    def __init__(
        self,
        optane_high_watermark: float = 0.80,
        gpu_target_usage: float = 0.75,
        default_batch_size: int = 4,
    ) -> None:
        self.optane_high_watermark = optane_high_watermark
        self.gpu_target_usage = gpu_target_usage
        self.default_batch_size = default_batch_size

    def evaluate(
        self,
        gpu_usage: float,
        optane_usage: float,
        blocks_in_optane: int,
    ) -> TierDemotionDecision:
        """Evaluate whether demotion should occur.

        Args:
            gpu_usage: Current GPU memory utilization (0.0 – 1.0).
            optane_usage: Current Optane fill fraction (0.0 – 1.0).
            blocks_in_optane: Total blocks currently stored in Optane.

        Returns:
            TierDemotionDecision with recommendation.
        """
        # No blocks to demote.
        if blocks_in_optane == 0:
            return TierDemotionDecision(
                should_demote=False,
                reason="No blocks in Optane",
            )

        # Avoid demoting into an already pressured GPU tier.
        if gpu_usage >= self.gpu_target_usage:
            return TierDemotionDecision(
                should_demote=False,
                reason=f"GPU usage {gpu_usage:.1%} >= target {self.gpu_target_usage:.1%}",
            )

        if optane_usage >= self.optane_high_watermark:
            return TierDemotionDecision(
                should_demote=True,
                trigger=DemotionTrigger.TIER_REBALANCE,
                num_candidates=self.default_batch_size,
                reason=(
                    f"Optane usage {optane_usage:.1%} >= "
                    f"high watermark {self.optane_high_watermark:.1%}"
                ),
            )

        return TierDemotionDecision(
            should_demote=False,
            reason=f"Optane usage {optane_usage:.1%} below watermark",
        )
