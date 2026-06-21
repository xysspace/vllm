# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, List, Optional
from enum import Enum

from vllm.logger import init_logger

logger = init_logger(__name__)

# Optane tier identifier for KV-Events
MEDIUM_OPTANE = "optane"


class OptaneMedium(Enum):
    """Optane storage medium variants."""
    NATIVE = "optane_native"  # Native mmap backend
    PMDK = "optane_pmdk"  # PMDK-based backend


@dataclass
class OptaneBlockStoredEvent:
    """Event emitted when KV blocks are stored in Optane.
    
    This event is part of the KV-Event system that enables llm-d
    to track block movements across tiers.
    """
    block_ids: List[int]
    parent_block_hash: Optional[str]
    token_ids: List[int]
    block_size: int
    timestamp: float
    compressed: bool = False
    compressed_size: Optional[int] = None
    backend: str = "native"  # 'native' or 'pmdk'
    group_idx: int = 0  # KV cache group index
    lora_id: Optional[int] = None
    lora_name: Optional[str] = None
    extra_keys: Optional[List[tuple[Any, ...]]] = None

    def __post_init__(self) -> None:
        """Validate event data."""
        if not self.block_ids:
            raise ValueError("block_ids cannot be empty")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.token_ids and len(self.token_ids) != (
            len(self.block_ids) * self.block_size
        ):
            logger.warning(
                f"Token count mismatch: expected "
                f"{len(self.block_ids) * self.block_size}, "
                f"got {len(self.token_ids)}"
            )


@dataclass
class OptaneBlockRemovedEvent:
    """Event emitted when KV blocks are removed from Optane.
    
    Triggered when blocks are evicted or manually deleted.
    """
    block_ids: List[int]
    timestamp: float
    reason: str = "eviction"  # 'eviction', 'manual', 'cache_reset'
    group_idx: int = 0

    def __post_init__(self) -> None:
        """Validate event data."""
        if not self.block_ids:
            raise ValueError("block_ids cannot be empty")
        if self.reason not in ("eviction", "manual", "cache_reset"):
            logger.warning(f"Unknown removal reason: {self.reason}")


@dataclass
class OptaneBlockAccessedEvent:
    """Event emitted when KV blocks are accessed from Optane.
    
    Enables tracking of access patterns for performance analysis.
    """
    block_ids: List[int]
    timestamp: float
    access_type: str = "read"  # 'read' or 'prefetch'
    latency_us: Optional[float] = None  # Measured latency in microseconds
    group_idx: int = 0

    def __post_init__(self) -> None:
        """Validate event data."""
        if not self.block_ids:
            raise ValueError("block_ids cannot be empty")
        if self.access_type not in ("read", "prefetch"):
            logger.warning(f"Unknown access type: {self.access_type}")


@dataclass
class OptaneStatsEvent:
    """Event emitted periodically with Optane tier statistics.
    
    Provides aggregated metrics for monitoring and debugging.
    """
    timestamp: float
    total_blocks: int
    allocated_blocks: int
    free_blocks: int
    total_stored: int
    total_retrieved: int
    total_evicted: int
    avg_store_latency_us: float
    avg_retrieve_latency_us: float
    compression_ratio: Optional[float] = None
    usage_fraction: float = 0.0


class OptaneEventQueue:
    """Queue for managing Optane-related KV events.
    
    Integrates with vLLM's event system for tracking block movements.
    """

    def __init__(self, max_events: int = 10000):
        """Initialize event queue.
        
        Args:
            max_events: Maximum events to keep in queue before dropping oldest
        """
        self.max_events = max_events
        self.events: List[Any] = []
        self.dropped_events = 0

    def append_block_stored(
        self,
        block_ids: List[int],
        parent_block_hash: Optional[str],
        token_ids: List[int],
        block_size: int,
        **kwargs,
    ) -> None:
        """Add block stored event to queue."""
        import time

        event = OptaneBlockStoredEvent(
            block_ids=block_ids,
            parent_block_hash=parent_block_hash,
            token_ids=token_ids,
            block_size=block_size,
            timestamp=time.time(),
            **kwargs,
        )
        self._add_event(event)

    def append_block_removed(
        self,
        block_ids: List[int],
        reason: str = "eviction",
        **kwargs,
    ) -> None:
        """Add block removed event to queue."""
        import time

        event = OptaneBlockRemovedEvent(
            block_ids=block_ids,
            timestamp=time.time(),
            reason=reason,
            **kwargs,
        )
        self._add_event(event)

    def append_block_accessed(
        self,
        block_ids: List[int],
        access_type: str = "read",
        **kwargs,
    ) -> None:
        """Add block accessed event to queue."""
        import time

        event = OptaneBlockAccessedEvent(
            block_ids=block_ids,
            timestamp=time.time(),
            access_type=access_type,
            **kwargs,
        )
        self._add_event(event)

    def append_stats(
        self,
        total_blocks: int,
        allocated_blocks: int,
        free_blocks: int,
        total_stored: int,
        total_retrieved: int,
        total_evicted: int,
        avg_store_latency_us: float,
        avg_retrieve_latency_us: float,
        **kwargs,
    ) -> None:
        """Add statistics event to queue."""
        import time

        event = OptaneStatsEvent(
            timestamp=time.time(),
            total_blocks=total_blocks,
            allocated_blocks=allocated_blocks,
            free_blocks=free_blocks,
            total_stored=total_stored,
            total_retrieved=total_retrieved,
            total_evicted=total_evicted,
            avg_store_latency_us=avg_store_latency_us,
            avg_retrieve_latency_us=avg_retrieve_latency_us,
            **kwargs,
        )
        self._add_event(event)

    def _add_event(self, event: Any) -> None:
        """Add event to queue, dropping oldest if necessary."""
        if len(self.events) >= self.max_events:
            self.events.pop(0)
            self.dropped_events += 1

        self.events.append(event)

    def take_events(self) -> List[Any]:
        """Atomically get all events and clear queue."""
        events = self.events
        self.events = []
        return events

    def __len__(self) -> int:
        """Get current queue size."""
        return len(self.events)

    def clear(self) -> None:
        """Clear all events from queue."""
        self.events.clear()
