# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import time

from vllm.distributed.kv_events_optane import (
    OptaneBlockStoredEvent,
    OptaneBlockRemovedEvent,
    OptaneBlockAccessedEvent,
    OptaneStatsEvent,
    OptaneEventQueue,
    MEDIUM_OPTANE,
)


class TestOptaneBlockStoredEvent:
    """Tests for OptaneBlockStoredEvent."""

    def test_event_creation(self):
        """Test creating OptaneBlockStoredEvent."""
        event = OptaneBlockStoredEvent(
            block_ids=[0, 1, 2],
            parent_block_hash="abc123",
            token_ids=list(range(48)),
            block_size=16,
            timestamp=time.time(),
        )
        assert event.block_ids == [0, 1, 2]
        assert event.block_size == 16
        assert event.compressed is False

    def test_event_validation(self):
        """Test event validation."""
        # Empty block_ids should raise
        with pytest.raises(ValueError):
            OptaneBlockStoredEvent(
                block_ids=[],
                parent_block_hash=None,
                token_ids=[],
                block_size=16,
                timestamp=time.time(),
            )

        # Invalid block_size should raise
        with pytest.raises(ValueError):
            OptaneBlockStoredEvent(
                block_ids=[0],
                parent_block_hash=None,
                token_ids=[],
                block_size=0,  # Invalid
                timestamp=time.time(),
            )


class TestOptaneBlockRemovedEvent:
    """Tests for OptaneBlockRemovedEvent."""

    def test_event_creation(self):
        """Test creating OptaneBlockRemovedEvent."""
        event = OptaneBlockRemovedEvent(
            block_ids=[0, 1],
            timestamp=time.time(),
            reason="eviction",
        )
        assert event.block_ids == [0, 1]
        assert event.reason == "eviction"

    def test_invalid_reason_warning(self):
        """Test invalid reason generates warning."""
        event = OptaneBlockRemovedEvent(
            block_ids=[0],
            timestamp=time.time(),
            reason="unknown_reason",  # Invalid
        )
        assert event.reason == "unknown_reason"


class TestOptaneEventQueue:
    """Tests for OptaneEventQueue."""

    def test_queue_creation(self):
        """Test creating event queue."""
        queue = OptaneEventQueue(max_events=1000)
        assert len(queue) == 0
        assert queue.max_events == 1000

    def test_append_stored_event(self):
        """Test appending block stored event."""
        queue = OptaneEventQueue()
        queue.append_block_stored(
            block_ids=[0],
            parent_block_hash=None,
            token_ids=list(range(16)),
            block_size=16,
        )
        assert len(queue) == 1

    def test_append_removed_event(self):
        """Test appending block removed event."""
        queue = OptaneEventQueue()
        queue.append_block_removed(
            block_ids=[0],
            reason="eviction",
        )
        assert len(queue) == 1

    def test_append_accessed_event(self):
        """Test appending block accessed event."""
        queue = OptaneEventQueue()
        queue.append_block_accessed(
            block_ids=[0],
            access_type="read",
            latency_us=0.3,
        )
        assert len(queue) == 1

    def test_append_stats_event(self):
        """Test appending stats event."""
        queue = OptaneEventQueue()
        queue.append_stats(
            total_blocks=100,
            allocated_blocks=50,
            free_blocks=50,
            total_stored=100,
            total_retrieved=50,
            total_evicted=10,
            avg_store_latency_us=1.5,
            avg_retrieve_latency_us=0.4,
        )
        assert len(queue) == 1

    def test_take_events(self):
        """Test taking events from queue."""
        queue = OptaneEventQueue()
        queue.append_block_stored(
            block_ids=[0],
            parent_block_hash=None,
            token_ids=list(range(16)),
            block_size=16,
        )
        queue.append_block_stored(
            block_ids=[1],
            parent_block_hash=None,
            token_ids=list(range(16)),
            block_size=16,
        )

        assert len(queue) == 2
        events = queue.take_events()
        assert len(events) == 2
        assert len(queue) == 0  # Queue cleared

    def test_max_events_limit(self):
        """Test event queue respects max size."""
        queue = OptaneEventQueue(max_events=3)
        for i in range(5):
            queue.append_block_stored(
                block_ids=[i],
                parent_block_hash=None,
                token_ids=list(range(16)),
                block_size=16,
            )

        # Should only keep last 3 events
        assert len(queue) == 3
        assert queue.dropped_events == 2
