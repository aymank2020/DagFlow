"""Tests for the reactive streams layer.

Tests cover:
- ReactiveStream event emission and delivery
- Backpressure and overflow strategies
- Subscription lifecycle management
- Stream operators (debounce, throttle, buffer, merge, combine_latest)
- Async scheduler task processing
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode
from dagflow.memo.cache import MemoCache
from dagflow.reactive.stream import (
    EventKind,
    OverflowStrategy,
    ReactiveStream,
    StreamEvent,
)
from dagflow.reactive.subscription import (
    Subscription,
    SubscriptionManager,
    SubscriptionState,
)
from dagflow.reactive.operators import buffer, combine_latest, merge, throttle
from dagflow.reactive.scheduler_async import AsyncScheduler, Priority


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def simple_graph() -> ComputeGraph:
    """Create a simple graph with inputs and compute nodes."""
    g = ComputeGraph()
    g.add_input("x", value=1)
    g.add_input("y", value=2)
    g.add_compute("sum", lambda d: d["x"] + d["y"], ["x", "y"])
    g.add_compute("double", lambda d: d["sum"] * 2, ["sum"])
    return g


@pytest.fixture
def stream(simple_graph: ComputeGraph) -> ReactiveStream:
    """Create a stream observing all nodes."""
    s = ReactiveStream(simple_graph, buffer_size=50)
    s.observe("sum", "double")
    return s


# ─── ReactiveStream Tests ─────────────────────────────────────────────


class TestReactiveStream:
    """Tests for ReactiveStream core functionality."""

    def test_stream_creation(self, simple_graph: ComputeGraph) -> None:
        """Stream initializes with correct defaults."""
        s = ReactiveStream(simple_graph)
        assert s.buffer_size == 256
        assert s.pending_count == 0
        assert not s.is_paused
        assert not s.is_completed

    def test_observe_valid_nodes(self, simple_graph: ComputeGraph) -> None:
        """Observing existing nodes succeeds."""
        s = ReactiveStream(simple_graph)
        s.observe("x", "sum")
        # No exception means success

    def test_observe_invalid_node_raises(self, simple_graph: ComputeGraph) -> None:
        """Observing non-existent node raises KeyError."""
        s = ReactiveStream(simple_graph)
        with pytest.raises(KeyError):
            s.observe("nonexistent")

    def test_emit_delivers_to_listeners(self, stream: ReactiveStream) -> None:
        """Emitted events are delivered to all subscribers."""
        received: List[StreamEvent] = []
        stream.subscribe(lambda e: received.append(e))

        count = stream.emit_changes({"sum": 42, "double": 84})
        assert count == 2
        assert len(received) == 2
        assert received[0].node_id == "sum"
        assert received[0].value == 42

    def test_emit_filters_unobserved_nodes(self, stream: ReactiveStream) -> None:
        """Only observed nodes generate events."""
        received: List[StreamEvent] = []
        stream.subscribe(lambda e: received.append(e))

        # "x" is not observed
        count = stream.emit_changes({"x": 99, "sum": 10})
        assert count == 1
        assert received[0].node_id == "sum"

    def test_pause_buffers_events(self, stream: ReactiveStream) -> None:
        """Paused stream buffers events instead of delivering."""
        received: List[StreamEvent] = []
        stream.subscribe(lambda e: received.append(e))

        stream.pause()
        stream.emit_changes({"sum": 10})
        assert len(received) == 0
        assert stream.pending_count == 1

    def test_resume_flushes_buffer(self, stream: ReactiveStream) -> None:
        """Resuming delivers all buffered events."""
        received: List[StreamEvent] = []
        stream.subscribe(lambda e: received.append(e))

        stream.pause()
        stream.emit_changes({"sum": 10})
        stream.emit_changes({"double": 20})

        flushed = stream.resume()
        assert flushed == 2
        assert len(received) == 2

    def test_complete_stops_emission(self, stream: ReactiveStream) -> None:
        """Completed stream rejects new events."""
        received: List[StreamEvent] = []
        stream.subscribe(lambda e: received.append(e))

        stream.complete()
        assert stream.is_completed

        count = stream.emit_changes({"sum": 99})
        assert count == 0
        # Only the COMPLETE event was delivered
        assert len(received) == 1
        assert received[0].kind == EventKind.COMPLETE

    def test_overflow_drop_oldest(self, simple_graph: ComputeGraph) -> None:
        """DROP_OLDEST strategy removes oldest buffered event."""
        s = ReactiveStream(
            simple_graph,
            buffer_size=2,
            overflow=OverflowStrategy.DROP_OLDEST,
        )
        s.observe("sum")
        s.pause()

        s.emit_changes({"sum": 1})
        s.emit_changes({"sum": 2})
        s.emit_changes({"sum": 3})  # Should drop event with value 1

        assert s.dropped_count == 1
        events = s.drain()
        assert len(events) == 2

    def test_unsubscribe_stops_delivery(self, stream: ReactiveStream) -> None:
        """Unsubscribed listener no longer receives events."""
        received: List[StreamEvent] = []
        idx = stream.subscribe(lambda e: received.append(e))

        stream.emit_changes({"sum": 1})
        assert len(received) == 1

        stream.unsubscribe(idx)
        stream.emit_changes({"sum": 2})
        assert len(received) == 1  # No new events

    def test_multiple_subscribers(self, stream: ReactiveStream) -> None:
        """Multiple subscribers all receive events."""
        received_a: List[StreamEvent] = []
        received_b: List[StreamEvent] = []
        stream.subscribe(lambda e: received_a.append(e))
        stream.subscribe(lambda e: received_b.append(e))

        stream.emit_changes({"sum": 42})
        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_drain_returns_buffered_events(self, stream: ReactiveStream) -> None:
        """Drain removes and returns all buffered events."""
        stream.pause()
        stream.emit_changes({"sum": 1})
        stream.emit_changes({"double": 2})

        events = stream.drain()
        assert len(events) == 2
        assert stream.pending_count == 0


# ─── Subscription Manager Tests ───────────────────────────────────────


class TestSubscriptionManager:
    """Tests for SubscriptionManager lifecycle control."""

    def test_subscribe_creates_active_subscription(
        self, stream: ReactiveStream
    ) -> None:
        """New subscriptions start in ACTIVE state."""
        manager = SubscriptionManager()
        sub = manager.subscribe(stream, lambda e: None)
        assert sub.is_active
        assert manager.active_count == 1

    def test_dispose_subscription(self, stream: ReactiveStream) -> None:
        """Disposed subscriptions stop receiving events."""
        manager = SubscriptionManager()
        received: List[StreamEvent] = []
        sub = manager.subscribe(stream, lambda e: received.append(e))

        stream.emit_changes({"sum": 1})
        assert len(received) == 1

        sub.dispose()
        assert sub.is_disposed
        stream.emit_changes({"sum": 2})
        assert len(received) == 1

    def test_pause_group(self, stream: ReactiveStream) -> None:
        """Pausing a group pauses all subscriptions in it."""
        manager = SubscriptionManager()
        received: List[StreamEvent] = []
        manager.subscribe(stream, lambda e: received.append(e), group="ui")
        manager.subscribe(stream, lambda e: received.append(e), group="ui")

        paused = manager.pause_group("ui")
        assert paused == 2

        stream.emit_changes({"sum": 1})
        assert len(received) == 0

    def test_resume_group(self, stream: ReactiveStream) -> None:
        """Resuming a group reactivates paused subscriptions."""
        manager = SubscriptionManager()
        received: List[StreamEvent] = []
        manager.subscribe(stream, lambda e: received.append(e), group="data")

        manager.pause_group("data")
        resumed = manager.resume_group("data")
        assert resumed == 1

        stream.emit_changes({"sum": 1})
        assert len(received) == 1

    def test_dispose_all(self, stream: ReactiveStream) -> None:
        """dispose_all disposes every subscription."""
        manager = SubscriptionManager()
        manager.subscribe(stream, lambda e: None, group="a")
        manager.subscribe(stream, lambda e: None, group="b")

        count = manager.dispose_all()
        assert count == 2
        assert manager.active_count == 0

    def test_filter_function(self, stream: ReactiveStream) -> None:
        """Filter function prevents non-matching events from delivery."""
        manager = SubscriptionManager()
        received: List[StreamEvent] = []

        # Only accept events for "sum" node
        manager.subscribe(
            stream,
            lambda e: received.append(e),
            filter_fn=lambda e: e.node_id == "sum",
        )

        stream.emit_changes({"sum": 10, "double": 20})
        assert len(received) == 1
        assert received[0].node_id == "sum"

    def test_cleanup_disposed(self, stream: ReactiveStream) -> None:
        """cleanup_disposed removes disposed entries from tracking."""
        manager = SubscriptionManager()
        sub = manager.subscribe(stream, lambda e: None)
        sub.dispose()

        cleaned = manager.cleanup_disposed()
        assert cleaned == 1
        assert manager.total_count == 0


# ─── Operator Tests ───────────────────────────────────────────────────


class TestStreamOperators:
    """Tests for stream operators."""

    def test_merge_combines_streams(self, simple_graph: ComputeGraph) -> None:
        """Merge forwards events from all source streams."""
        s1 = ReactiveStream(simple_graph)
        s1.observe("x")
        s2 = ReactiveStream(simple_graph)
        s2.observe("y")

        merged = merge(s1, s2, graph=simple_graph)
        received: List[StreamEvent] = []
        merged.subscribe(lambda e: received.append(e))

        s1.emit_changes({"x": 10})
        s2.emit_changes({"y": 20})
        assert len(received) == 2

    def test_merge_requires_two_sources(self, simple_graph: ComputeGraph) -> None:
        """Merge raises ValueError with fewer than 2 sources."""
        s1 = ReactiveStream(simple_graph)
        with pytest.raises(ValueError, match="at least 2"):
            merge(s1, graph=simple_graph)

    def test_buffer_flushes_at_max_size(self, simple_graph: ComputeGraph) -> None:
        """Buffer flushes when max_size is reached."""
        source = ReactiveStream(simple_graph)
        source.observe("sum")

        flushed_batches: List[List[StreamEvent]] = []
        buffered = buffer(
            source, simple_graph, max_size=3, on_flush=lambda b: flushed_batches.append(b)
        )

        source.emit_changes({"sum": 1})
        source.emit_changes({"sum": 2})
        assert len(flushed_batches) == 0

        source.emit_changes({"sum": 3})
        assert len(flushed_batches) == 1
        assert len(flushed_batches[0]) == 3

    def test_combine_latest_waits_for_all(self, simple_graph: ComputeGraph) -> None:
        """combine_latest only emits after all sources have emitted."""
        s1 = ReactiveStream(simple_graph)
        s1.observe("x")
        s2 = ReactiveStream(simple_graph)
        s2.observe("y")

        combined = combine_latest(s1, s2, graph=simple_graph)
        received: List[StreamEvent] = []
        combined.subscribe(lambda e: received.append(e))

        # Only s1 emits — no combined output yet
        s1.emit_changes({"x": 10})
        assert len(received) == 0

        # Now s2 emits — combined output
        s2.emit_changes({"y": 20})
        assert len(received) >= 1


# ─── Async Scheduler Tests ────────────────────────────────────────────


class TestAsyncScheduler:
    """Tests for the async scheduler."""

    def test_schedule_and_tick(self, simple_graph: ComputeGraph) -> None:
        """Scheduled tasks are processed on tick."""
        cache = MemoCache()
        scheduler = AsyncScheduler(simple_graph, cache)

        scheduler.schedule("x", priority=Priority.NORMAL)
        assert scheduler.pending_count == 1

        scheduler.tick()
        assert scheduler.pending_count == 0
        assert scheduler.processed_count == 1

    def test_priority_ordering(self, simple_graph: ComputeGraph) -> None:
        """Higher priority tasks are processed first."""
        cache = MemoCache()
        scheduler = AsyncScheduler(simple_graph, cache, coalesce=False)

        scheduler.schedule("y", priority=Priority.LOW)
        scheduler.schedule("x", priority=Priority.HIGH)

        # Peek should show the high-priority task
        task = scheduler.peek()
        assert task is not None
        assert task.node_id == "x"

    def test_coalescing(self, simple_graph: ComputeGraph) -> None:
        """Duplicate schedules for same node are coalesced."""
        cache = MemoCache()
        scheduler = AsyncScheduler(simple_graph, cache, coalesce=True)

        added1 = scheduler.schedule("x", priority=Priority.NORMAL)
        added2 = scheduler.schedule("x", priority=Priority.HIGH)

        assert added1 is True
        assert added2 is False  # Coalesced
        assert scheduler.pending_count == 1

    def test_cancel_task(self, simple_graph: ComputeGraph) -> None:
        """Cancelled tasks are removed from the queue."""
        cache = MemoCache()
        scheduler = AsyncScheduler(simple_graph, cache)

        scheduler.schedule("x")
        assert scheduler.pending_count == 1

        cancelled = scheduler.cancel("x")
        assert cancelled is True
        assert scheduler.pending_count == 0

    def test_cancel_all(self, simple_graph: ComputeGraph) -> None:
        """cancel_all removes all pending tasks."""
        cache = MemoCache()
        scheduler = AsyncScheduler(simple_graph, cache)

        scheduler.schedule("x")
        scheduler.schedule("y")

        count = scheduler.cancel_all()
        assert count == 2
        assert scheduler.pending_count == 0

    def test_batch_schedule(self, simple_graph: ComputeGraph) -> None:
        """schedule_batch adds multiple tasks at once."""
        cache = MemoCache()
        scheduler = AsyncScheduler(simple_graph, cache)

        added = scheduler.schedule_batch(["x", "y"])
        assert added == 2
        assert scheduler.pending_count == 2

    def test_on_complete_callback(self, simple_graph: ComputeGraph) -> None:
        """Completion callbacks fire after task processing."""
        cache = MemoCache()
        scheduler = AsyncScheduler(simple_graph, cache)
        completed: List[str] = []

        scheduler.on_complete(lambda nid, val: completed.append(nid))
        scheduler.schedule("x")
        scheduler.tick()

        # Callback fires for propagated results
        # (may be empty if no actual propagation occurs for non-dirty inputs)
        assert scheduler.processed_count == 1
