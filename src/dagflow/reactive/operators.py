"""Stream operators — composable transformations on reactive streams.

Provides functional operators that transform, filter, and combine streams.
Each operator creates a new derived stream without modifying the source.

Operators:
- debounce: Suppress rapid-fire events, emit only after quiet period
- throttle: Limit event rate to at most one per interval
- buffer: Collect events into batches by count or time window
- merge: Combine multiple streams into one
- combine_latest: Emit combined value whenever any source emits

This module depends on:
- reactive.stream (ReactiveStream, StreamEvent, EventKind)
- core.graph (ComputeGraph)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.reactive.stream import (
    EventKind,
    OverflowStrategy,
    ReactiveStream,
    StreamEvent,
)


@dataclass
class DebounceState:
    """Internal state for the debounce operator.

    Tracks the last event time and pending event to determine
    when the quiet period has elapsed.
    """

    quiet_period: float
    last_event_time: float = 0.0
    pending_event: Optional[StreamEvent] = None
    suppressed_count: int = 0


def debounce(
    source: ReactiveStream,
    graph: ComputeGraph,
    quiet_period: float = 0.1,
) -> ReactiveStream:
    """Create a debounced stream that only emits after a quiet period.

    Events are suppressed until no new event arrives for `quiet_period`
    seconds. Only the most recent event in a burst is emitted.

    Args:
        source: The source stream to debounce.
        graph: ComputeGraph for the derived stream.
        quiet_period: Minimum seconds of silence before emitting.

    Returns:
        A new ReactiveStream that emits debounced events.
    """
    derived = ReactiveStream(graph, buffer_size=source.buffer_size)
    state = DebounceState(quiet_period=quiet_period)

    def on_event(event: StreamEvent) -> None:
        now = time.monotonic()
        elapsed = now - state.last_event_time

        if elapsed >= state.quiet_period and state.pending_event is not None:
            # Quiet period elapsed, emit the pending event
            derived.emit_changes({state.pending_event.node_id: state.pending_event.value})
            state.pending_event = None

        state.last_event_time = now
        state.pending_event = event
        state.suppressed_count += 1

    source.subscribe(on_event)
    return derived


@dataclass
class ThrottleState:
    """Internal state for the throttle operator."""

    interval: float
    last_emit_time: float = 0.0
    dropped_count: int = 0


def throttle(
    source: ReactiveStream,
    graph: ComputeGraph,
    interval: float = 0.1,
) -> ReactiveStream:
    """Create a throttled stream that emits at most once per interval.

    The first event in each interval window is emitted immediately.
    Subsequent events within the same window are dropped.

    Args:
        source: The source stream to throttle.
        graph: ComputeGraph for the derived stream.
        interval: Minimum seconds between emitted events.

    Returns:
        A new ReactiveStream with rate-limited events.
    """
    derived = ReactiveStream(graph, buffer_size=source.buffer_size)
    state = ThrottleState(interval=interval)

    def on_event(event: StreamEvent) -> None:
        now = time.monotonic()
        elapsed = now - state.last_emit_time

        if elapsed >= state.interval:
            state.last_emit_time = now
            derived.emit_changes({event.node_id: event.value})
        else:
            state.dropped_count += 1

    source.subscribe(on_event)
    return derived


@dataclass
class BufferState:
    """Internal state for the buffer operator."""

    max_size: int
    max_time: Optional[float]
    events: List[StreamEvent] = field(default_factory=list)
    first_event_time: float = 0.0
    flush_count: int = 0


def buffer(
    source: ReactiveStream,
    graph: ComputeGraph,
    max_size: int = 10,
    max_time: Optional[float] = None,
    on_flush: Optional[Callable[[List[StreamEvent]], None]] = None,
) -> ReactiveStream:
    """Buffer events and emit them in batches.

    Flushes when either the buffer reaches max_size or max_time seconds
    have elapsed since the first buffered event (whichever comes first).

    Args:
        source: The source stream to buffer.
        graph: ComputeGraph for the derived stream.
        max_size: Maximum events to buffer before flushing.
        max_time: Maximum seconds to hold events (None = no time limit).
        on_flush: Optional callback invoked with the batch on flush.

    Returns:
        A new ReactiveStream that emits batched events.
    """
    derived = ReactiveStream(graph, buffer_size=source.buffer_size)
    state = BufferState(max_size=max_size, max_time=max_time)

    def flush_buffer() -> None:
        """Flush accumulated events as a single batch emission."""
        if not state.events:
            return

        batch = list(state.events)
        state.events.clear()
        state.flush_count += 1

        if on_flush:
            on_flush(batch)

        # Emit a combined event with all node changes
        changes: Dict[str, Any] = {}
        for evt in batch:
            if evt.kind == EventKind.VALUE_CHANGED:
                changes[evt.node_id] = evt.value
        if changes:
            derived.emit_changes(changes)

    def on_event(event: StreamEvent) -> None:
        if not state.events:
            state.first_event_time = time.monotonic()

        state.events.append(event)

        # Check size trigger
        if len(state.events) >= state.max_size:
            flush_buffer()
            return

        # Check time trigger
        if state.max_time is not None:
            elapsed = time.monotonic() - state.first_event_time
            if elapsed >= state.max_time:
                flush_buffer()

    source.subscribe(on_event)

    # Attach flush method to derived stream for manual flushing
    derived._flush_buffer = flush_buffer  # type: ignore[attr-defined]
    return derived


def merge(
    *sources: ReactiveStream,
    graph: ComputeGraph,
) -> ReactiveStream:
    """Merge multiple streams into a single stream.

    Events from all source streams are forwarded to the merged stream
    in the order they arrive. The merged stream completes only when
    ALL source streams have completed.

    Args:
        sources: Two or more ReactiveStreams to merge.
        graph: ComputeGraph for the merged stream.

    Returns:
        A new ReactiveStream that emits events from all sources.
    """
    if len(sources) < 2:
        raise ValueError("merge requires at least 2 source streams")

    merged = ReactiveStream(graph, buffer_size=sum(s.buffer_size for s in sources))
    completed_count = [0]
    total_sources = len(sources)

    # Auto-observe all nodes observed by source streams
    for source in sources:
        for node_id in source._observed_nodes:
            merged._observed_nodes.add(node_id)

    def make_handler(source_index: int) -> Callable[[StreamEvent], None]:
        def on_event(event: StreamEvent) -> None:
            if event.kind == EventKind.COMPLETE:
                completed_count[0] += 1
                if completed_count[0] >= total_sources:
                    merged.complete()
            elif event.kind == EventKind.VALUE_CHANGED:
                merged.emit_changes({event.node_id: event.value})
            elif event.kind == EventKind.ERROR:
                merged.emit_error(event.node_id, Exception(event.metadata.get("error", "")))

        return on_event

    for i, source in enumerate(sources):
        source.subscribe(make_handler(i))

    return merged


@dataclass
class CombineLatestState:
    """Internal state for combine_latest operator."""

    latest_values: Dict[int, Optional[StreamEvent]]
    has_emitted: Dict[int, bool]
    emission_count: int = 0


def combine_latest(
    *sources: ReactiveStream,
    graph: ComputeGraph,
    combiner: Optional[Callable[[List[StreamEvent]], Dict[str, Any]]] = None,
) -> ReactiveStream:
    """Emit combined value whenever any source stream emits.

    Waits until all sources have emitted at least once, then emits
    a combined result on every subsequent event from any source.

    Args:
        sources: Two or more ReactiveStreams to combine.
        graph: ComputeGraph for the combined stream.
        combiner: Optional function to combine latest events into a
                  dict of changes. Default collects all node values.

    Returns:
        A new ReactiveStream emitting combined values.
    """
    if len(sources) < 2:
        raise ValueError("combine_latest requires at least 2 source streams")

    combined = ReactiveStream(graph, buffer_size=256)
    state = CombineLatestState(
        latest_values={i: None for i in range(len(sources))},
        has_emitted={i: False for i in range(len(sources))},
    )

    # Auto-observe all nodes observed by source streams
    for source in sources:
        for node_id in source._observed_nodes:
            combined._observed_nodes.add(node_id)

    def default_combiner(events: List[StreamEvent]) -> Dict[str, Any]:
        """Default combiner: collect all latest values by node_id."""
        result: Dict[str, Any] = {}
        for evt in events:
            if evt is not None and evt.kind == EventKind.VALUE_CHANGED:
                result[evt.node_id] = evt.value
        return result

    actual_combiner = combiner or default_combiner

    def make_handler(source_index: int) -> Callable[[StreamEvent], None]:
        def on_event(event: StreamEvent) -> None:
            if event.kind != EventKind.VALUE_CHANGED:
                return

            state.latest_values[source_index] = event
            state.has_emitted[source_index] = True

            # Only emit when all sources have produced at least one value
            if all(state.has_emitted.values()):
                latest_events = [
                    state.latest_values[i] for i in range(len(sources))
                ]
                changes = actual_combiner(latest_events)  # type: ignore[arg-type]
                if changes:
                    combined.emit_changes(changes)
                    state.emission_count += 1

        return on_event

    for i, source in enumerate(sources):
        source.subscribe(make_handler(i))

    return combined
