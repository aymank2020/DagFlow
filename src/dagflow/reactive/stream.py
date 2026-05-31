"""ReactiveStream — push-based change notification with backpressure.

Provides a stream abstraction that emits events when DAG nodes change value.
Supports backpressure via bounded internal buffers and configurable overflow
strategies (drop-oldest, drop-newest, block).

This module depends on:
- core.graph (ComputeGraph) for node observation
- core.node (InputNode, ComputeNode) for value access
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, Generic, List, Optional, TypeVar

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode

T = TypeVar("T")


class EventKind(Enum):
    """Classification of stream events."""

    VALUE_CHANGED = auto()
    NODE_ADDED = auto()
    NODE_REMOVED = auto()
    ERROR = auto()
    COMPLETE = auto()


class OverflowStrategy(Enum):
    """Strategy when the internal buffer is full."""

    DROP_OLDEST = auto()
    DROP_NEWEST = auto()
    ERROR = auto()


@dataclass(frozen=True)
class StreamEvent:
    """An immutable event emitted by a ReactiveStream.

    Attributes:
        kind: The type of event.
        node_id: The node that triggered this event.
        value: The new value (for VALUE_CHANGED events).
        old_value: The previous value (for VALUE_CHANGED events).
        timestamp: Monotonic time when the event was created.
        metadata: Optional extra context about the event.
    """

    kind: EventKind
    node_id: str
    value: Any = None
    old_value: Any = None
    timestamp: float = field(default_factory=time.monotonic)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReactiveStream:
    """A push-based stream of DAG change events with backpressure.

    The stream observes a set of nodes in a ComputeGraph and emits
    StreamEvent objects to registered listeners whenever those nodes
    change value. Backpressure is managed via a bounded internal buffer.

    Usage:
        stream = ReactiveStream(graph, buffer_size=100)
        stream.observe("node_a", "node_b")
        stream.subscribe(lambda event: print(event))
        # ... after propagation ...
        stream.emit_changes({"node_a": 42})
    """

    def __init__(
        self,
        graph: ComputeGraph,
        buffer_size: int = 256,
        overflow: OverflowStrategy = OverflowStrategy.DROP_OLDEST,
    ) -> None:
        self._graph = graph
        self._buffer_size = max(1, buffer_size)
        self._overflow = overflow
        self._buffer: Deque[StreamEvent] = deque(maxlen=None)
        self._listeners: List[Callable[[StreamEvent], None]] = []
        self._observed_nodes: set[str] = set()
        self._paused: bool = False
        self._completed: bool = False
        self._dropped_count: int = 0

    @property
    def buffer_size(self) -> int:
        """Maximum number of events buffered before overflow handling."""
        return self._buffer_size

    @property
    def pending_count(self) -> int:
        """Number of events currently buffered."""
        return len(self._buffer)

    @property
    def is_paused(self) -> bool:
        """Whether the stream is currently paused."""
        return self._paused

    @property
    def is_completed(self) -> bool:
        """Whether the stream has been completed (no more events)."""
        return self._completed

    @property
    def dropped_count(self) -> int:
        """Number of events dropped due to overflow."""
        return self._dropped_count

    def observe(self, *node_ids: str) -> None:
        """Register nodes to observe for changes.

        Args:
            node_ids: One or more node IDs to watch.

        Raises:
            KeyError: If a node_id doesn't exist in the graph.
        """
        for nid in node_ids:
            self._graph.get_node(nid)  # Validates existence
            self._observed_nodes.add(nid)

    def unobserve(self, *node_ids: str) -> None:
        """Stop observing specified nodes."""
        for nid in node_ids:
            self._observed_nodes.discard(nid)

    def subscribe(self, listener: Callable[[StreamEvent], None]) -> int:
        """Register a listener that receives events.

        Args:
            listener: Callable that accepts a StreamEvent.

        Returns:
            Index of the listener (for unsubscribe).
        """
        self._listeners.append(listener)
        return len(self._listeners) - 1

    def unsubscribe(self, index: int) -> None:
        """Remove a listener by index. Sets slot to no-op."""
        if 0 <= index < len(self._listeners):
            self._listeners[index] = lambda _: None

    def emit_changes(self, recomputed: Dict[str, Any]) -> int:
        """Emit events for nodes that changed value.

        Called after propagation with the dict of recomputed values.
        Only emits for nodes in the observed set.

        Args:
            recomputed: Dict of {node_id: new_value} from propagation.

        Returns:
            Number of events emitted.
        """
        if self._completed:
            return 0

        emitted = 0
        for node_id, new_value in recomputed.items():
            if node_id not in self._observed_nodes:
                continue

            node = self._graph.get_node(node_id)
            old_value = None
            if isinstance(node, ComputeNode):
                old_value = node.cached_value if node.cached_value != new_value else None

            event = StreamEvent(
                kind=EventKind.VALUE_CHANGED,
                node_id=node_id,
                value=new_value,
                old_value=old_value,
            )
            self._push_event(event)
            emitted += 1

        return emitted

    def emit_error(self, node_id: str, error: Exception) -> None:
        """Emit an error event for a node."""
        if self._completed:
            return
        event = StreamEvent(
            kind=EventKind.ERROR,
            node_id=node_id,
            metadata={"error": str(error), "type": type(error).__name__},
        )
        self._push_event(event)

    def complete(self) -> None:
        """Mark the stream as completed. No further events will be emitted."""
        if not self._completed:
            self._completed = True
            event = StreamEvent(
                kind=EventKind.COMPLETE,
                node_id="__stream__",
            )
            self._deliver(event)

    def pause(self) -> None:
        """Pause event delivery. Events are buffered but not delivered."""
        self._paused = True

    def resume(self) -> int:
        """Resume delivery and flush buffered events.

        Returns:
            Number of buffered events that were delivered.
        """
        self._paused = False
        delivered = 0
        while self._buffer:
            event = self._buffer.popleft()
            self._deliver(event)
            delivered += 1
        return delivered

    def drain(self) -> List[StreamEvent]:
        """Remove and return all buffered events without delivering them."""
        events = list(self._buffer)
        self._buffer.clear()
        return events

    def _push_event(self, event: StreamEvent) -> None:
        """Push an event, handling backpressure via overflow strategy."""
        if self._paused:
            if len(self._buffer) >= self._buffer_size:
                self._handle_overflow()
            self._buffer.append(event)
        else:
            self._deliver(event)

    def _deliver(self, event: StreamEvent) -> None:
        """Deliver an event to all registered listeners."""
        for listener in self._listeners:
            listener(event)

    def _handle_overflow(self) -> None:
        """Apply overflow strategy when buffer is full."""
        if self._overflow == OverflowStrategy.DROP_OLDEST:
            if self._buffer:
                self._buffer.popleft()
                self._dropped_count += 1
        elif self._overflow == OverflowStrategy.DROP_NEWEST:
            self._dropped_count += 1
        elif self._overflow == OverflowStrategy.ERROR:
            raise BufferError(
                f"Stream buffer overflow: {len(self._buffer)} events buffered"
            )
