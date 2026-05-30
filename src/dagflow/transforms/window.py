"""Window — sliding window computations over node value history.

Maintains a buffer of the last N values produced by a node, enabling
computations like moving averages, trend detection, and rate-of-change
calculations.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode


class WindowBuffer:
    """Fixed-size buffer that maintains the last N values.

    Acts as a sliding window over a stream of values. When the buffer
    is full, the oldest value is evicted to make room for the new one.

    Attributes:
        max_size: Maximum number of values to retain.
    """

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("Window size must be at least 1")
        self._max_size = max_size
        self._buffer: Deque[Any] = deque(maxlen=max_size)

    @property
    def max_size(self) -> int:
        """Maximum window size."""
        return self._max_size

    @property
    def current_size(self) -> int:
        """Number of values currently in the buffer."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached its maximum size."""
        return len(self._buffer) == self._max_size

    @property
    def values(self) -> List[Any]:
        """Current values in the buffer (oldest first)."""
        return list(self._buffer)

    def push(self, value: Any) -> Optional[Any]:
        """Add a value to the buffer.

        Args:
            value: The value to add.

        Returns:
            The evicted value if the buffer was full, None otherwise.
        """
        evicted = None
        if self.is_full:
            evicted = self._buffer[0]
        self._buffer.append(value)
        return evicted

    def peek_oldest(self) -> Optional[Any]:
        """Get the oldest value without removing it."""
        return self._buffer[0] if self._buffer else None

    def peek_newest(self) -> Optional[Any]:
        """Get the newest value without removing it."""
        return self._buffer[-1] if self._buffer else None

    def clear(self) -> None:
        """Remove all values from the buffer."""
        self._buffer.clear()


class WindowNode:
    """Factory for creating sliding-window compute nodes.

    A WindowNode maintains a history buffer and computes an aggregate
    function over the window contents each time the source value changes.

    Usage:
        window = WindowNode(graph, window_size=5)
        window.create_moving_average("avg_temp", source="temperature")
        window.create_custom("trend", source="price",
                            compute=lambda values: values[-1] - values[0])
    """

    def __init__(self, graph: ComputeGraph, window_size: int = 10) -> None:
        self._graph = graph
        self._window_size = window_size
        self._buffers: Dict[str, WindowBuffer] = {}

    @property
    def window_size(self) -> int:
        """Default window size for new nodes."""
        return self._window_size

    def get_buffer(self, node_id: str) -> Optional[WindowBuffer]:
        """Get the buffer associated with a window node."""
        return self._buffers.get(node_id)

    def _attach_buffer(self, node: ComputeNode, buffer: WindowBuffer) -> ComputeNode:
        """Expose a window buffer to graph-level checkpointing."""
        setattr(node, "_dagflow_window_buffer", buffer)
        return node

    def create_moving_average(
        self,
        node_id: str,
        source: str,
        size: Optional[int] = None,
        priority: int = 0,
    ) -> ComputeNode:
        """Create a moving average window node.

        Args:
            node_id: ID for the new node.
            source: ID of the source node.
            size: Window size (defaults to self.window_size).
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        buf_size = size or self._window_size
        buffer = WindowBuffer(buf_size)
        self._buffers[node_id] = buffer

        def avg_func(deps: Dict[str, Any]) -> Any:
            value = deps[source]
            buffer.push(value)
            values = buffer.values
            if not values:
                return 0.0
            return sum(values) / len(values)

        return self._attach_buffer(
            self._graph.add_compute(
                node_id, func=avg_func, dependencies=[source], priority=priority
            ),
            buffer,
        )

    def create_moving_sum(
        self,
        node_id: str,
        source: str,
        size: Optional[int] = None,
        priority: int = 0,
    ) -> ComputeNode:
        """Create a moving sum window node.

        Args:
            node_id: ID for the new node.
            source: ID of the source node.
            size: Window size.
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        buf_size = size or self._window_size
        buffer = WindowBuffer(buf_size)
        self._buffers[node_id] = buffer

        def sum_func(deps: Dict[str, Any]) -> Any:
            value = deps[source]
            buffer.push(value)
            return sum(buffer.values)

        return self._attach_buffer(
            self._graph.add_compute(
                node_id, func=sum_func, dependencies=[source], priority=priority
            ),
            buffer,
        )

    def create_custom(
        self,
        node_id: str,
        source: str,
        compute: Callable[[List[Any]], Any],
        size: Optional[int] = None,
        priority: int = 0,
    ) -> ComputeNode:
        """Create a custom window computation node.

        Args:
            node_id: ID for the new node.
            source: ID of the source node.
            compute: Function that receives the window contents and returns a value.
            size: Window size.
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        buf_size = size or self._window_size
        buffer = WindowBuffer(buf_size)
        self._buffers[node_id] = buffer

        def custom_func(deps: Dict[str, Any]) -> Any:
            value = deps[source]
            buffer.push(value)
            return compute(buffer.values)

        return self._attach_buffer(
            self._graph.add_compute(
                node_id, func=custom_func, dependencies=[source], priority=priority
            ),
            buffer,
        )
