"""ReplayLog — record and replay sequences of input changes.

Captures a time-ordered log of input mutations applied to a graph,
allowing the exact sequence to be replayed on a fresh graph (or the
same graph after a reset). Useful for debugging, testing, and
reproducing specific computation states.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode)
- memo.cache (MemoCache)
- propagation.eager (EagerPropagator)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator


@dataclass
class ReplayEntry:
    """A single recorded input change.

    Attributes:
        timestamp: When the change was recorded (unix time).
        node_id: The input node that was changed.
        old_value: Value before the change.
        new_value: Value after the change.
        sequence_num: Monotonic sequence number within the log.
    """

    timestamp: float
    node_id: str
    old_value: Any
    new_value: Any
    sequence_num: int


class ReplayLog:
    """Records and replays input change sequences.

    Recording mode:
        log = ReplayLog(graph)
        log.start_recording()
        # ... make changes via log.record_change() ...
        log.stop_recording()

    Replay mode:
        log.replay(target_graph, target_cache)
        # or replay up to a specific point:
        log.replay_until(target_graph, target_cache, sequence_num=5)
    """

    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph
        self._entries: List[ReplayEntry] = []
        self._recording: bool = False
        self._sequence_counter: int = 0

    @property
    def is_recording(self) -> bool:
        """Whether the log is currently recording changes."""
        return self._recording

    @property
    def entry_count(self) -> int:
        """Number of recorded entries."""
        return len(self._entries)

    @property
    def entries(self) -> List[ReplayEntry]:
        """Read-only access to recorded entries."""
        return list(self._entries)

    def start_recording(self) -> None:
        """Begin recording input changes."""
        self._recording = True

    def stop_recording(self) -> None:
        """Stop recording input changes."""
        self._recording = False

    def record_change(self, node_id: str, new_value: Any) -> ReplayEntry:
        """Record an input change and apply it to the graph.

        Args:
            node_id: The input node to change.
            new_value: The new value to set.

        Returns:
            The recorded ReplayEntry.

        Raises:
            RuntimeError: If not currently recording.
            ValueError: If node_id is not an input node.
        """
        if not self._recording:
            raise RuntimeError("Not recording — call start_recording() first")

        node = self._graph.get_node(node_id)
        if not isinstance(node, InputNode):
            raise ValueError(f"Node '{node_id}' is not an input node")

        old_value = node.value
        node.set(new_value)

        self._sequence_counter += 1
        entry = ReplayEntry(
            timestamp=time.time(),
            node_id=node_id,
            old_value=old_value,
            new_value=new_value,
            sequence_num=self._sequence_counter,
        )
        self._entries.append(entry)
        return entry

    def replay(
        self,
        target_graph: ComputeGraph,
        target_cache: MemoCache,
        propagate: bool = True,
    ) -> List[Dict[str, Any]]:
        """Replay all recorded changes on a target graph.

        Args:
            target_graph: The graph to replay changes on.
            target_cache: The cache for the target graph.
            propagate: Whether to propagate after each change.

        Returns:
            List of propagation results (one per entry), or empty dicts
            if propagate=False.
        """
        return self.replay_until(
            target_graph, target_cache, len(self._entries), propagate
        )

    def replay_until(
        self,
        target_graph: ComputeGraph,
        target_cache: MemoCache,
        sequence_num: int,
        propagate: bool = True,
    ) -> List[Dict[str, Any]]:
        """Replay changes up to (and including) a given sequence number.

        Args:
            target_graph: The graph to replay on.
            target_cache: The cache for the target graph.
            sequence_num: Stop after this sequence number.
            propagate: Whether to propagate after each change.

        Returns:
            List of propagation results for each replayed entry.
        """
        propagator = EagerPropagator(target_graph, target_cache) if propagate else None
        results: List[Dict[str, Any]] = []

        for entry in self._entries:
            if entry.sequence_num > sequence_num:
                break

            node = target_graph.get_node(entry.node_id)
            if isinstance(node, InputNode):
                node.set(entry.new_value)

                if propagator is not None:
                    result = propagator.propagate([entry.node_id])
                    results.append(result)
                else:
                    results.append({})

        return results

    def clear(self) -> None:
        """Clear all recorded entries."""
        self._entries.clear()
        self._sequence_counter = 0

    def get_changes_for_node(self, node_id: str) -> List[ReplayEntry]:
        """Get all recorded changes for a specific node.

        Args:
            node_id: The node to filter by.

        Returns:
            List of entries affecting this node, in order.
        """
        return [e for e in self._entries if e.node_id == node_id]

    def to_dict(self) -> List[Dict[str, Any]]:
        """Export the log as a list of dictionaries (JSON-serializable).

        Returns:
            List of entry dictionaries.
        """
        return [
            {
                "sequence_num": e.sequence_num,
                "timestamp": e.timestamp,
                "node_id": e.node_id,
                "old_value": e.old_value,
                "new_value": e.new_value,
            }
            for e in self._entries
        ]

    @classmethod
    def from_dict(
        cls, entries: List[Dict[str, Any]], graph: ComputeGraph
    ) -> "ReplayLog":
        """Reconstruct a ReplayLog from exported dictionaries.

        Args:
            entries: List of entry dicts (from to_dict()).
            graph: The graph this log is associated with.

        Returns:
            A new ReplayLog with the entries loaded.
        """
        log = cls(graph)
        for entry_data in entries:
            entry = ReplayEntry(
                timestamp=entry_data["timestamp"],
                node_id=entry_data["node_id"],
                old_value=entry_data["old_value"],
                new_value=entry_data["new_value"],
                sequence_num=entry_data["sequence_num"],
            )
            log._entries.append(entry)
        if log._entries:
            log._sequence_counter = log._entries[-1].sequence_num
        return log
