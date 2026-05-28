"""Snapshot — immutable point-in-time captures of graph state.

Unlike Checkpoint (which is mutable and tied to a single graph), Snapshots
are immutable, timestamped records that can be stored and compared. The
SnapshotStore manages a collection of named snapshots.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode, NodeState)
- memo.cache (MemoCache)
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode, NodeState
from dagflow.memo.cache import MemoCache


@dataclass(frozen=True)
class NodeRecord:
    """Immutable record of a node's state at snapshot time."""

    node_id: str
    is_input: bool
    value: Any
    generation: int
    state: Optional[str]  # NodeState name or None for inputs


@dataclass
class Snapshot:
    """Immutable point-in-time capture of the entire graph state.

    Attributes:
        name: Human-readable identifier for this snapshot.
        timestamp: Unix timestamp when the snapshot was taken.
        node_records: Dict of node_id -> NodeRecord.
        metadata: Optional user-provided metadata dict.
    """

    name: str
    timestamp: float
    node_records: Dict[str, NodeRecord] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        name: str,
        graph: ComputeGraph,
        cache: MemoCache,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Snapshot":
        """Create a snapshot from the current graph state.

        Args:
            name: Identifier for this snapshot.
            graph: The computation graph to capture.
            cache: The memo cache with current values.
            metadata: Optional metadata to attach.

        Returns:
            A new immutable Snapshot instance.
        """
        records: Dict[str, NodeRecord] = {}

        for node_id in graph.all_node_ids:
            node = graph.get_node(node_id)
            if isinstance(node, InputNode):
                records[node_id] = NodeRecord(
                    node_id=node_id,
                    is_input=True,
                    value=deepcopy(node.value),
                    generation=node.generation,
                    state=None,
                )
            elif isinstance(node, ComputeNode):
                cached_val = cache.get(node_id)
                records[node_id] = NodeRecord(
                    node_id=node_id,
                    is_input=False,
                    value=deepcopy(cached_val),
                    generation=cache.get_generation(node_id),
                    state=node.state.name,
                )

        return cls(
            name=name,
            timestamp=time.time(),
            node_records=records,
            metadata=metadata or {},
        )

    def get_value(self, node_id: str) -> Any:
        """Get the captured value of a node.

        Raises KeyError if node_id is not in this snapshot.
        """
        if node_id not in self.node_records:
            raise KeyError(f"Node '{node_id}' not in snapshot '{self.name}'")
        return self.node_records[node_id].value

    def get_all_values(self) -> Dict[str, Any]:
        """Return a dict of {node_id: value} for all captured nodes."""
        return {nid: rec.value for nid, rec in self.node_records.items()}

    @property
    def node_count(self) -> int:
        """Number of nodes captured in this snapshot."""
        return len(self.node_records)

    def diff(self, other: "Snapshot") -> Dict[str, Tuple[Any, Any]]:
        """Compare this snapshot with another, returning differences.

        Args:
            other: Another snapshot to compare against.

        Returns:
            Dict of {node_id: (self_value, other_value)} for nodes that differ.
            Nodes present in only one snapshot are included with None for the
            missing side.
        """
        all_ids = set(self.node_records.keys()) | set(other.node_records.keys())
        differences: Dict[str, Tuple[Any, Any]] = {}

        for node_id in all_ids:
            self_val = self.node_records.get(node_id)
            other_val = other.node_records.get(node_id)

            self_value = self_val.value if self_val else None
            other_value = other_val.value if other_val else None

            if self_value != other_value:
                differences[node_id] = (self_value, other_value)

        return differences


class SnapshotStore:
    """Manages a collection of named snapshots.

    Provides storage, retrieval, and comparison of multiple snapshots.
    Useful for tracking graph state evolution over time.
    """

    def __init__(self) -> None:
        self._snapshots: Dict[str, Snapshot] = {}
        self._history: List[str] = []  # Ordered list of snapshot names

    def save(
        self,
        name: str,
        graph: ComputeGraph,
        cache: MemoCache,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Snapshot:
        """Capture and store a new snapshot.

        Args:
            name: Unique name for this snapshot.
            graph: The graph to capture.
            cache: The cache to capture.
            metadata: Optional metadata.

        Returns:
            The newly created Snapshot.

        Raises:
            ValueError: If a snapshot with this name already exists.
        """
        if name in self._snapshots:
            raise ValueError(f"Snapshot '{name}' already exists")

        snap = Snapshot.capture(name, graph, cache, metadata)
        self._snapshots[name] = snap
        self._history.append(name)
        return snap

    def get(self, name: str) -> Snapshot:
        """Retrieve a snapshot by name.

        Raises KeyError if not found.
        """
        if name not in self._snapshots:
            raise KeyError(f"Snapshot '{name}' not found")
        return self._snapshots[name]

    def list_snapshots(self) -> List[str]:
        """Return snapshot names in chronological order."""
        return list(self._history)

    def delete(self, name: str) -> None:
        """Remove a snapshot from the store."""
        if name in self._snapshots:
            del self._snapshots[name]
            self._history.remove(name)

    @property
    def count(self) -> int:
        """Number of stored snapshots."""
        return len(self._snapshots)
