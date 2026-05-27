"""Checkpoint — save and restore graph state for rollback.

Captures a point-in-time snapshot of all node values and states,
allowing the graph to be rolled back to a previous consistent state.

This module depends on:
- core.graph (ComputeGraph) for node access
- core.node (InputNode, ComputeNode, NodeState) for state capture
- memo.cache (MemoCache) for cached value capture
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode, NodeState
from dagflow.memo.cache import MemoCache


@dataclass
class NodeSnapshot:
    """Captured state of a single node at checkpoint time."""

    node_id: str
    is_input: bool
    value: Any = None
    generation: int = 0
    state: Optional[NodeState] = None
    last_valid_generation: Dict[str, int] = field(default_factory=dict)


@dataclass
class CacheSnapshot:
    """Captured state of the memo cache at checkpoint time."""

    entries: Dict[str, Tuple[Any, int, Dict[str, int]]] = field(default_factory=dict)


class Checkpoint:
    """Captures and restores graph + cache state.

    Usage:
        cp = Checkpoint(graph, cache)
        cp.capture()       # Save current state
        # ... make changes ...
        cp.restore()       # Roll back to saved state

    The checkpoint stores a deep copy of all mutable state, so restoring
    is always safe regardless of what mutations occurred in between.
    """

    def __init__(self, graph: ComputeGraph, cache: MemoCache) -> None:
        self._graph = graph
        self._cache = cache
        self._node_snapshots: Dict[str, NodeSnapshot] = {}
        self._cache_snapshot: Optional[CacheSnapshot] = None
        self._captured: bool = False

    @property
    def is_captured(self) -> bool:
        """Whether a checkpoint has been captured."""
        return self._captured

    def capture(self) -> None:
        """Capture current state of all nodes and the cache.

        Overwrites any previously captured state.
        """
        self._node_snapshots.clear()

        for node_id in self._graph.all_node_ids:
            node = self._graph.get_node(node_id)
            if isinstance(node, InputNode):
                snap = NodeSnapshot(
                    node_id=node_id,
                    is_input=True,
                    value=deepcopy(node.value),
                    generation=node.generation,
                )
            elif isinstance(node, ComputeNode):
                snap = NodeSnapshot(
                    node_id=node_id,
                    is_input=False,
                    value=deepcopy(node.cached_value),
                    state=node.state,
                    last_valid_generation=dict(node.last_valid_generation),
                )
            else:
                continue
            self._node_snapshots[node_id] = snap

        # Capture cache state
        self._cache_snapshot = CacheSnapshot(
            entries=deepcopy(self._cache._entries)
        )
        self._captured = True

    def restore(self) -> None:
        """Restore graph and cache to the captured state.

        Raises RuntimeError if no checkpoint has been captured.
        """
        if not self._captured:
            raise RuntimeError("No checkpoint captured — call capture() first")

        for node_id, snap in self._node_snapshots.items():
            node = self._graph.get_node(node_id)
            if isinstance(node, InputNode) and snap.is_input:
                node.value = deepcopy(snap.value)
                node.generation = snap.generation
            elif isinstance(node, ComputeNode) and not snap.is_input:
                node.cached_value = deepcopy(snap.value)
                node.state = snap.state  # type: ignore[assignment]
                node.last_valid_generation = dict(snap.last_valid_generation)

        # Restore cache
        if self._cache_snapshot is not None:
            self._cache._entries = deepcopy(self._cache_snapshot.entries)

    def discard(self) -> None:
        """Discard the captured checkpoint, freeing memory."""
        self._node_snapshots.clear()
        self._cache_snapshot = None
        self._captured = False
