"""GraphDiff — compare two graph states and report differences.

Compares node values, states, and generations between two points in time.
Works with Snapshot objects or directly with two graph/cache pairs.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode, NodeState)
- memo.cache (MemoCache)
- persistence.snapshot (Snapshot)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode, NodeState
from dagflow.memo.cache import MemoCache


class ChangeType(Enum):
    """Type of change detected between two states."""

    VALUE_CHANGED = auto()
    STATE_CHANGED = auto()
    NODE_ADDED = auto()
    NODE_REMOVED = auto()
    GENERATION_ADVANCED = auto()


@dataclass
class DiffEntry:
    """A single difference between two graph states.

    Attributes:
        node_id: The node that differs.
        change_type: What kind of change was detected.
        old_value: Value in the 'before' state.
        new_value: Value in the 'after' state.
        details: Optional additional context.
    """

    node_id: str
    change_type: ChangeType
    old_value: Any = None
    new_value: Any = None
    details: Optional[str] = None


class GraphDiff:
    """Computes differences between two graph states.

    Usage:
        diff = GraphDiff.compare_states(graph, cache_before, cache_after)
        for entry in diff.entries:
            print(f"{entry.node_id}: {entry.change_type.name}")

    Or using snapshots:
        diff = GraphDiff.compare_snapshots(snap_before, snap_after)
    """

    def __init__(self, entries: Optional[List[DiffEntry]] = None) -> None:
        self._entries: List[DiffEntry] = entries or []

    @property
    def entries(self) -> List[DiffEntry]:
        """All diff entries."""
        return list(self._entries)

    @property
    def changed_nodes(self) -> Set[str]:
        """Set of node IDs that have any difference."""
        return {e.node_id for e in self._entries}

    @property
    def has_changes(self) -> bool:
        """Whether any differences were found."""
        return len(self._entries) > 0

    @property
    def change_count(self) -> int:
        """Total number of differences."""
        return len(self._entries)

    def filter_by_type(self, change_type: ChangeType) -> List[DiffEntry]:
        """Get entries of a specific change type."""
        return [e for e in self._entries if e.change_type == change_type]

    @classmethod
    def compare_caches(
        cls,
        graph: ComputeGraph,
        before: MemoCache,
        after: MemoCache,
    ) -> "GraphDiff":
        """Compare two cache states for the same graph.

        Args:
            graph: The computation graph (structure reference).
            before: Cache state before changes.
            after: Cache state after changes.

        Returns:
            GraphDiff with all detected differences.
        """
        entries: List[DiffEntry] = []

        for node_id in graph.get_compute_nodes():
            old_val = before.get(node_id)
            new_val = after.get(node_id)

            if old_val != new_val:
                entries.append(
                    DiffEntry(
                        node_id=node_id,
                        change_type=ChangeType.VALUE_CHANGED,
                        old_value=old_val,
                        new_value=new_val,
                    )
                )

            old_gen = before.get_generation(node_id)
            new_gen = after.get_generation(node_id)
            if new_gen > old_gen:
                entries.append(
                    DiffEntry(
                        node_id=node_id,
                        change_type=ChangeType.GENERATION_ADVANCED,
                        old_value=old_gen,
                        new_value=new_gen,
                        details=f"Generation advanced from {old_gen} to {new_gen}",
                    )
                )

        return cls(entries)

    @classmethod
    def compare_snapshots(
        cls,
        before: Any,  # Snapshot type (avoid circular import)
        after: Any,
    ) -> "GraphDiff":
        """Compare two Snapshot objects.

        Args:
            before: The earlier snapshot.
            after: The later snapshot.

        Returns:
            GraphDiff with all detected differences.
        """
        entries: List[DiffEntry] = []
        all_ids = set(before.node_records.keys()) | set(after.node_records.keys())

        for node_id in all_ids:
            before_rec = before.node_records.get(node_id)
            after_rec = after.node_records.get(node_id)

            if before_rec is None:
                entries.append(
                    DiffEntry(
                        node_id=node_id,
                        change_type=ChangeType.NODE_ADDED,
                        new_value=after_rec.value if after_rec else None,
                    )
                )
                continue

            if after_rec is None:
                entries.append(
                    DiffEntry(
                        node_id=node_id,
                        change_type=ChangeType.NODE_REMOVED,
                        old_value=before_rec.value,
                    )
                )
                continue

            # Both exist — compare values
            if before_rec.value != after_rec.value:
                entries.append(
                    DiffEntry(
                        node_id=node_id,
                        change_type=ChangeType.VALUE_CHANGED,
                        old_value=before_rec.value,
                        new_value=after_rec.value,
                    )
                )

            # Compare state (for compute nodes)
            if before_rec.state != after_rec.state:
                entries.append(
                    DiffEntry(
                        node_id=node_id,
                        change_type=ChangeType.STATE_CHANGED,
                        old_value=before_rec.state,
                        new_value=after_rec.state,
                    )
                )

            # Compare generation
            if after_rec.generation > before_rec.generation:
                entries.append(
                    DiffEntry(
                        node_id=node_id,
                        change_type=ChangeType.GENERATION_ADVANCED,
                        old_value=before_rec.generation,
                        new_value=after_rec.generation,
                    )
                )

        return cls(entries)

    def summary(self) -> str:
        """Human-readable summary of differences.

        Returns:
            Multi-line string describing all changes.
        """
        if not self._entries:
            return "No differences found."

        lines = [f"Found {len(self._entries)} difference(s):"]
        for entry in self._entries:
            line = f"  [{entry.change_type.name}] {entry.node_id}"
            if entry.old_value is not None or entry.new_value is not None:
                line += f": {entry.old_value!r} -> {entry.new_value!r}"
            if entry.details:
                line += f" ({entry.details})"
            lines.append(line)

        return "\n".join(lines)
