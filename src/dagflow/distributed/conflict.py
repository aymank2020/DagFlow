"""Conflict resolution for concurrent updates to shared nodes.

When multiple federated graphs attempt to update the same node
concurrently, conflicts arise. This module provides configurable
resolution strategies and maintains an audit trail of resolved conflicts.

This module depends on:
- core.node (InputNode, ComputeNode)
- distributed.sync (VectorClock, SyncState)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class ConflictPolicy(Enum):
    """Strategy for resolving concurrent update conflicts."""

    LAST_WRITER_WINS = auto()
    FIRST_WRITER_WINS = auto()
    HIGHEST_VALUE = auto()
    LOWEST_VALUE = auto()
    MERGE_FUNCTION = auto()
    REJECT = auto()


@dataclass
class ConflictRecord:
    """Record of a detected and resolved conflict.

    Attributes:
        node_id: The node where the conflict occurred.
        source_graph: Graph that produced the conflicting value.
        target_graph: Graph that already had a value.
        source_value: Value from the source graph.
        target_value: Existing value in the target graph.
        resolved_value: The value chosen after resolution.
        policy_used: Which policy resolved this conflict.
        timestamp: When the conflict was detected.
        source_clock: Source's vector clock at conflict time.
        target_clock: Target's vector clock at conflict time.
    """

    node_id: str
    source_graph: str
    target_graph: str
    source_value: Any
    target_value: Any
    resolved_value: Any
    policy_used: ConflictPolicy
    timestamp: float = field(default_factory=time.monotonic)
    source_clock: int = 0
    target_clock: int = 0

    @property
    def was_source_chosen(self) -> bool:
        """Whether the source value was chosen as the resolution."""
        return self.resolved_value == self.source_value

    @property
    def was_target_chosen(self) -> bool:
        """Whether the target (existing) value was kept."""
        return self.resolved_value == self.target_value


@dataclass
class ConflictStats:
    """Aggregate statistics about conflicts.

    Attributes:
        total_conflicts: Total conflicts detected.
        resolved_count: Conflicts successfully resolved.
        rejected_count: Conflicts that were rejected (no resolution).
        by_policy: Count of resolutions per policy type.
        by_node: Count of conflicts per node.
        by_graph_pair: Count of conflicts per graph pair.
    """

    total_conflicts: int = 0
    resolved_count: int = 0
    rejected_count: int = 0
    by_policy: Dict[ConflictPolicy, int] = field(default_factory=dict)
    by_node: Dict[str, int] = field(default_factory=dict)
    by_graph_pair: Dict[Tuple[str, str], int] = field(default_factory=dict)


class ConflictResolver:
    """Resolves conflicts when multiple graphs update shared nodes.

    Supports configurable per-node policies, custom merge functions,
    and maintains a full audit trail of all conflict resolutions.

    Usage:
        resolver = ConflictResolver(default_policy=ConflictPolicy.LAST_WRITER_WINS)
        resolver.set_node_policy("price", ConflictPolicy.HIGHEST_VALUE)
        resolved = resolver.resolve("price", "graph_a", 100, "graph_b", 95)
    """

    def __init__(
        self,
        default_policy: ConflictPolicy = ConflictPolicy.LAST_WRITER_WINS,
        max_history: int = 500,
    ) -> None:
        """Initialize the conflict resolver.

        Args:
            default_policy: Default resolution strategy.
            max_history: Maximum conflict records to retain.
        """
        self._default_policy = default_policy
        self._node_policies: Dict[str, ConflictPolicy] = {}
        self._merge_functions: Dict[str, Callable[[Any, Any], Any]] = {}
        self._history: List[ConflictRecord] = []
        self._max_history = max_history
        self._stats = ConflictStats()

    @property
    def history(self) -> List[ConflictRecord]:
        """Read-only access to conflict history."""
        return list(self._history)

    @property
    def stats(self) -> ConflictStats:
        """Current conflict statistics."""
        return self._stats

    def set_node_policy(self, node_id: str, policy: ConflictPolicy) -> None:
        """Set a specific conflict resolution policy for a node.

        Args:
            node_id: The node to configure.
            policy: The policy to use for this node's conflicts.
        """
        self._node_policies[node_id] = policy

    def set_merge_function(
        self, node_id: str, merge_fn: Callable[[Any, Any], Any]
    ) -> None:
        """Set a custom merge function for a node.

        The merge function receives (source_value, target_value) and
        returns the resolved value. Automatically sets the node's
        policy to MERGE_FUNCTION.

        Args:
            node_id: The node to configure.
            merge_fn: Function(source_val, target_val) -> resolved_val.
        """
        self._merge_functions[node_id] = merge_fn
        self._node_policies[node_id] = ConflictPolicy.MERGE_FUNCTION

    def resolve(
        self,
        node_id: str,
        source_graph: str,
        source_value: Any,
        target_graph: str,
        target_value: Any,
        source_clock: int = 0,
        target_clock: int = 0,
    ) -> Optional[Any]:
        """Resolve a conflict between two values for a node.

        Args:
            node_id: The conflicting node.
            source_graph: Graph providing the new value.
            source_value: The new value from source.
            target_graph: Graph with the existing value.
            target_value: The existing value in target.
            source_clock: Source's logical clock value.
            target_clock: Target's logical clock value.

        Returns:
            The resolved value, or None if policy is REJECT.
        """
        policy = self._node_policies.get(node_id, self._default_policy)
        resolved = self._apply_policy(
            policy, node_id, source_value, target_value,
            source_clock, target_clock
        )

        # Record the conflict
        record = ConflictRecord(
            node_id=node_id,
            source_graph=source_graph,
            target_graph=target_graph,
            source_value=source_value,
            target_value=target_value,
            resolved_value=resolved,
            policy_used=policy,
            source_clock=source_clock,
            target_clock=target_clock,
        )
        self._record_conflict(record)

        return resolved

    def resolve_batch(
        self,
        conflicts: List[Tuple[str, str, Any, str, Any]],
    ) -> Dict[str, Optional[Any]]:
        """Resolve multiple conflicts at once.

        Args:
            conflicts: List of (node_id, source_graph, source_val, target_graph, target_val).

        Returns:
            Dict of {node_id: resolved_value}.
        """
        results: Dict[str, Optional[Any]] = {}
        for node_id, src_graph, src_val, tgt_graph, tgt_val in conflicts:
            results[node_id] = self.resolve(
                node_id, src_graph, src_val, tgt_graph, tgt_val
            )
        return results

    def get_node_history(self, node_id: str) -> List[ConflictRecord]:
        """Get conflict history for a specific node."""
        return [r for r in self._history if r.node_id == node_id]

    def get_frequent_conflicts(self, min_count: int = 3) -> List[str]:
        """Find nodes that frequently have conflicts.

        Args:
            min_count: Minimum conflicts to be considered frequent.

        Returns:
            List of node IDs with frequent conflicts.
        """
        return [
            nid for nid, count in self._stats.by_node.items()
            if count >= min_count
        ]

    def clear_history(self) -> int:
        """Clear conflict history. Returns number of records cleared."""
        count = len(self._history)
        self._history.clear()
        return count

    def _apply_policy(
        self,
        policy: ConflictPolicy,
        node_id: str,
        source_value: Any,
        target_value: Any,
        source_clock: int,
        target_clock: int,
    ) -> Optional[Any]:
        """Apply a resolution policy to determine the winning value."""
        if policy == ConflictPolicy.LAST_WRITER_WINS:
            # Higher clock value wins
            return source_value if source_clock >= target_clock else target_value

        elif policy == ConflictPolicy.FIRST_WRITER_WINS:
            # Lower clock value wins (first to write)
            return source_value if source_clock <= target_clock else target_value

        elif policy == ConflictPolicy.HIGHEST_VALUE:
            try:
                return max(source_value, target_value)
            except TypeError:
                return source_value

        elif policy == ConflictPolicy.LOWEST_VALUE:
            try:
                return min(source_value, target_value)
            except TypeError:
                return source_value

        elif policy == ConflictPolicy.MERGE_FUNCTION:
            merge_fn = self._merge_functions.get(node_id)
            if merge_fn:
                return merge_fn(source_value, target_value)
            # Fallback to last-writer-wins if no merge function
            return source_value if source_clock >= target_clock else target_value

        elif policy == ConflictPolicy.REJECT:
            return None

        return source_value  # Default fallback

    def _record_conflict(self, record: ConflictRecord) -> None:
        """Record a conflict and update statistics."""
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Update stats
        self._stats.total_conflicts += 1
        if record.resolved_value is not None:
            self._stats.resolved_count += 1
        else:
            self._stats.rejected_count += 1

        policy = record.policy_used
        self._stats.by_policy[policy] = self._stats.by_policy.get(policy, 0) + 1
        self._stats.by_node[record.node_id] = (
            self._stats.by_node.get(record.node_id, 0) + 1
        )
        pair = (record.source_graph, record.target_graph)
        self._stats.by_graph_pair[pair] = (
            self._stats.by_graph_pair.get(pair, 0) + 1
        )
