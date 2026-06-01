"""Synchronization protocol between federated graphs.

Implements a vector-clock-based synchronization protocol that ensures
consistent state across federated graphs. Handles ordering of cross-graph
updates, detects stale reads, and provides barrier synchronization.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode)
- memo.cache (MemoCache)
- distributed.federation (FederatedGraph, GraphEndpoint)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode
from dagflow.memo.cache import MemoCache


class SyncState(Enum):
    """State of synchronization between two graphs."""

    IN_SYNC = auto()
    STALE = auto()
    DIVERGED = auto()
    UNKNOWN = auto()


@dataclass
class VectorClock:
    """Logical vector clock for ordering distributed events.

    Each graph maintains a counter that increments on every mutation.
    The vector clock captures the "knowledge" of each graph about
    the state of all other graphs.
    """

    clocks: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def increment(self, graph_name: str) -> int:
        """Increment the clock for a graph. Returns new value."""
        self.clocks[graph_name] = self.clocks.get(graph_name, 0) + 1
        return self.clocks[graph_name]

    def get(self, graph_name: str) -> int:
        """Get current clock value for a graph."""
        return self.clocks.get(graph_name, 0)

    def merge(self, other: "VectorClock") -> None:
        """Merge another vector clock (take max of each component)."""
        for name, value in other.clocks.items():
            self.clocks[name] = max(self.clocks.get(name, 0), value)

    def dominates(self, other: "VectorClock") -> bool:
        """Check if this clock dominates (is causally after) another."""
        all_keys = set(self.clocks.keys()) | set(other.clocks.keys())
        at_least_one_greater = False
        for key in all_keys:
            mine = self.clocks.get(key, 0)
            theirs = other.clocks.get(key, 0)
            if mine < theirs:
                return False
            if mine > theirs:
                at_least_one_greater = True
        return at_least_one_greater

    def concurrent_with(self, other: "VectorClock") -> bool:
        """Check if two clocks are concurrent (neither dominates)."""
        return not self.dominates(other) and not other.dominates(self)

    def snapshot(self) -> Dict[str, int]:
        """Return a frozen snapshot of current clock values."""
        return dict(self.clocks)


@dataclass
class SyncResult:
    """Result of a synchronization operation.

    Attributes:
        state: The sync state after the operation.
        synced_nodes: Nodes that were synchronized.
        stale_nodes: Nodes that are still stale after sync.
        conflicts: Node IDs where conflicts were detected.
        vector_clock: The vector clock state after sync.
    """

    state: SyncState
    synced_nodes: Set[str] = field(default_factory=set)
    stale_nodes: Set[str] = field(default_factory=set)
    conflicts: Set[str] = field(default_factory=set)
    vector_clock: Optional[VectorClock] = None


@dataclass
class SyncBarrier:
    """A synchronization barrier that waits for multiple graphs.

    All participating graphs must reach the barrier before any
    can proceed past it. Used for coordinated checkpoints.
    """

    participants: FrozenSet[str]
    arrived: Set[str] = field(default_factory=set)
    released: bool = False

    @property
    def is_complete(self) -> bool:
        """Whether all participants have arrived."""
        return self.participants <= frozenset(self.arrived)

    def arrive(self, graph_name: str) -> bool:
        """Mark a graph as having arrived at the barrier.

        Returns:
            True if this arrival completes the barrier.
        """
        if graph_name not in self.participants:
            return False
        self.arrived.add(graph_name)
        if self.is_complete:
            self.released = True
            return True
        return False

    @property
    def pending(self) -> Set[str]:
        """Graphs that haven't arrived yet."""
        return self.participants - frozenset(self.arrived)


class SyncProtocol:
    """Synchronization protocol for federated graph coordination.

    Maintains vector clocks for each graph and provides operations
    for checking sync state, performing synchronization, and
    coordinating barriers.

    Usage:
        protocol = SyncProtocol()
        protocol.register_graph("pricing")
        protocol.register_graph("inventory")
        protocol.record_mutation("pricing")
        state = protocol.check_sync("pricing", "inventory")
    """

    def __init__(self) -> None:
        self._clocks: Dict[str, VectorClock] = {}
        self._last_sync: Dict[Tuple[str, str], VectorClock] = {}
        self._barriers: List[SyncBarrier] = []
        self._mutation_log: List[Tuple[str, int, str]] = []  # (graph, clock, node_id)

    def register_graph(self, name: str) -> None:
        """Register a graph for synchronization tracking."""
        if name not in self._clocks:
            self._clocks[name] = VectorClock()

    def record_mutation(self, graph_name: str, node_id: str = "") -> int:
        """Record that a graph has been mutated.

        Increments the graph's vector clock component.

        Args:
            graph_name: The graph that was mutated.
            node_id: Optional node that was changed.

        Returns:
            New clock value for this graph.
        """
        if graph_name not in self._clocks:
            self.register_graph(graph_name)

        clock = self._clocks[graph_name]
        new_value = clock.increment(graph_name)
        self._mutation_log.append((graph_name, new_value, node_id))

        # Keep log bounded
        if len(self._mutation_log) > 1000:
            self._mutation_log = self._mutation_log[-500:]

        return new_value

    def check_sync(self, source: str, target: str) -> SyncState:
        """Check synchronization state between two graphs.

        Args:
            source: The graph that produces values.
            target: The graph that consumes values.

        Returns:
            SyncState indicating the relationship.
        """
        if source not in self._clocks or target not in self._clocks:
            return SyncState.UNKNOWN

        source_clock = self._clocks[source]
        target_clock = self._clocks[target]

        # Check if target has seen source's latest state
        pair_key = (source, target)
        last_sync = self._last_sync.get(pair_key)

        if last_sync is None:
            return SyncState.STALE

        # Target is in sync if it has seen source's current clock
        source_at_sync = last_sync.get(source)
        source_current = source_clock.get(source)

        if source_at_sync == source_current:
            return SyncState.IN_SYNC
        elif source_clock.concurrent_with(target_clock):
            return SyncState.DIVERGED
        else:
            return SyncState.STALE

    def synchronize(
        self,
        source: str,
        target: str,
        node_values: Dict[str, Any],
    ) -> SyncResult:
        """Perform synchronization from source to target.

        Pushes node values from source to target and updates
        vector clocks to reflect the sync.

        Args:
            source: Source graph name.
            target: Target graph name.
            node_values: Dict of {node_id: value} to sync.

        Returns:
            SyncResult with details of the operation.
        """
        if source not in self._clocks or target not in self._clocks:
            return SyncResult(state=SyncState.UNKNOWN)

        source_clock = self._clocks[source]
        target_clock = self._clocks[target]

        # Detect conflicts (concurrent modifications)
        conflicts: Set[str] = set()
        if source_clock.concurrent_with(target_clock):
            # Both have been modified since last sync — potential conflicts
            for node_id in node_values:
                # Check if target also modified this node
                target_mutations = [
                    entry for entry in self._mutation_log
                    if entry[0] == target and entry[2] == node_id
                ]
                if target_mutations:
                    conflicts.add(node_id)

        # Perform sync (non-conflicting nodes)
        synced: Set[str] = set()
        for node_id in node_values:
            if node_id not in conflicts:
                synced.add(node_id)

        # Update vector clocks
        target_clock.merge(source_clock)
        pair_key = (source, target)
        self._last_sync[pair_key] = VectorClock(
            clocks=dict(source_clock.clocks)
        )

        state = SyncState.IN_SYNC if not conflicts else SyncState.DIVERGED

        return SyncResult(
            state=state,
            synced_nodes=synced,
            conflicts=conflicts,
            vector_clock=target_clock,
        )

    def create_barrier(self, participants: Set[str]) -> SyncBarrier:
        """Create a synchronization barrier for multiple graphs.

        Args:
            participants: Set of graph names that must all arrive.

        Returns:
            A SyncBarrier that tracks arrival.
        """
        barrier = SyncBarrier(participants=frozenset(participants))
        self._barriers.append(barrier)
        return barrier

    def get_clock(self, graph_name: str) -> Optional[VectorClock]:
        """Get the current vector clock for a graph."""
        return self._clocks.get(graph_name)

    def get_mutation_count(self, graph_name: str) -> int:
        """Get total mutations recorded for a graph."""
        return sum(1 for entry in self._mutation_log if entry[0] == graph_name)
