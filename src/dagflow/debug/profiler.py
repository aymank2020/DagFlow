"""Profiler — timing profiler for node computations.

Measures how long each node takes to compute, tracks cumulative time,
and identifies hot spots in the computation graph. Wraps the computation
function of each node to add timing instrumentation.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode, InputNode)
- memo.cache (MemoCache)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode
from dagflow.memo.cache import MemoCache


@dataclass
class TimingRecord:
    """Timing data for a single node computation.

    Attributes:
        node_id: The node that was timed.
        duration_ms: Computation time in milliseconds.
        timestamp: When the computation started (monotonic).
        invocation: Which invocation number this was for this node.
    """

    node_id: str
    duration_ms: float
    timestamp: float
    invocation: int


class Profiler:
    """Profiles node computation times.

    Usage:
        profiler = Profiler(graph, cache)
        profiler.enable()

        # Run propagation through the profiler
        result = profiler.profile_propagation(["input_x"])

        profiler.disable()
        print(profiler.summary())
    """

    def __init__(self, graph: ComputeGraph, cache: MemoCache) -> None:
        self._graph = graph
        self._cache = cache
        self._records: List[TimingRecord] = []
        self._invocation_counts: Dict[str, int] = {}
        self._enabled: bool = False

    @property
    def is_enabled(self) -> bool:
        """Whether profiling is currently active."""
        return self._enabled

    @property
    def records(self) -> List[TimingRecord]:
        """All timing records collected."""
        return list(self._records)

    def enable(self) -> None:
        """Enable profiling."""
        self._enabled = True

    def disable(self) -> None:
        """Disable profiling."""
        self._enabled = False

    def clear(self) -> None:
        """Clear all collected timing data."""
        self._records.clear()
        self._invocation_counts.clear()

    def profile_propagation(self, changed_inputs: List[str]) -> Dict[str, Any]:
        """Run propagation with timing instrumentation.

        Each node computation is timed individually. Results are stored
        in self.records for later analysis.

        Args:
            changed_inputs: Input node IDs that changed.

        Returns:
            Dict of {node_id: new_value} for recomputed nodes.
        """
        from dagflow.propagation.invalidator import Invalidator
        from dagflow.scheduler.topo import TopologicalScheduler

        invalidator = Invalidator(self._graph)
        dirty_set = invalidator.invalidate_from(changed_inputs)

        scheduler = TopologicalScheduler(self._graph)
        execution_order = scheduler.schedule(dirty_set)

        recomputed: Dict[str, Any] = {}

        for node_id in execution_order:
            node = self._graph.get_node(node_id)
            if not isinstance(node, ComputeNode):
                continue

            # Gather dependency values
            dep_values: Dict[str, Any] = {}
            dep_gens: Dict[str, int] = {}
            for dep_id in node.dependencies:
                dep_node = self._graph.get_node(dep_id)
                if isinstance(dep_node, InputNode):
                    dep_values[dep_id] = dep_node.value
                    dep_gens[dep_id] = dep_node.generation
                elif isinstance(dep_node, ComputeNode):
                    dep_values[dep_id] = dep_node.cached_value
                    dep_gens[dep_id] = self._cache.get_generation(dep_id)

            # Timed execution
            start = time.perf_counter()
            new_value = node.func(dep_values)
            end = time.perf_counter()

            duration_ms = (end - start) * 1000.0

            # Record timing
            self._invocation_counts[node_id] = (
                self._invocation_counts.get(node_id, 0) + 1
            )
            self._records.append(
                TimingRecord(
                    node_id=node_id,
                    duration_ms=duration_ms,
                    timestamp=start,
                    invocation=self._invocation_counts[node_id],
                )
            )

            # Update node state
            node.mark_clean(new_value, dep_gens)
            self._cache.store(node_id, new_value, dep_gens)
            recomputed[node_id] = new_value

        return recomputed

    def total_time_ms(self) -> float:
        """Total computation time across all recorded nodes."""
        return sum(r.duration_ms for r in self._records)

    def time_per_node(self) -> Dict[str, float]:
        """Cumulative computation time per node (across all invocations)."""
        totals: Dict[str, float] = {}
        for record in self._records:
            totals[record.node_id] = (
                totals.get(record.node_id, 0.0) + record.duration_ms
            )
        return totals

    def hotspots(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Return the top N nodes by cumulative computation time.

        Args:
            top_n: Number of top nodes to return.

        Returns:
            List of (node_id, total_ms) sorted by time descending.
        """
        per_node = self.time_per_node()
        sorted_nodes = sorted(per_node.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]

    def invocation_count(self, node_id: str) -> int:
        """How many times a node has been computed."""
        return self._invocation_counts.get(node_id, 0)

    def summary(self) -> str:
        """Human-readable profiling summary.

        Returns:
            Multi-line string with timing breakdown.
        """
        if not self._records:
            return "No profiling data collected."

        lines = ["=== DagFlow Profiler Summary ==="]
        lines.append(f"Total computations: {len(self._records)}")
        lines.append(f"Total time: {self.total_time_ms():.3f} ms")
        lines.append("")
        lines.append("Top nodes by time:")

        for node_id, total_ms in self.hotspots():
            count = self._invocation_counts.get(node_id, 0)
            avg = total_ms / count if count > 0 else 0
            lines.append(
                f"  {node_id}: {total_ms:.3f} ms "
                f"({count} invocations, avg {avg:.3f} ms)"
            )

        return "\n".join(lines)
