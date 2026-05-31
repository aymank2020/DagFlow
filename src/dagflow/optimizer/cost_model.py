"""Cost estimation model for DAG nodes.

Provides cost estimation for compute nodes based on their characteristics:
memory usage, computation time, I/O operations, and dependency fan-in/fan-out.
Used by the optimizer to make informed decisions about fusion, partitioning,
and scheduling priorities.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode, InputNode)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode


@dataclass
class NodeCost:
    """Estimated cost metrics for a single node.

    Attributes:
        node_id: The node this cost applies to.
        compute_time: Estimated computation time in seconds.
        memory_bytes: Estimated memory usage in bytes.
        io_operations: Estimated number of I/O operations.
        fan_in: Number of direct dependencies.
        fan_out: Number of direct dependents.
        total_cost: Weighted aggregate cost score.
    """

    node_id: str
    compute_time: float = 0.0
    memory_bytes: int = 0
    io_operations: int = 0
    fan_in: int = 0
    fan_out: int = 0
    total_cost: float = 0.0

    @property
    def is_expensive(self) -> bool:
        """Whether this node exceeds typical cost thresholds."""
        return self.total_cost > 1.0

    @property
    def is_io_bound(self) -> bool:
        """Whether this node is primarily I/O bound."""
        if self.total_cost == 0:
            return False
        io_fraction = self.io_operations * 0.01 / max(self.total_cost, 0.001)
        return io_fraction > 0.5


@dataclass
class CostProfile:
    """Aggregate cost profile for the entire graph.

    Attributes:
        node_costs: Individual cost for each node.
        total_compute_time: Sum of all node compute times.
        total_memory: Peak estimated memory usage.
        critical_path_cost: Cost along the critical (longest) path.
        bottleneck_nodes: Nodes with disproportionately high cost.
    """

    node_costs: Dict[str, NodeCost] = field(default_factory=dict)
    total_compute_time: float = 0.0
    total_memory: int = 0
    critical_path_cost: float = 0.0
    bottleneck_nodes: List[str] = field(default_factory=list)


class CostModel:
    """Estimates computational cost for DAG nodes.

    Supports multiple estimation strategies:
    - Static analysis: based on node structure (fan-in, fan-out)
    - Profiled: based on actual measured execution times
    - Heuristic: configurable weights for different cost factors

    Usage:
        model = CostModel(graph)
        cost = model.estimate_node("compute_1")
        profile = model.profile_graph()
    """

    # Default weights for cost aggregation
    DEFAULT_WEIGHTS = {
        "compute_time": 1.0,
        "memory": 0.001,  # per KB
        "io": 0.1,  # per operation
        "fan_in": 0.05,
        "fan_out": 0.02,
    }

    def __init__(
        self,
        graph: ComputeGraph,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the cost model.

        Args:
            graph: The computation graph to analyze.
            weights: Custom weights for cost factors. Uses defaults if None.
        """
        self._graph = graph
        self._weights = weights or dict(self.DEFAULT_WEIGHTS)
        self._measured_times: Dict[str, List[float]] = {}
        self._memory_estimates: Dict[str, int] = {}
        self._io_annotations: Dict[str, int] = {}

    def estimate_node(self, node_id: str) -> NodeCost:
        """Estimate the cost of a single node.

        Combines structural analysis with any available profiling data.

        Args:
            node_id: The node to estimate.

        Returns:
            NodeCost with all estimated metrics.
        """
        node = self._graph.get_node(node_id)
        fan_in = len(self._graph.get_dependencies(node_id))
        fan_out = len(self._graph.get_dependents(node_id))

        # Compute time: use measured if available, else heuristic
        compute_time = self._estimate_compute_time(node_id, fan_in)

        # Memory: use annotation if available, else estimate from fan-in
        memory = self._estimate_memory(node_id, fan_in)

        # I/O: use annotation if available
        io_ops = self._io_annotations.get(node_id, 0)

        # Aggregate cost
        total = (
            self._weights["compute_time"] * compute_time
            + self._weights["memory"] * (memory / 1024)
            + self._weights["io"] * io_ops
            + self._weights["fan_in"] * fan_in
            + self._weights["fan_out"] * fan_out
        )

        return NodeCost(
            node_id=node_id,
            compute_time=compute_time,
            memory_bytes=memory,
            io_operations=io_ops,
            fan_in=fan_in,
            fan_out=fan_out,
            total_cost=total,
        )

    def profile_graph(self) -> CostProfile:
        """Compute cost profile for the entire graph.

        Returns:
            CostProfile with per-node costs and aggregate metrics.
        """
        profile = CostProfile()

        for node_id in self._graph.get_compute_nodes():
            cost = self.estimate_node(node_id)
            profile.node_costs[node_id] = cost
            profile.total_compute_time += cost.compute_time
            profile.total_memory += cost.memory_bytes

        # Find bottlenecks (nodes with cost > 2x average)
        if profile.node_costs:
            avg_cost = sum(c.total_cost for c in profile.node_costs.values()) / len(
                profile.node_costs
            )
            threshold = avg_cost * 2.0
            profile.bottleneck_nodes = [
                nid
                for nid, cost in profile.node_costs.items()
                if cost.total_cost > threshold
            ]

        # Critical path cost
        profile.critical_path_cost = self._compute_critical_path_cost(
            profile.node_costs
        )

        return profile

    def record_execution(self, node_id: str, elapsed: float) -> None:
        """Record an actual execution time for a node.

        Used to improve future estimates with real profiling data.

        Args:
            node_id: The node that was executed.
            elapsed: Actual execution time in seconds.
        """
        if node_id not in self._measured_times:
            self._measured_times[node_id] = []
        self._measured_times[node_id].append(elapsed)

        # Keep only last 10 measurements for rolling average
        if len(self._measured_times[node_id]) > 10:
            self._measured_times[node_id] = self._measured_times[node_id][-10:]

    def annotate_memory(self, node_id: str, bytes_estimate: int) -> None:
        """Annotate a node with an estimated memory usage.

        Args:
            node_id: The node to annotate.
            bytes_estimate: Estimated memory in bytes.
        """
        self._memory_estimates[node_id] = bytes_estimate

    def annotate_io(self, node_id: str, operations: int) -> None:
        """Annotate a node with estimated I/O operations.

        Args:
            node_id: The node to annotate.
            operations: Number of I/O operations.
        """
        self._io_annotations[node_id] = operations

    def compare_nodes(self, node_a: str, node_b: str) -> int:
        """Compare cost of two nodes.

        Returns:
            -1 if a < b, 0 if equal, 1 if a > b.
        """
        cost_a = self.estimate_node(node_a).total_cost
        cost_b = self.estimate_node(node_b).total_cost
        if cost_a < cost_b:
            return -1
        elif cost_a > cost_b:
            return 1
        return 0

    def rank_nodes_by_cost(self, descending: bool = True) -> List[Tuple[str, float]]:
        """Rank all compute nodes by their estimated cost.

        Args:
            descending: If True, most expensive first.

        Returns:
            List of (node_id, total_cost) tuples, sorted.
        """
        costs: List[Tuple[str, float]] = []
        for node_id in self._graph.get_compute_nodes():
            cost = self.estimate_node(node_id)
            costs.append((node_id, cost.total_cost))

        costs.sort(key=lambda x: x[1], reverse=descending)
        return costs

    def _estimate_compute_time(self, node_id: str, fan_in: int) -> float:
        """Estimate compute time using measurements or heuristics."""
        # Use measured data if available
        measurements = self._measured_times.get(node_id)
        if measurements:
            return sum(measurements) / len(measurements)

        # Heuristic: base cost + cost per dependency
        base_cost = 0.001  # 1ms base
        per_dep_cost = 0.0005  # 0.5ms per dependency
        return base_cost + per_dep_cost * fan_in

    def _estimate_memory(self, node_id: str, fan_in: int) -> int:
        """Estimate memory usage from annotations or heuristics."""
        annotated = self._memory_estimates.get(node_id)
        if annotated is not None:
            return annotated

        # Heuristic: 1KB base + 512 bytes per dependency
        return 1024 + 512 * fan_in

    def _compute_critical_path_cost(
        self, node_costs: Dict[str, NodeCost]
    ) -> float:
        """Compute the cost along the critical path."""
        if not node_costs:
            return 0.0

        # Dynamic programming: longest cost path
        path_cost: Dict[str, float] = {}

        # Process in topological order
        from dagflow.scheduler.topo import TopologicalScheduler

        scheduler = TopologicalScheduler(self._graph)
        order = scheduler.full_schedule()

        for node_id in order:
            cost = node_costs.get(node_id)
            if cost is None:
                continue

            deps = self._graph.get_dependencies(node_id)
            max_dep_cost = max(
                (path_cost.get(d, 0.0) for d in deps), default=0.0
            )
            path_cost[node_id] = max_dep_cost + cost.total_cost

        return max(path_cost.values(), default=0.0)
