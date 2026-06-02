"""Graph complexity metrics — depth, width, fan-in/fan-out analysis.

Computes structural complexity metrics for a computation graph,
useful for understanding performance characteristics, identifying
potential bottlenecks, and comparing graph designs.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode, InputNode)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode


@dataclass
class ComplexityMetrics:
    """Comprehensive complexity metrics for a graph.

    Attributes:
        depth: Longest path from any input to any output.
        width: Maximum number of nodes at any single level.
        total_nodes: Total node count.
        input_count: Number of input nodes.
        compute_count: Number of compute nodes.
        edge_count: Total number of edges.
        avg_fan_in: Average number of dependencies per compute node.
        avg_fan_out: Average number of dependents per node.
        max_fan_in: Maximum dependencies on any single node.
        max_fan_out: Maximum dependents of any single node.
        density: Edge density (edges / max possible edges).
        dag_entropy: Structural entropy measure (higher = more complex).
    """

    depth: int = 0
    width: int = 0
    total_nodes: int = 0
    input_count: int = 0
    compute_count: int = 0
    edge_count: int = 0
    avg_fan_in: float = 0.0
    avg_fan_out: float = 0.0
    max_fan_in: int = 0
    max_fan_out: int = 0
    density: float = 0.0
    dag_entropy: float = 0.0

    @property
    def is_linear(self) -> bool:
        """Whether the graph is essentially a linear chain."""
        return self.width <= 1 and self.max_fan_in <= 1 and self.max_fan_out <= 1

    @property
    def is_wide(self) -> bool:
        """Whether the graph is wider than it is deep."""
        return self.width > self.depth

    @property
    def parallelism_ratio(self) -> float:
        """Ratio of width to depth (higher = more parallelizable)."""
        return self.width / self.depth if self.depth > 0 else 0.0


class ComplexityAnalyzer:
    """Analyzes structural complexity of a computation graph.

    Computes various metrics that characterize the graph's shape,
    connectivity, and potential for parallel execution.

    Usage:
        analyzer = ComplexityAnalyzer(graph)
        metrics = analyzer.analyze()
        print(f"Depth: {metrics.depth}, Width: {metrics.width}")
    """

    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def analyze(self) -> ComplexityMetrics:
        """Compute all complexity metrics for the graph.

        Returns:
            ComplexityMetrics with all computed values.
        """
        metrics = ComplexityMetrics()

        all_nodes = self._graph.all_node_ids
        metrics.total_nodes = len(all_nodes)
        metrics.input_count = len(self._graph.get_input_nodes())
        metrics.compute_count = len(self._graph.get_compute_nodes())

        # Edge count and fan metrics
        fan_ins: List[int] = []
        fan_outs: List[int] = []
        total_edges = 0

        for node_id in all_nodes:
            deps = self._graph.get_dependencies(node_id)
            dependents = self._graph.get_dependents(node_id)
            fan_in = len(deps)
            fan_out = len(dependents)
            fan_ins.append(fan_in)
            fan_outs.append(fan_out)
            total_edges += fan_out

        metrics.edge_count = total_edges
        metrics.max_fan_in = max(fan_ins, default=0)
        metrics.max_fan_out = max(fan_outs, default=0)
        metrics.avg_fan_in = (
            sum(fan_ins) / len(fan_ins) if fan_ins else 0.0
        )
        metrics.avg_fan_out = (
            sum(fan_outs) / len(fan_outs) if fan_outs else 0.0
        )

        # Density
        n = metrics.total_nodes
        max_edges = n * (n - 1) / 2 if n > 1 else 1
        metrics.density = total_edges / max_edges if max_edges > 0 else 0.0

        # Depth and width via level assignment
        levels = self._compute_levels()
        if levels:
            metrics.depth = max(levels.values()) + 1
            level_counts = defaultdict(int)
            for lvl in levels.values():
                level_counts[lvl] += 1
            metrics.width = max(level_counts.values(), default=0)

        # Structural entropy
        metrics.dag_entropy = self._compute_entropy(fan_outs)

        return metrics

    def compute_depth(self) -> int:
        """Compute the depth (longest path) of the graph."""
        levels = self._compute_levels()
        return (max(levels.values()) + 1) if levels else 0

    def compute_width(self) -> int:
        """Compute the width (max nodes at any level) of the graph."""
        levels = self._compute_levels()
        if not levels:
            return 0
        level_counts = defaultdict(int)
        for lvl in levels.values():
            level_counts[lvl] += 1
        return max(level_counts.values(), default=0)

    def fan_in_distribution(self) -> Dict[int, int]:
        """Compute the distribution of fan-in values.

        Returns:
            Dict mapping fan-in value -> count of nodes with that fan-in.
        """
        distribution: Dict[int, int] = defaultdict(int)
        for node_id in self._graph.all_node_ids:
            fan_in = len(self._graph.get_dependencies(node_id))
            distribution[fan_in] += 1
        return dict(distribution)

    def fan_out_distribution(self) -> Dict[int, int]:
        """Compute the distribution of fan-out values.

        Returns:
            Dict mapping fan-out value -> count of nodes with that fan-out.
        """
        distribution: Dict[int, int] = defaultdict(int)
        for node_id in self._graph.all_node_ids:
            fan_out = len(self._graph.get_dependents(node_id))
            distribution[fan_out] += 1
        return dict(distribution)

    def level_profile(self) -> List[int]:
        """Compute node count at each level.

        Returns:
            List where index i = number of nodes at level i.
        """
        levels = self._compute_levels()
        if not levels:
            return []
        max_level = max(levels.values())
        profile = [0] * (max_level + 1)
        for lvl in levels.values():
            profile[lvl] += 1
        return profile

    def identify_critical_nodes(self, threshold: float = 0.5) -> List[str]:
        """Identify nodes that are critical to graph connectivity.

        A node is critical if removing it would disconnect a significant
        portion of the graph (more than threshold fraction of nodes).

        Args:
            threshold: Fraction of nodes that must be disconnected.

        Returns:
            List of critical node IDs.
        """
        total = self._graph.node_count
        if total <= 2:
            return []

        critical: List[str] = []
        for node_id in self._graph.get_compute_nodes():
            downstream = self._graph.get_all_downstream(node_id)
            upstream = self._graph.get_all_upstream(node_id)
            # Node is critical if it's on many paths
            reach = len(downstream) + len(upstream)
            if reach / total >= threshold:
                critical.append(node_id)

        return critical

    def _compute_levels(self) -> Dict[str, int]:
        """Assign topological levels to all nodes."""
        levels: Dict[str, int] = {}

        # Input nodes at level 0
        for nid in self._graph.get_input_nodes():
            levels[nid] = 0

        # BFS level assignment
        queue: deque = deque(self._graph.get_input_nodes())
        while queue:
            current = queue.popleft()
            current_level = levels[current]

            for dep_id in self._graph.get_dependents(current):
                if dep_id in levels:
                    # Take the maximum level from all dependencies
                    levels[dep_id] = max(levels[dep_id], current_level + 1)
                else:
                    # Check if all dependencies are resolved
                    deps = self._graph.get_dependencies(dep_id)
                    if all(d in levels for d in deps):
                        dep_levels = [levels[d] for d in deps]
                        levels[dep_id] = max(dep_levels, default=-1) + 1
                        queue.append(dep_id)
                    else:
                        # Re-queue for later processing
                        queue.append(dep_id)

        return levels

    def _compute_entropy(self, fan_outs: List[int]) -> float:
        """Compute structural entropy from fan-out distribution.

        Higher entropy indicates more complex/irregular structure.
        """
        import math

        if not fan_outs:
            return 0.0

        total = sum(fan_outs) or 1
        entropy = 0.0
        for fo in fan_outs:
            if fo > 0:
                p = fo / total
                entropy -= p * math.log2(p)
        return entropy
