"""Impact analysis — determine what nodes are affected by changes.

Provides forward and backward impact analysis: given a set of changed
inputs, determine which compute nodes will be recomputed, estimate
the propagation cost, and identify the blast radius of changes.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode, InputNode)
- optimizer.cost_model (CostModel, NodeCost)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode


@dataclass
class ImpactReport:
    """Report of impact analysis for a set of changes.

    Attributes:
        changed_inputs: The input nodes that were changed.
        affected_nodes: All nodes that would be recomputed.
        affected_count: Number of affected nodes.
        blast_radius: Fraction of graph affected (0.0 to 1.0).
        propagation_depth: Maximum depth of propagation.
        critical_paths: Longest propagation chains.
        isolated_effects: Nodes affected that don't affect others.
    """

    changed_inputs: FrozenSet[str] = field(default_factory=frozenset)
    affected_nodes: Set[str] = field(default_factory=set)
    affected_count: int = 0
    blast_radius: float = 0.0
    propagation_depth: int = 0
    critical_paths: List[List[str]] = field(default_factory=list)
    isolated_effects: Set[str] = field(default_factory=set)

    @property
    def is_localized(self) -> bool:
        """Whether the impact is localized (< 20% of graph)."""
        return self.blast_radius < 0.2

    @property
    def is_widespread(self) -> bool:
        """Whether the impact is widespread (> 50% of graph)."""
        return self.blast_radius > 0.5


class ImpactAnalyzer:
    """Analyzes the impact of changes on the computation graph.

    Determines which nodes would be affected by changing specific
    inputs, computes the blast radius, and identifies propagation
    paths.

    Usage:
        analyzer = ImpactAnalyzer(graph)
        report = analyzer.analyze_impact(["input_x", "input_y"])
        print(f"Blast radius: {report.blast_radius:.1%}")
    """

    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def analyze_impact(self, changed_inputs: List[str]) -> ImpactReport:
        """Analyze the full impact of changing specified inputs.

        Args:
            changed_inputs: List of input node IDs being changed.

        Returns:
            ImpactReport with complete analysis.
        """
        affected = self._compute_affected_nodes(changed_inputs)
        total_compute = len(self._graph.get_compute_nodes())

        # Compute propagation depth
        depth = self._compute_propagation_depth(changed_inputs, affected)

        # Find critical paths
        critical_paths = self._find_critical_paths(changed_inputs, affected)

        # Find isolated effects (affected nodes with no affected dependents)
        isolated = self._find_isolated_effects(affected)

        blast_radius = len(affected) / total_compute if total_compute > 0 else 0.0

        return ImpactReport(
            changed_inputs=frozenset(changed_inputs),
            affected_nodes=affected,
            affected_count=len(affected),
            blast_radius=blast_radius,
            propagation_depth=depth,
            critical_paths=critical_paths,
            isolated_effects=isolated,
        )

    def what_if(self, node_id: str) -> Set[str]:
        """Quick query: what nodes are affected if node_id changes?

        Args:
            node_id: The node to hypothetically change.

        Returns:
            Set of all downstream affected node IDs.
        """
        return set(self._graph.get_all_downstream(node_id))

    def reverse_impact(self, target_node: str) -> Set[str]:
        """Reverse query: what inputs could cause target_node to recompute?

        Args:
            target_node: The node we want to understand triggers for.

        Returns:
            Set of input node IDs that can trigger recomputation.
        """
        all_upstream = set(self._graph.get_all_upstream(target_node))
        input_nodes = set(self._graph.get_input_nodes())
        return all_upstream & input_nodes

    def compare_impacts(
        self, inputs_a: List[str], inputs_b: List[str]
    ) -> Dict[str, Set[str]]:
        """Compare the impact of two different change sets.

        Args:
            inputs_a: First set of changed inputs.
            inputs_b: Second set of changed inputs.

        Returns:
            Dict with keys "only_a", "only_b", "both" containing node sets.
        """
        affected_a = self._compute_affected_nodes(inputs_a)
        affected_b = self._compute_affected_nodes(inputs_b)

        return {
            "only_a": affected_a - affected_b,
            "only_b": affected_b - affected_a,
            "both": affected_a & affected_b,
        }

    def sensitivity_ranking(self) -> List[Tuple[str, int]]:
        """Rank input nodes by their impact (most impactful first).

        Returns:
            List of (input_node_id, affected_count) sorted descending.
        """
        rankings: List[Tuple[str, int]] = []
        for input_id in self._graph.get_input_nodes():
            affected = self._compute_affected_nodes([input_id])
            rankings.append((input_id, len(affected)))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def find_shared_dependencies(self, node_ids: List[str]) -> Set[str]:
        """Find inputs that affect ALL specified nodes.

        Args:
            node_ids: Nodes to find common dependencies for.

        Returns:
            Set of input IDs that affect every specified node.
        """
        if not node_ids:
            return set()

        # Get upstream inputs for each node
        input_sets: List[Set[str]] = []
        all_inputs = set(self._graph.get_input_nodes())

        for node_id in node_ids:
            upstream = set(self._graph.get_all_upstream(node_id))
            input_deps = upstream & all_inputs
            input_sets.append(input_deps)

        # Intersection of all input dependency sets
        if not input_sets:
            return set()
        return set.intersection(*input_sets)

    def propagation_layers(
        self, changed_inputs: List[str]
    ) -> List[Set[str]]:
        """Compute propagation in layers (BFS levels from inputs).

        Each layer contains nodes that would be recomputed at that
        distance from the changed inputs.

        Args:
            changed_inputs: Input nodes that changed.

        Returns:
            List of sets, where index i = nodes at distance i from inputs.
        """
        layers: List[Set[str]] = []
        visited: Set[str] = set()
        current_layer: Set[str] = set()

        # Seed with immediate dependents of changed inputs
        for input_id in changed_inputs:
            for dep_id in self._graph.get_dependents(input_id):
                current_layer.add(dep_id)

        while current_layer:
            layers.append(current_layer)
            visited.update(current_layer)

            next_layer: Set[str] = set()
            for node_id in current_layer:
                for dep_id in self._graph.get_dependents(node_id):
                    if dep_id not in visited:
                        next_layer.add(dep_id)

            current_layer = next_layer

        return layers

    def _compute_affected_nodes(self, changed_inputs: List[str]) -> Set[str]:
        """BFS to find all nodes affected by input changes."""
        affected: Set[str] = set()
        queue: deque = deque()

        for input_id in changed_inputs:
            for dep_id in self._graph.get_dependents(input_id):
                if dep_id not in affected:
                    queue.append(dep_id)
                    affected.add(dep_id)

        while queue:
            current = queue.popleft()
            for dep_id in self._graph.get_dependents(current):
                if dep_id not in affected:
                    affected.add(dep_id)
                    queue.append(dep_id)

        return affected

    def _compute_propagation_depth(
        self, changed_inputs: List[str], affected: Set[str]
    ) -> int:
        """Compute the maximum propagation depth."""
        if not affected:
            return 0

        max_depth = 0
        depths: Dict[str, int] = {}

        queue: deque = deque()
        for input_id in changed_inputs:
            for dep_id in self._graph.get_dependents(input_id):
                if dep_id in affected:
                    depths[dep_id] = 1
                    queue.append(dep_id)

        while queue:
            current = queue.popleft()
            current_depth = depths[current]
            max_depth = max(max_depth, current_depth)

            for dep_id in self._graph.get_dependents(current):
                if dep_id in affected:
                    new_depth = current_depth + 1
                    if dep_id not in depths or new_depth > depths[dep_id]:
                        depths[dep_id] = new_depth
                        queue.append(dep_id)

        return max_depth

    def _find_critical_paths(
        self, changed_inputs: List[str], affected: Set[str]
    ) -> List[List[str]]:
        """Find the longest propagation paths."""
        if not affected:
            return []

        # Find leaf affected nodes (no affected dependents)
        leaves = [
            n for n in affected
            if not any(d in affected for d in self._graph.get_dependents(n))
        ]

        # Trace back from leaves to inputs
        paths: List[List[str]] = []
        for leaf in leaves[:5]:  # Limit to top 5 paths
            path = self._trace_path_to_input(leaf, changed_inputs)
            if path:
                paths.append(path)

        # Sort by length descending
        paths.sort(key=len, reverse=True)
        return paths[:3]  # Return top 3 longest

    def _trace_path_to_input(
        self, node_id: str, inputs: List[str]
    ) -> List[str]:
        """Trace a path from node back to one of the changed inputs."""
        path = [node_id]
        current = node_id
        input_set = set(inputs)
        visited: Set[str] = {node_id}

        while current not in input_set:
            deps = self._graph.get_dependencies(current)
            # Pick the dependency closest to an input
            next_node = None
            for dep in deps:
                if dep in input_set:
                    path.append(dep)
                    return list(reversed(path))
                if dep not in visited:
                    next_node = dep
                    break

            if next_node is None:
                break

            visited.add(next_node)
            path.append(next_node)
            current = next_node

        return list(reversed(path))

    def _find_isolated_effects(self, affected: Set[str]) -> Set[str]:
        """Find affected nodes whose dependents are NOT affected."""
        isolated: Set[str] = set()
        for node_id in affected:
            dependents = self._graph.get_dependents(node_id)
            if not any(d in affected for d in dependents):
                isolated.add(node_id)
        return isolated
