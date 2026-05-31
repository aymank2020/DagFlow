"""Dead node elimination and unreachable subgraph removal.

Identifies nodes that cannot contribute to any output (dead nodes)
and removes them from the graph. Also detects disconnected subgraphs
that have no path to any designated output node.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode, InputNode)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode


@dataclass
class PruneResult:
    """Result of a dead node pruning pass.

    Attributes:
        removed_nodes: Set of node IDs that were removed.
        removed_count: Total nodes eliminated.
        orphaned_inputs: Input nodes with no dependents.
        disconnected_subgraphs: Number of disconnected components found.
        preserved_nodes: Nodes kept because they're reachable from outputs.
    """

    removed_nodes: Set[str] = field(default_factory=set)
    removed_count: int = 0
    orphaned_inputs: Set[str] = field(default_factory=set)
    disconnected_subgraphs: int = 0
    preserved_nodes: Set[str] = field(default_factory=set)


class DeadNodePruner:
    """Identifies and removes dead nodes from the computation graph.

    A node is considered "dead" if:
    1. It has no dependents (leaf compute node with no observers), OR
    2. It cannot reach any designated output node, OR
    3. It belongs to a disconnected subgraph with no outputs.

    The pruner works backwards from output nodes: any node NOT reachable
    by walking upstream from an output is dead.

    Usage:
        pruner = DeadNodePruner(graph, outputs={"result", "total"})
        dead = pruner.find_dead_nodes()
        result = pruner.prune()
    """

    def __init__(
        self,
        graph: ComputeGraph,
        outputs: Optional[Set[str]] = None,
        preserve: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the pruner.

        Args:
            graph: The computation graph to analyze.
            outputs: Set of node IDs considered as outputs (must be preserved).
                     If None, nodes with no dependents are treated as outputs.
            preserve: Additional node IDs to never remove regardless of reachability.
        """
        self._graph = graph
        self._outputs = outputs or self._infer_outputs()
        self._preserve = preserve or set()

    def find_dead_nodes(self) -> Set[str]:
        """Identify all dead nodes in the graph.

        Performs backward reachability from output nodes. Any compute node
        not reachable is considered dead.

        Returns:
            Set of node IDs that are dead (can be safely removed).
        """
        reachable = self._compute_reachable_from_outputs()
        all_nodes = set(self._graph.all_node_ids)
        dead = all_nodes - reachable - self._preserve - self._outputs
        return dead

    def find_orphaned_inputs(self) -> Set[str]:
        """Find input nodes that have no dependents.

        These inputs feed nothing and can be removed unless preserved.

        Returns:
            Set of orphaned input node IDs.
        """
        orphaned: Set[str] = set()
        for node_id in self._graph.get_input_nodes():
            if node_id in self._preserve or node_id in self._outputs:
                continue
            dependents = self._graph.get_dependents(node_id)
            if not dependents:
                orphaned.add(node_id)
        return orphaned

    def find_disconnected_subgraphs(self) -> List[FrozenSet[str]]:
        """Identify disconnected components in the graph.

        Returns:
            List of frozensets, each containing node IDs of a connected component.
        """
        all_nodes = set(self._graph.all_node_ids)
        visited: Set[str] = set()
        components: List[FrozenSet[str]] = []

        for node_id in all_nodes:
            if node_id in visited:
                continue

            # BFS to find all nodes in this component
            component: Set[str] = set()
            queue: deque = deque([node_id])

            while queue:
                current = queue.popleft()
                if current in component:
                    continue
                component.add(current)

                # Walk both directions (undirected connectivity)
                for neighbor in self._graph.get_dependents(current):
                    if neighbor not in component:
                        queue.append(neighbor)
                for neighbor in self._graph.get_dependencies(current):
                    if neighbor not in component:
                        queue.append(neighbor)

            visited.update(component)
            components.append(frozenset(component))

        return components

    def prune(self, dry_run: bool = False) -> PruneResult:
        """Execute the pruning pass.

        Removes dead nodes and optionally orphaned inputs from the graph.

        Args:
            dry_run: If True, identify dead nodes without removing them.

        Returns:
            PruneResult with details about what was (or would be) removed.
        """
        dead_nodes = self.find_dead_nodes()
        orphaned = self.find_orphaned_inputs()
        components = self.find_disconnected_subgraphs()

        # Determine which components have no outputs
        dead_components = 0
        for component in components:
            has_output = bool(component & self._outputs)
            has_preserved = bool(component & self._preserve)
            if not has_output and not has_preserved:
                dead_components += 1
                dead_nodes.update(component)

        reachable = self._compute_reachable_from_outputs()

        result = PruneResult(
            removed_nodes=dead_nodes,
            removed_count=len(dead_nodes),
            orphaned_inputs=orphaned,
            disconnected_subgraphs=dead_components,
            preserved_nodes=reachable | self._preserve,
        )

        if not dry_run:
            self._remove_nodes(dead_nodes)

        return result

    def compute_node_liveness(self) -> Dict[str, bool]:
        """Compute liveness for every node in the graph.

        A node is "live" if it's reachable from any output node
        via backward traversal.

        Returns:
            Dict mapping node_id -> is_live.
        """
        reachable = self._compute_reachable_from_outputs()
        return {
            nid: (nid in reachable or nid in self._preserve)
            for nid in self._graph.all_node_ids
        }

    def suggest_removals(self) -> Dict[str, str]:
        """Provide human-readable removal suggestions.

        Returns:
            Dict of {node_id: reason} for each removable node.
        """
        suggestions: Dict[str, str] = {}
        dead = self.find_dead_nodes()
        orphaned = self.find_orphaned_inputs()

        for nid in dead:
            if nid in orphaned:
                suggestions[nid] = "Orphaned input: no dependents"
            else:
                suggestions[nid] = "Unreachable: no path to any output node"

        return suggestions

    def _compute_reachable_from_outputs(self) -> Set[str]:
        """BFS backward from outputs to find all reachable nodes."""
        reachable: Set[str] = set()
        queue: deque = deque(self._outputs)

        while queue:
            current = queue.popleft()
            if current in reachable:
                continue
            reachable.add(current)

            # Walk upstream
            for dep_id in self._graph.get_dependencies(current):
                if dep_id not in reachable:
                    queue.append(dep_id)

        return reachable

    def _infer_outputs(self) -> Set[str]:
        """Infer output nodes as compute nodes with no dependents."""
        outputs: Set[str] = set()
        for node_id in self._graph.get_compute_nodes():
            if not self._graph.get_dependents(node_id):
                outputs.add(node_id)
        return outputs

    def _remove_nodes(self, node_ids: Set[str]) -> None:
        """Remove nodes from the graph's internal structures.

        Note: This modifies the graph in place. The graph's internal
        adjacency structures are updated to maintain consistency.
        """
        for node_id in node_ids:
            if node_id not in self._graph._nodes:
                continue

            # Remove from adjacency lists
            for dep_id in list(self._graph._reverse.get(node_id, set())):
                self._graph._adjacency.get(dep_id, set()).discard(node_id)
                dep_node = self._graph._nodes.get(dep_id)
                if dep_node and node_id in dep_node.dependents:
                    dep_node.dependents.remove(node_id)

            for dependent_id in list(self._graph._adjacency.get(node_id, set())):
                self._graph._reverse.get(dependent_id, set()).discard(node_id)
                dep_node = self._graph._nodes.get(dependent_id)
                if isinstance(dep_node, ComputeNode) and node_id in dep_node.dependencies:
                    dep_node.dependencies.remove(node_id)

            # Remove the node itself
            del self._graph._nodes[node_id]
            self._graph._adjacency.pop(node_id, None)
            self._graph._reverse.pop(node_id, None)
