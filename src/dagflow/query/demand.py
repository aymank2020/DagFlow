"""DemandEngine — demand-driven (lazy) recomputation.

Instead of eagerly propagating all changes, the DemandEngine only
recomputes a node when its value is explicitly requested. It walks
upstream to find the minimal set of nodes that need recomputation,
then executes them bottom-up.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode, NodeState)
- memo.cache (MemoCache)
- scheduler.topo (TopologicalScheduler)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Set

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode, NodeState
from dagflow.memo.cache import MemoCache
from dagflow.scheduler.topo import TopologicalScheduler


class DemandEngine:
    """Lazy evaluation engine — computes only what's requested.

    When a node's value is demanded:
    1. Walk upstream to find all dirty ancestors.
    2. Schedule them in topological order.
    3. Execute only those needed to produce the requested value.
    """

    def __init__(self, graph: ComputeGraph, cache: MemoCache) -> None:
        self._graph = graph
        self._cache = cache
        self._scheduler = TopologicalScheduler(graph)

    def demand(self, node_id: str) -> Any:
        """Get the current value of a node, recomputing if necessary.

        Args:
            node_id: The node whose value is requested.

        Returns:
            The current (possibly freshly computed) value.
        """
        node = self._graph.get_node(node_id)

        # Input nodes always have their value ready
        if isinstance(node, InputNode):
            return node.value

        # If clean and cached, return immediately
        if node.state == NodeState.CLEAN and self._cache.get(node_id) is not None:
            return self._cache.get(node_id)

        # Find all dirty ancestors that need recomputation
        dirty_ancestors = self._find_dirty_ancestors(node_id)
        dirty_ancestors.add(node_id)

        # Schedule them
        execution_order = self._scheduler.schedule(dirty_ancestors)

        # Execute in order
        for exec_id in execution_order:
            self._execute_node(exec_id)

        # Return the freshly computed value
        return self._cache.get(node_id)

    def demand_multiple(self, node_ids: List[str]) -> Dict[str, Any]:
        """Demand values for multiple nodes efficiently.

        Finds the union of all dirty ancestors, schedules once, executes once.
        """
        all_dirty: Set[str] = set()
        results: Dict[str, Any] = {}

        for nid in node_ids:
            node = self._graph.get_node(nid)
            if isinstance(node, InputNode):
                results[nid] = node.value
                continue
            if node.state == NodeState.CLEAN and self._cache.get(nid) is not None:
                results[nid] = self._cache.get(nid)
                continue
            ancestors = self._find_dirty_ancestors(nid)
            ancestors.add(nid)
            all_dirty.update(ancestors)

        if all_dirty:
            execution_order = self._scheduler.schedule(all_dirty)
            for exec_id in execution_order:
                self._execute_node(exec_id)

        # Collect results
        for nid in node_ids:
            if nid not in results:
                node = self._graph.get_node(nid)
                if isinstance(node, InputNode):
                    results[nid] = node.value
                else:
                    results[nid] = self._cache.get(nid)

        return results

    def _find_dirty_ancestors(self, node_id: str) -> Set[str]:
        """Walk upstream to find all dirty compute nodes that feed into node_id."""
        dirty: Set[str] = set()
        visited: Set[str] = set()
        queue: deque = deque([node_id])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            node = self._graph.get_node(current)
            if isinstance(node, InputNode):
                continue

            if node.state == NodeState.DIRTY:
                dirty.add(current)
                # Must also check this node's dependencies
                for dep_id in node.dependencies:
                    if dep_id not in visited:
                        queue.append(dep_id)
            elif node.state == NodeState.CLEAN:
                # Even clean nodes might have dirty ancestors
                # Check if cache is still valid
                dep_gens = self._get_current_dep_generations(node)
                if not self._cache.is_valid(current, dep_gens):
                    dirty.add(current)
                    node.mark_dirty()
                    for dep_id in node.dependencies:
                        if dep_id not in visited:
                            queue.append(dep_id)

        return dirty

    def _execute_node(self, node_id: str) -> None:
        """Execute a single compute node and update cache."""
        node = self._graph.get_node(node_id)
        if not isinstance(node, ComputeNode):
            return
        if node.state == NodeState.CLEAN:
            return

        # Gather dependency values
        dep_values: Dict[str, Any] = {}
        dep_generations: Dict[str, int] = {}

        for dep_id in node.dependencies:
            dep_node = self._graph.get_node(dep_id)
            if isinstance(dep_node, InputNode):
                dep_values[dep_id] = dep_node.value
                dep_generations[dep_id] = dep_node.generation
            elif isinstance(dep_node, ComputeNode):
                dep_values[dep_id] = dep_node.cached_value
                dep_generations[dep_id] = self._cache.get_generation(dep_id)

        # Compute
        new_value = node.func(dep_values)
        node.mark_clean(new_value, dep_generations)
        self._cache.store(node_id, new_value, dep_generations)

    def _get_current_dep_generations(self, node: ComputeNode) -> Dict[str, int]:
        """Get current generation map for a node's dependencies."""
        gens: Dict[str, int] = {}
        for dep_id in node.dependencies:
            dep_node = self._graph.get_node(dep_id)
            if isinstance(dep_node, InputNode):
                gens[dep_id] = dep_node.generation
            elif isinstance(dep_node, ComputeNode):
                gens[dep_id] = self._cache.get_generation(dep_id)
        return gens
