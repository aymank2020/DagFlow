"""Eager propagation — recomputes dirty nodes immediately after invalidation.

Combines the Invalidator (to find dirty nodes) with the TopologicalScheduler
(to order them) and executes each node's function in sequence.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode, NodeState)
- scheduler.topo (TopologicalScheduler)
- propagation.invalidator (Invalidator)
- memo.cache (MemoCache) for early-cutoff
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode, NodeState
from dagflow.scheduler.topo import TopologicalScheduler
from dagflow.propagation.invalidator import Invalidator
from dagflow.memo.cache import MemoCache


class EagerPropagator:
    """Eagerly recomputes all dirty nodes after each input change.

    Implements early-cutoff: if a node recomputes to the same value,
    its dependents are removed from the execution plan (not recomputed).
    """

    def __init__(self, graph: ComputeGraph, cache: MemoCache) -> None:
        self._graph = graph
        self._cache = cache
        self._invalidator = Invalidator(graph)
        self._scheduler = TopologicalScheduler(graph)

    def propagate(self, changed_inputs: List[str]) -> Dict[str, Any]:
        """Propagate changes from modified inputs through the graph.

        Steps:
        1. Invalidate downstream nodes.
        2. Schedule dirty nodes in topological order.
        3. Execute each, applying early-cutoff.

        Args:
            changed_inputs: List of input node IDs that were modified.

        Returns:
            Dict of {node_id: new_value} for all nodes that were recomputed.
        """
        # Step 1: Invalidate
        dirty_set = self._invalidator.invalidate_from(changed_inputs)

        # Step 2: Schedule
        execution_order = self._scheduler.schedule(dirty_set)

        # Step 3: Execute with early-cutoff
        recomputed: Dict[str, Any] = {}
        cutoff_clean: Set[str] = set()

        for node_id in execution_order:
            if node_id in cutoff_clean:
                continue

            node = self._graph.get_node(node_id)
            if not isinstance(node, ComputeNode):
                continue

            # Gather dependency values
            dep_values = self._gather_dep_values(node)
            dep_generations = self._gather_dep_generations(node)

            # Execute computation
            new_value = node.func(dep_values)

            # Early cutoff: if value unchanged, don't propagate further
            old_value = node.cached_value
            node.mark_clean(new_value, dep_generations)
            self._cache.store(node_id, new_value, dep_generations)

            if new_value == old_value and old_value is not None:
                # Value didn't change — remove all downstream from dirty
                downstream = self._graph.get_all_downstream(node_id)
                for ds_id in downstream:
                    ds_node = self._graph.get_node(ds_id)
                    if isinstance(ds_node, ComputeNode) and ds_node.state == NodeState.DIRTY:
                        # Only cutoff if ALL their dirty deps have been cutoff
                        ds_node.state = NodeState.CLEAN
                        cutoff_clean.add(ds_id)
            else:
                recomputed[node_id] = new_value

        return recomputed

    def _gather_dep_values(self, node: ComputeNode) -> Dict[str, Any]:
        """Collect current values of all dependencies."""
        values: Dict[str, Any] = {}
        for dep_id in node.dependencies:
            dep_node = self._graph.get_node(dep_id)
            if isinstance(dep_node, InputNode):
                values[dep_id] = dep_node.value
            elif isinstance(dep_node, ComputeNode):
                values[dep_id] = dep_node.cached_value
        return values

    def _gather_dep_generations(self, node: ComputeNode) -> Dict[str, int]:
        """Collect current generations of all dependencies."""
        gens: Dict[str, int] = {}
        for dep_id in node.dependencies:
            dep_node = self._graph.get_node(dep_id)
            if isinstance(dep_node, InputNode):
                gens[dep_id] = dep_node.generation
            elif isinstance(dep_node, ComputeNode):
                # Compute nodes use a hash of their cached_value as generation proxy
                gens[dep_id] = self._cache.get_generation(dep_id)
        return gens
