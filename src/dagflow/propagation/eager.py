"""Eager propagation — recomputes dirty nodes immediately."""
from __future__ import annotations
from typing import Any, Dict, List, Set
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode, NodeState
from dagflow.scheduler.topo import TopologicalScheduler
from dagflow.propagation.invalidator import Invalidator
from dagflow.memo.cache import MemoCache

class EagerPropagator:
    """Eagerly recomputes all dirty nodes after each input change."""
    def __init__(self, graph: ComputeGraph, cache: MemoCache) -> None:
        self._graph = graph
        self._cache = cache
        self._invalidator = Invalidator(graph)
        self._scheduler = TopologicalScheduler(graph)

    def propagate(self, changed_inputs: List[str]) -> Dict[str, Any]:
        dirty_set = self._invalidator.invalidate_from(changed_inputs)
        execution_order = self._scheduler.schedule(dirty_set)
        recomputed: Dict[str, Any] = {}
        for node_id in execution_order:
            node = self._graph.get_node(node_id)
            if not isinstance(node, ComputeNode):
                continue
            dep_values = self._gather_dep_values(node)
            dep_gens = self._gather_dep_generations(node)
            new_value = node.func(dep_values)
            node.mark_clean(new_value, dep_gens)
            self._cache.store(node_id, new_value, dep_gens)
            recomputed[node_id] = new_value
        return recomputed

    def _gather_dep_values(self, node: ComputeNode) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for dep_id in node.dependencies:
            dep_node = self._graph.get_node(dep_id)
            if isinstance(dep_node, InputNode):
                values[dep_id] = dep_node.value
            elif isinstance(dep_node, ComputeNode):
                values[dep_id] = dep_node.cached_value
        return values

    def _gather_dep_generations(self, node: ComputeNode) -> Dict[str, int]:
        gens: Dict[str, int] = {}
        for dep_id in node.dependencies:
            dep_node = self._graph.get_node(dep_id)
            if isinstance(dep_node, InputNode):
                gens[dep_id] = dep_node.generation
            elif isinstance(dep_node, ComputeNode):
                gens[dep_id] = self._cache.get_generation(dep_id)
        return gens
