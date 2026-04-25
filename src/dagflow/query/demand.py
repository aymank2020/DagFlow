"""DemandEngine — demand-driven (lazy) recomputation."""
from __future__ import annotations
from collections import deque
from typing import Any, Dict, List, Set
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode, NodeState
from dagflow.memo.cache import MemoCache
from dagflow.scheduler.topo import TopologicalScheduler

class DemandEngine:
    """Lazy evaluation — computes only what's requested."""
    def __init__(self, graph: ComputeGraph, cache: MemoCache) -> None:
        self._graph = graph
        self._cache = cache
        self._scheduler = TopologicalScheduler(graph)

    def demand(self, node_id: str) -> Any:
        node = self._graph.get_node(node_id)
        if isinstance(node, InputNode):
            return node.value
        if node.state == NodeState.CLEAN and self._cache.get(node_id) is not None:
            return self._cache.get(node_id)
        dirty_ancestors = self._find_dirty_ancestors(node_id)
        dirty_ancestors.add(node_id)
        execution_order = self._scheduler.schedule(dirty_ancestors)
        for exec_id in execution_order:
            self._execute_node(exec_id)
        return self._cache.get(node_id)

    def _find_dirty_ancestors(self, node_id: str) -> Set[str]:
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
                for dep_id in node.dependencies:
                    if dep_id not in visited:
                        queue.append(dep_id)
        return dirty

    def _execute_node(self, node_id: str) -> None:
        node = self._graph.get_node(node_id)
        if not isinstance(node, ComputeNode):
            return
        if node.state == NodeState.CLEAN:
            return
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
        new_value = node.func(dep_values)
        node.mark_clean(new_value, dep_generations)
        self._cache.store(node_id, new_value, dep_generations)
