"""Invalidator — marks downstream nodes dirty after input change."""
from __future__ import annotations
from collections import deque
from typing import List, Set
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode

class Invalidator:
    """Walks DAG downstream, marking nodes dirty via BFS."""
    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def invalidate_from(self, changed_ids: List[str]) -> Set[str]:
        newly_dirty: Set[str] = set()
        queue: deque = deque()
        for cid in changed_ids:
            for dep_id in self._graph.get_dependents(cid):
                if dep_id not in newly_dirty:
                    queue.append(dep_id)
                    newly_dirty.add(dep_id)
        while queue:
            current_id = queue.popleft()
            node = self._graph.get_node(current_id)
            if isinstance(node, ComputeNode):
                node.mark_dirty()
            for dep_id in self._graph.get_dependents(current_id):
                if dep_id not in newly_dirty:
                    newly_dirty.add(dep_id)
                    queue.append(dep_id)
        return newly_dirty
