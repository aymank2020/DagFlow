"""Topological scheduler with priority-aware tie-breaking."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode

class TopologicalScheduler:
    """Produce an execution plan in topological order with priority."""
    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def schedule(self, dirty_set: Set[str]) -> List[str]:
        if not dirty_set:
            return []
        in_degree: Dict[str, int] = {}
        sub_adj: Dict[str, List[str]] = defaultdict(list)
        for nid in dirty_set:
            node = self._graph.get_node(nid)
            if node.is_input:
                continue
            in_degree[nid] = 0
        for nid in in_degree:
            deps = self._graph.get_dependencies(nid)
            count = 0
            for dep_id in deps:
                if dep_id in in_degree:
                    sub_adj[dep_id].append(nid)
                    count += 1
            in_degree[nid] = count
        ready: List[Tuple[int, str]] = []
        for nid, deg in in_degree.items():
            if deg == 0:
                node = self._graph.get_node(nid)
                pri = node.priority if isinstance(node, ComputeNode) else 0
                ready.append((pri, nid))
        ready.sort()
        result: List[str] = []
        while ready:
            _, current = ready.pop(0)
            result.append(current)
            for dependent in sub_adj[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    dep_node = self._graph.get_node(dependent)
                    pri = dep_node.priority if isinstance(dep_node, ComputeNode) else 0
                    ready.append((pri, dependent))
                    ready.sort()
        return result

    def full_schedule(self) -> List[str]:
        all_compute = set(self._graph.get_compute_nodes())
        return self.schedule(all_compute)
