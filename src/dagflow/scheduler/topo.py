"""Topological scheduler — determines execution order for dirty nodes."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode

class TopologicalScheduler:
    """Produces execution plan for dirty nodes in topological order."""
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
        # Kahn's algorithm
        ready = [nid for nid, deg in in_degree.items() if deg == 0]
        ready.sort()
        result: List[str] = []
        while ready:
            current = ready.pop(0)
            result.append(current)
            for dep in sub_adj[current]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    ready.append(dep)
                    ready.sort()
        return result
