"""ComputeGraph — the central DAG registry."""
from __future__ import annotations
from collections import deque
from typing import Any, Callable, Dict, List, Set, Union
from dagflow.core.node import ComputeNode, InputNode

class CycleError(Exception):
    """Raised when adding an edge would create a cycle."""

class ComputeGraph:
    """Directed acyclic graph of computation nodes."""
    def __init__(self) -> None:
        self._nodes: Dict[str, Union[InputNode, ComputeNode]] = {}
        self._adjacency: Dict[str, Set[str]] = {}

    def add_input(self, node_id: str, value: Any = None) -> InputNode:
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")
        node = InputNode(node_id=node_id, value=value)
        self._nodes[node_id] = node
        self._adjacency[node_id] = set()
        return node

    def add_compute(self, node_id: str, func: Callable[[dict], Any],
                    dependencies: List[str], priority: int = 0) -> ComputeNode:
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")
        for dep_id in dependencies:
            if dep_id not in self._nodes:
                raise ValueError(f"Dependency '{dep_id}' not found in graph")
        node = ComputeNode(node_id=node_id, func=func,
                          dependencies=list(dependencies))
        self._nodes[node_id] = node
        self._adjacency[node_id] = set()
        for dep_id in dependencies:
            self._adjacency[dep_id].add(node_id)
            self._nodes[dep_id].dependents.append(node_id)
        return node

    def add_edge(self, from_id: str, to_id: str) -> None:
        if from_id not in self._nodes or to_id not in self._nodes:
            raise ValueError("Both nodes must exist")
        if from_id == to_id:
            raise CycleError("Self-loop detected")
        if self._can_reach(to_id, from_id):
            raise CycleError(f"Adding edge {from_id}->{to_id} would create a cycle")
        self._adjacency[from_id].add(to_id)

    def get_node(self, node_id: str) -> Union[InputNode, ComputeNode]:
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found")
        return self._nodes[node_id]

    def get_dependents(self, node_id: str) -> List[str]:
        return list(self._adjacency.get(node_id, set()))

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def _can_reach(self, source: str, target: str) -> bool:
        visited: Set[str] = set()
        queue = deque([source])
        while queue:
            current = queue.popleft()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self._adjacency.get(current, set()))
        return False
