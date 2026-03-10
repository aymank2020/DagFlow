"""ComputeGraph — the central DAG registry."""
from __future__ import annotations
from typing import Any, Dict, List, Set, Union
from dagflow.core.node import ComputeNode, InputNode

class ComputeGraph:
    """Directed acyclic graph of computation nodes."""
    def __init__(self) -> None:
        self._nodes: Dict[str, Union[InputNode, ComputeNode]] = {}

    def add_input(self, node_id: str, value: Any = None) -> InputNode:
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")
        node = InputNode(node_id=node_id, value=value)
        self._nodes[node_id] = node
        return node

    def add_compute(self, node_id: str, func, dependencies: List[str]) -> ComputeNode:
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")
        for dep_id in dependencies:
            if dep_id not in self._nodes:
                raise ValueError(f"Dependency '{dep_id}' not found")
        node = ComputeNode(node_id=node_id, func=func, dependencies=list(dependencies))
        self._nodes[node_id] = node
        for dep_id in dependencies:
            self._nodes[dep_id].dependents.append(node_id)
        return node

    def get_node(self, node_id: str) -> Union[InputNode, ComputeNode]:
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found")
        return self._nodes[node_id]

    @property
    def node_count(self) -> int:
        return len(self._nodes)
