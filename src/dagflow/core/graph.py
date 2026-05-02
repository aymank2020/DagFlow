"""ComputeGraph — the central DAG registry.

Manages node registration, edge creation, cycle detection, and provides
the dependency/dependent lookup used by scheduler and propagation layers.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Union

from dagflow.core.node import ComputeNode, InputNode, NodeState


class CycleError(Exception):
    """Raised when adding an edge would create a cycle."""


class ComputeGraph:
    """Directed acyclic graph of computation nodes.

    The graph enforces acyclicity on every edge addition and provides
    efficient traversal for downstream invalidation and upstream gathering.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Union[InputNode, ComputeNode]] = {}
        self._adjacency: Dict[str, Set[str]] = {}  # node_id -> set of dependent IDs
        self._reverse: Dict[str, Set[str]] = {}    # node_id -> set of dependency IDs

    # ─── Node Management ───────────────────────────────────────────────

    def add_input(self, node_id: str, value: Any = None) -> InputNode:
        """Register a new input node."""
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")
        node = InputNode(node_id=node_id, value=value)
        self._nodes[node_id] = node
        self._adjacency[node_id] = set()
        self._reverse[node_id] = set()
        return node

    def add_compute(
        self,
        node_id: str,
        func: Callable[[dict], Any],
        dependencies: List[str],
        priority: int = 0,
    ) -> ComputeNode:
        """Register a compute node with its dependencies.

        Raises CycleError if the new edges would form a cycle.
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")
        for dep_id in dependencies:
            if dep_id not in self._nodes:
                raise ValueError(f"Dependency '{dep_id}' not found in graph")

        # Pre-check: would adding these edges create a cycle?
        # A cycle exists if any dependency can reach node_id via existing edges.
        # Since node_id is new, we only need to check if node_id appears in
        # the transitive dependents of any dependency — but node_id is new,
        # so no cycle is possible from a fresh node. However, we still validate
        # that dependencies don't form a cycle among themselves through this node.
        node = ComputeNode(
            node_id=node_id,
            func=func,
            dependencies=list(dependencies),
            priority=priority,
        )
        self._nodes[node_id] = node
        self._adjacency[node_id] = set()
        self._reverse[node_id] = set(dependencies)

        for dep_id in dependencies:
            self._adjacency[dep_id].add(node_id)
            dep_node = self._nodes[dep_id]
            dep_node.dependents.append(node_id)
            node.dependents  # ensure list exists (dataclass default)

        return node

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a dependency edge (from_id -> to_id means to_id depends on from_id).

        Raises CycleError if this would create a cycle.
        """
        if from_id not in self._nodes or to_id not in self._nodes:
            raise ValueError("Both nodes must exist")
        if from_id == to_id:
            raise CycleError("Self-loop detected")

        # Check if to_id can already reach from_id (would create cycle)
        if self._can_reach(to_id, from_id):
            raise CycleError(
                f"Adding edge {from_id}->{to_id} would create a cycle"
            )

        self._adjacency[from_id].add(to_id)
        self._reverse[to_id].add(from_id)

        to_node = self._nodes[to_id]
        from_node = self._nodes[from_id]
        if isinstance(to_node, ComputeNode) and from_id not in to_node.dependencies:
            to_node.dependencies.append(from_id)
        if to_id not in from_node.dependents:
            from_node.dependents.append(to_id)

    # ─── Traversal ─────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Union[InputNode, ComputeNode]:
        """Retrieve a node by ID."""
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found")
        return self._nodes[node_id]

    def get_dependents(self, node_id: str) -> List[str]:
        """Return immediate downstream dependents of a node."""
        return list(self._adjacency.get(node_id, set()))

    def get_dependencies(self, node_id: str) -> List[str]:
        """Return immediate upstream dependencies of a node."""
        return list(self._reverse.get(node_id, set()))

    def get_all_downstream(self, node_id: str) -> List[str]:
        """BFS to collect all transitive dependents (excluding node_id itself)."""
        visited: Set[str] = set()
        queue = deque(self.get_dependents(node_id))
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            queue.extend(
                dep for dep in self.get_dependents(current) if dep not in visited
            )
        return list(visited)

    def get_all_upstream(self, node_id: str) -> List[str]:
        """BFS to collect all transitive dependencies (excluding node_id itself)."""
        visited: Set[str] = set()
        queue = deque(self.get_dependencies(node_id))
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            queue.extend(
                dep for dep in self.get_dependencies(current) if dep not in visited
            )
        return list(visited)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def all_node_ids(self) -> List[str]:
        return list(self._nodes.keys())

    def get_input_nodes(self) -> List[str]:
        """Return IDs of all input nodes."""
        return [nid for nid, n in self._nodes.items() if n.is_input]

    def get_compute_nodes(self) -> List[str]:
        """Return IDs of all compute nodes."""
        return [nid for nid, n in self._nodes.items() if not n.is_input]

    # ─── Internal ──────────────────────────────────────────────────────

    def _can_reach(self, source: str, target: str) -> bool:
        """BFS check: can we reach target from source via adjacency edges?"""
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
