"""Federated graph coordination — multiple graphs with cross-graph dependencies.

Enables composing multiple independent ComputeGraphs into a federated
system where nodes in one graph can depend on nodes in another. The
federation layer manages cross-graph edge resolution, propagation
ordering, and namespace isolation.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode)
- propagation.eager (EagerPropagator)
- memo.cache (MemoCache)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator


@dataclass
class GraphEndpoint:
    """A registered graph within the federation.

    Attributes:
        name: Unique namespace for this graph.
        graph: The ComputeGraph instance.
        cache: Associated MemoCache.
        propagator: EagerPropagator for this graph.
        exported_nodes: Node IDs that other graphs can depend on.
    """

    name: str
    graph: ComputeGraph
    cache: MemoCache
    propagator: Optional[EagerPropagator] = None
    exported_nodes: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.propagator is None:
            self.propagator = EagerPropagator(self.graph, self.cache)

    @property
    def qualified_prefix(self) -> str:
        """Namespace prefix for qualified node references."""
        return f"{self.name}::"


@dataclass
class CrossGraphEdge:
    """A dependency edge that crosses graph boundaries.

    Attributes:
        source_graph: Name of the graph containing the source node.
        source_node: Node ID in the source graph.
        target_graph: Name of the graph containing the target node.
        target_node: Node ID in the target graph (must be an input).
        transform: Optional function to transform the value crossing the boundary.
        last_synced_generation: Generation of source when last synced.
    """

    source_graph: str
    source_node: str
    target_graph: str
    target_node: str
    transform: Optional[Callable[[Any], Any]] = None
    last_synced_generation: int = 0

    @property
    def qualified_source(self) -> str:
        """Fully qualified source reference."""
        return f"{self.source_graph}::{self.source_node}"

    @property
    def qualified_target(self) -> str:
        """Fully qualified target reference."""
        return f"{self.target_graph}::{self.target_node}"


class FederationError(Exception):
    """Raised when federation operations fail."""


class FederatedGraph:
    """Coordinates multiple ComputeGraphs with cross-graph dependencies.

    The federation maintains a registry of graphs (endpoints) and
    cross-graph edges. When a source node changes, the federation
    propagates the value to the target graph's input node, triggering
    downstream recomputation in the target graph.

    Usage:
        fed = FederatedGraph()
        fed.register("pricing", pricing_graph, pricing_cache)
        fed.register("inventory", inv_graph, inv_cache)
        fed.add_cross_edge("pricing", "unit_price", "inventory", "price_input")
        results = fed.propagate("pricing", ["unit_price"])
    """

    def __init__(self) -> None:
        self._endpoints: Dict[str, GraphEndpoint] = {}
        self._cross_edges: List[CrossGraphEdge] = []
        self._edge_index: Dict[str, List[CrossGraphEdge]] = {}  # source_qualified -> edges

    @property
    def graph_count(self) -> int:
        """Number of registered graphs."""
        return len(self._endpoints)

    @property
    def cross_edge_count(self) -> int:
        """Number of cross-graph edges."""
        return len(self._cross_edges)

    def register(
        self,
        name: str,
        graph: ComputeGraph,
        cache: MemoCache,
        exported: Optional[Set[str]] = None,
    ) -> GraphEndpoint:
        """Register a graph as an endpoint in the federation.

        Args:
            name: Unique namespace for this graph.
            graph: The ComputeGraph instance.
            cache: Associated MemoCache.
            exported: Set of node IDs that can be referenced by other graphs.

        Returns:
            The created GraphEndpoint.

        Raises:
            FederationError: If name is already registered.
        """
        if name in self._endpoints:
            raise FederationError(f"Graph '{name}' already registered")

        endpoint = GraphEndpoint(
            name=name,
            graph=graph,
            cache=cache,
            exported_nodes=exported or set(),
        )
        self._endpoints[name] = endpoint
        return endpoint

    def unregister(self, name: str) -> bool:
        """Remove a graph from the federation.

        Also removes all cross-edges involving this graph.

        Returns:
            True if the graph was found and removed.
        """
        if name not in self._endpoints:
            return False

        # Remove cross-edges involving this graph
        self._cross_edges = [
            e
            for e in self._cross_edges
            if e.source_graph != name and e.target_graph != name
        ]
        self._rebuild_edge_index()
        del self._endpoints[name]
        return True

    def add_cross_edge(
        self,
        source_graph: str,
        source_node: str,
        target_graph: str,
        target_node: str,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> CrossGraphEdge:
        """Add a cross-graph dependency edge.

        The target_node must be an InputNode in the target graph.
        When source_node changes, its value (optionally transformed)
        is pushed to target_node.

        Args:
            source_graph: Name of the source graph.
            source_node: Node ID in the source graph.
            target_graph: Name of the target graph.
            target_node: Node ID in the target graph (must be InputNode).
            transform: Optional value transformation function.

        Returns:
            The created CrossGraphEdge.

        Raises:
            FederationError: If graphs don't exist or target isn't an input.
        """
        if source_graph not in self._endpoints:
            raise FederationError(f"Source graph '{source_graph}' not registered")
        if target_graph not in self._endpoints:
            raise FederationError(f"Target graph '{target_graph}' not registered")
        if source_graph == target_graph:
            raise FederationError("Cross-edge cannot be within the same graph")

        # Validate nodes exist
        src_endpoint = self._endpoints[source_graph]
        tgt_endpoint = self._endpoints[target_graph]
        src_endpoint.graph.get_node(source_node)  # Raises KeyError if missing

        tgt_node = tgt_endpoint.graph.get_node(target_node)
        if not isinstance(tgt_node, InputNode):
            raise FederationError(
                f"Target node '{target_node}' must be an InputNode"
            )

        edge = CrossGraphEdge(
            source_graph=source_graph,
            source_node=source_node,
            target_graph=target_graph,
            target_node=target_node,
            transform=transform,
        )
        self._cross_edges.append(edge)

        # Update index
        key = edge.qualified_source
        if key not in self._edge_index:
            self._edge_index[key] = []
        self._edge_index[key].append(edge)

        return edge

    def propagate(
        self, graph_name: str, changed_inputs: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Propagate changes within a graph and across cross-edges.

        First propagates within the source graph, then pushes changed
        values across cross-edges to target graphs, triggering their
        propagation in turn.

        Args:
            graph_name: The graph where changes originated.
            changed_inputs: Input node IDs that changed.

        Returns:
            Dict of {graph_name: {node_id: new_value}} for all affected graphs.
        """
        if graph_name not in self._endpoints:
            raise FederationError(f"Graph '{graph_name}' not registered")

        all_results: Dict[str, Dict[str, Any]] = {}
        propagation_queue: deque = deque([(graph_name, changed_inputs)])
        visited_graphs: Set[str] = set()

        while propagation_queue:
            current_graph, inputs = propagation_queue.popleft()

            if current_graph in visited_graphs:
                continue
            visited_graphs.add(current_graph)

            endpoint = self._endpoints[current_graph]
            assert endpoint.propagator is not None

            # Propagate within this graph
            recomputed = endpoint.propagator.propagate(inputs)
            if recomputed:
                all_results[current_graph] = recomputed

            # Check cross-edges for any changed nodes
            all_changed = set(inputs) | set(recomputed.keys())
            for node_id in all_changed:
                key = f"{current_graph}::{node_id}"
                edges = self._edge_index.get(key, [])

                for edge in edges:
                    # Get the source value
                    src_node = endpoint.graph.get_node(edge.source_node)
                    if isinstance(src_node, InputNode):
                        value = src_node.value
                    elif isinstance(src_node, ComputeNode):
                        value = src_node.cached_value
                    else:
                        continue

                    # Apply transform if present
                    if edge.transform:
                        value = edge.transform(value)

                    # Push to target
                    tgt_endpoint = self._endpoints[edge.target_graph]
                    tgt_node = tgt_endpoint.graph.get_node(edge.target_node)
                    if isinstance(tgt_node, InputNode):
                        tgt_node.set(value)
                        propagation_queue.append(
                            (edge.target_graph, [edge.target_node])
                        )

        return all_results

    def get_endpoint(self, name: str) -> GraphEndpoint:
        """Retrieve a registered endpoint by name."""
        if name not in self._endpoints:
            raise FederationError(f"Graph '{name}' not registered")
        return self._endpoints[name]

    def get_cross_edges_from(self, graph_name: str) -> List[CrossGraphEdge]:
        """Get all cross-edges originating from a graph."""
        return [e for e in self._cross_edges if e.source_graph == graph_name]

    def get_cross_edges_to(self, graph_name: str) -> List[CrossGraphEdge]:
        """Get all cross-edges targeting a graph."""
        return [e for e in self._cross_edges if e.target_graph == graph_name]

    def _rebuild_edge_index(self) -> None:
        """Rebuild the edge lookup index after modifications."""
        self._edge_index.clear()
        for edge in self._cross_edges:
            key = edge.qualified_source
            if key not in self._edge_index:
                self._edge_index[key] = []
            self._edge_index[key].append(edge)
