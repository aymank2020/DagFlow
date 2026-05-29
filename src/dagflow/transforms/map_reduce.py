"""Map/Filter/Reduce — functional transforms over DAG node values.

Provides factory functions that create compute nodes implementing
common functional patterns. These nodes integrate with the standard
DAG propagation and memoization infrastructure.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode


class MapNode:
    """Factory for creating map-transform compute nodes.

    A MapNode applies a transformation function to a single input node's
    value. If the input value is a collection, the transform is applied
    to each element.

    Usage:
        mapper = MapNode(graph)
        mapper.create("doubled", source="numbers", transform=lambda x: x * 2)
    """

    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def create(
        self,
        node_id: str,
        source: str,
        transform: Callable[[Any], Any],
        element_wise: bool = True,
        priority: int = 0,
    ) -> ComputeNode:
        """Create a map node that transforms values from a source node.

        Args:
            node_id: ID for the new compute node.
            source: ID of the source node to read from.
            transform: Function to apply to the value (or each element).
            element_wise: If True and source value is a list, apply to each element.
                         If False, apply to the entire value.
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        if element_wise:
            def map_func(deps: Dict[str, Any]) -> Any:
                value = deps[source]
                if isinstance(value, (list, tuple)):
                    return type(value)(transform(item) for item in value)
                return transform(value)
        else:
            def map_func(deps: Dict[str, Any]) -> Any:
                return transform(deps[source])

        return self._graph.add_compute(
            node_id, func=map_func, dependencies=[source], priority=priority
        )

    def create_chain(
        self,
        node_ids: List[str],
        source: str,
        transforms: List[Callable[[Any], Any]],
    ) -> List[ComputeNode]:
        """Create a chain of map nodes, each feeding into the next.

        Args:
            node_ids: IDs for each node in the chain.
            source: ID of the initial source node.
            transforms: Transform functions, one per node.

        Returns:
            List of created ComputeNodes.
        """
        if len(node_ids) != len(transforms):
            raise ValueError("node_ids and transforms must have same length")

        nodes: List[ComputeNode] = []
        current_source = source

        for nid, transform in zip(node_ids, transforms):
            node = self.create(nid, current_source, transform, element_wise=False)
            nodes.append(node)
            current_source = nid

        return nodes


class FilterNode:
    """Factory for creating filter-transform compute nodes.

    A FilterNode takes a collection-valued source and produces a filtered
    subset based on a predicate function.

    Usage:
        filterer = FilterNode(graph)
        filterer.create("evens", source="numbers", predicate=lambda x: x % 2 == 0)
    """

    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def create(
        self,
        node_id: str,
        source: str,
        predicate: Callable[[Any], bool],
        priority: int = 0,
    ) -> ComputeNode:
        """Create a filter node.

        Args:
            node_id: ID for the new node.
            source: ID of the source node (should produce a collection).
            predicate: Function that returns True for items to keep.
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        def filter_func(deps: Dict[str, Any]) -> Any:
            value = deps[source]
            if isinstance(value, (list, tuple)):
                return type(value)(item for item in value if predicate(item))
            # For non-collections, return value if predicate passes, else None
            return value if predicate(value) else None

        return self._graph.add_compute(
            node_id, func=filter_func, dependencies=[source], priority=priority
        )


class ReduceNode:
    """Factory for creating reduce-transform compute nodes.

    A ReduceNode takes a collection-valued source and reduces it to a
    single value using a binary function and initial accumulator.

    Usage:
        reducer = ReduceNode(graph)
        reducer.create("total", source="numbers",
                       reducer=lambda acc, x: acc + x, initial=0)
    """

    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def create(
        self,
        node_id: str,
        source: str,
        reducer: Callable[[Any, Any], Any],
        initial: Any = None,
        priority: int = 0,
    ) -> ComputeNode:
        """Create a reduce node.

        Args:
            node_id: ID for the new node.
            source: ID of the source node (should produce a collection).
            reducer: Binary function (accumulator, item) -> new_accumulator.
            initial: Initial accumulator value.
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        def reduce_func(deps: Dict[str, Any]) -> Any:
            value = deps[source]
            if not isinstance(value, (list, tuple)):
                return value

            acc = initial
            for item in value:
                if acc is None:
                    acc = item
                else:
                    acc = reducer(acc, item)
            return acc

        return self._graph.add_compute(
            node_id, func=reduce_func, dependencies=[source], priority=priority
        )

    def create_multi_source(
        self,
        node_id: str,
        sources: List[str],
        reducer: Callable[[Any, Any], Any],
        initial: Any = None,
        priority: int = 0,
    ) -> ComputeNode:
        """Create a reduce node that combines values from multiple sources.

        Args:
            node_id: ID for the new node.
            sources: IDs of source nodes to reduce over.
            reducer: Binary function (accumulator, value) -> new_accumulator.
            initial: Initial accumulator value.
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        def multi_reduce_func(deps: Dict[str, Any]) -> Any:
            acc = initial
            for src_id in sources:
                value = deps.get(src_id)
                if acc is None:
                    acc = value
                else:
                    acc = reducer(acc, value)
            return acc

        return self._graph.add_compute(
            node_id, func=multi_reduce_func, dependencies=sources, priority=priority
        )
