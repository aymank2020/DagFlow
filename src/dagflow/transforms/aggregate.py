"""Aggregate — multi-input aggregation nodes.

Provides nodes that combine values from multiple input sources using
standard aggregation operations (sum, average, min, max, count) or
custom aggregation functions.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode


class AggregateOp(Enum):
    """Built-in aggregation operations."""

    SUM = auto()
    AVERAGE = auto()
    MIN = auto()
    MAX = auto()
    COUNT = auto()
    PRODUCT = auto()


def _apply_aggregate(op: AggregateOp, values: List[Any]) -> Any:
    """Apply a built-in aggregation operation to a list of values.

    Args:
        op: The aggregation operation.
        values: List of numeric values to aggregate.

    Returns:
        The aggregated result.
    """
    # Filter out None values
    numeric_values = [v for v in values if v is not None]

    if not numeric_values:
        return None

    if op == AggregateOp.SUM:
        return sum(numeric_values)
    elif op == AggregateOp.AVERAGE:
        return sum(numeric_values) / len(numeric_values)
    elif op == AggregateOp.MIN:
        return min(numeric_values)
    elif op == AggregateOp.MAX:
        return max(numeric_values)
    elif op == AggregateOp.COUNT:
        return len(numeric_values)
    elif op == AggregateOp.PRODUCT:
        result = 1
        for v in numeric_values:
            result *= v
        return result
    else:
        raise ValueError(f"Unknown aggregate operation: {op}")


class AggregateNode:
    """Factory for creating aggregation compute nodes.

    An AggregateNode combines values from multiple source nodes using
    a specified aggregation operation.

    Usage:
        agg = AggregateNode(graph)
        agg.create("total", sources=["a", "b", "c"], op=AggregateOp.SUM)
        agg.create("avg_score", sources=["s1", "s2", "s3"], op=AggregateOp.AVERAGE)
    """

    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def create(
        self,
        node_id: str,
        sources: List[str],
        op: AggregateOp,
        priority: int = 0,
    ) -> ComputeNode:
        """Create an aggregation node with a built-in operation.

        Args:
            node_id: ID for the new node.
            sources: IDs of source nodes to aggregate.
            op: The aggregation operation to apply.
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        # Capture sources in closure for stable ordering
        source_ids = list(sources)

        def agg_func(deps: Dict[str, Any]) -> Any:
            values = [deps.get(sid) for sid in source_ids]
            return _apply_aggregate(op, values)

        return self._graph.add_compute(
            node_id, func=agg_func, dependencies=source_ids, priority=priority
        )

    def create_custom(
        self,
        node_id: str,
        sources: List[str],
        aggregator: Callable[[List[Any]], Any],
        priority: int = 0,
    ) -> ComputeNode:
        """Create an aggregation node with a custom function.

        Args:
            node_id: ID for the new node.
            sources: IDs of source nodes.
            aggregator: Custom function that receives a list of values.
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        source_ids = list(sources)

        def custom_agg_func(deps: Dict[str, Any]) -> Any:
            values = [deps.get(sid) for sid in source_ids]
            return aggregator(values)

        return self._graph.add_compute(
            node_id, func=custom_agg_func, dependencies=source_ids, priority=priority
        )

    def create_weighted(
        self,
        node_id: str,
        sources: List[str],
        weights: List[float],
        priority: int = 0,
    ) -> ComputeNode:
        """Create a weighted average aggregation node.

        Args:
            node_id: ID for the new node.
            sources: IDs of source nodes.
            weights: Weight for each source (must match length of sources).
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.

        Raises:
            ValueError: If sources and weights have different lengths.
        """
        if len(sources) != len(weights):
            raise ValueError("sources and weights must have the same length")

        source_ids = list(sources)
        weight_list = list(weights)
        total_weight = sum(weight_list)

        def weighted_func(deps: Dict[str, Any]) -> Any:
            weighted_sum = 0.0
            for sid, w in zip(source_ids, weight_list):
                value = deps.get(sid)
                if value is not None:
                    weighted_sum += value * w
            if total_weight == 0:
                return 0.0
            return weighted_sum / total_weight

        return self._graph.add_compute(
            node_id, func=weighted_func, dependencies=source_ids, priority=priority
        )
