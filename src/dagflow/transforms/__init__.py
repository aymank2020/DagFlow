"""Transforms — functional operations over DAG node values.

This module provides higher-level computation patterns built on top of
the core DAG primitives:
- map_reduce: Map, filter, and reduce operations over node values.
- window: Sliding window computations (last N values of a node).
- aggregate: Aggregation nodes (sum, avg, min, max over multiple inputs).
- conditional: Conditional computation (if-then-else, switch).
"""

from dagflow.transforms.map_reduce import MapNode, FilterNode, ReduceNode
from dagflow.transforms.window import WindowNode, WindowBuffer
from dagflow.transforms.aggregate import AggregateNode, AggregateOp
from dagflow.transforms.conditional import ConditionalNode, SwitchNode

__all__ = [
    "MapNode",
    "FilterNode",
    "ReduceNode",
    "WindowNode",
    "WindowBuffer",
    "AggregateNode",
    "AggregateOp",
    "ConditionalNode",
    "SwitchNode",
]
