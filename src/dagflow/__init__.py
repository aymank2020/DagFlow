"""DagFlow — Incremental computation engine with DAG-based memoization."""

__version__ = "0.2.0"

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode, ComputeNode
from dagflow.scheduler.topo import TopologicalScheduler
from dagflow.propagation.eager import EagerPropagator
from dagflow.propagation.invalidator import Invalidator
from dagflow.memo.cache import MemoCache
from dagflow.query.demand import DemandEngine

__all__ = [
    "ComputeGraph",
    "InputNode",
    "ComputeNode",
    "TopologicalScheduler",
    "EagerPropagator",
    "Invalidator",
    "MemoCache",
    "DemandEngine",
]
