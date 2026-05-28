"""Debug — tracing, profiling, and visualization tools.

This module provides:
- ExecutionTracer: Records which nodes computed and in what order.
- Profiler: Timing profiler for node computations.
- DotExporter: Export graph to DOT/Graphviz format.
- GraphDiff: Compare two graph states and show what changed.
"""

from dagflow.debug.tracer import ExecutionTracer, TraceEvent
from dagflow.debug.profiler import Profiler, TimingRecord
from dagflow.debug.dot_export import DotExporter
from dagflow.debug.diff import GraphDiff, DiffEntry

__all__ = [
    "ExecutionTracer",
    "TraceEvent",
    "Profiler",
    "TimingRecord",
    "DotExporter",
    "GraphDiff",
    "DiffEntry",
]
