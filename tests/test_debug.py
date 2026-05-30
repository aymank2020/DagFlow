"""Tests for debug tools — tracer, profiler, and DOT export.

Tests verify behavioral invariants:
- Tracer records events in correct order.
- Profiler collects timing data for computed nodes.
- DOT export produces valid graph syntax.
- Diff detects actual changes between states.
"""

import pytest
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode, ComputeNode
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.debug.tracer import ExecutionTracer, EventType
from dagflow.debug.profiler import Profiler
from dagflow.debug.dot_export import DotExporter
from dagflow.debug.diff import GraphDiff, ChangeType
from dagflow.persistence.snapshot import Snapshot


class TestExecutionTracer:
    """Tests for the execution tracer."""

    def test_tracer_records_computed_nodes(self):
        """Tracer records which nodes were computed."""
        g = ComputeGraph()
        g.add_input("x", value=5)
        g.add_compute("y", func=lambda d: d["x"] * 2, dependencies=["x"])
        cache = MemoCache()

        tracer = ExecutionTracer(g, cache)
        tracer.start()
        tracer.trace_propagation(["x"])
        tracer.stop()

        assert "y" in tracer.computed_nodes

    def test_tracer_records_correct_order(self):
        """Events are recorded in sequential order."""
        g = ComputeGraph()
        g.add_input("a", value=1)
        g.add_compute("b", func=lambda d: d["a"] + 1, dependencies=["a"])
        g.add_compute("c", func=lambda d: d["b"] + 1, dependencies=["b"])
        cache = MemoCache()

        tracer = ExecutionTracer(g, cache)
        tracer.start()
        tracer.trace_propagation(["a"])

        computed = tracer.computed_nodes
        # b must be computed before c (topological order)
        assert computed.index("b") < computed.index("c")

    def test_tracer_detects_early_cutoff(self):
        """Tracer records cutoff when value doesn't change."""
        g = ComputeGraph()
        inp = g.add_input("x", value=5)
        # This node always returns constant regardless of input
        g.add_compute("const", func=lambda d: 42, dependencies=["x"])
        g.add_compute("downstream", func=lambda d: d["const"] + 1, dependencies=["const"])
        cache = MemoCache()

        tracer = ExecutionTracer(g, cache)
        tracer.start()
        tracer.trace_propagation(["x"])  # First computation

        tracer.clear()
        inp.set(999)  # Change input but const still returns 42
        tracer.trace_propagation(["x"])

        # const should be cutoff (same value), downstream should not compute
        assert "const" in tracer.cutoff_nodes

    def test_tracer_inactive_does_plain_propagation(self):
        """When not active, tracer just propagates without recording."""
        g = ComputeGraph()
        g.add_input("x", value=3)
        g.add_compute("y", func=lambda d: d["x"] + 1, dependencies=["x"])
        cache = MemoCache()

        tracer = ExecutionTracer(g, cache)
        # Don't call start()
        result = tracer.trace_propagation(["x"])

        assert "y" in result
        assert len(tracer.events) == 0

    def test_get_events_for_node(self):
        """Can filter events by node ID."""
        g = ComputeGraph()
        g.add_input("x", value=1)
        g.add_compute("y", func=lambda d: d["x"], dependencies=["x"])
        cache = MemoCache()

        tracer = ExecutionTracer(g, cache)
        tracer.start()
        tracer.trace_propagation(["x"])

        y_events = tracer.get_events_for_node("y")
        assert len(y_events) > 0
        assert all(e.node_id == "y" for e in y_events)


class TestProfiler:
    """Tests for the computation profiler."""

    def test_profiler_collects_timing_data(self):
        """Profiler records timing for each computed node."""
        g = ComputeGraph()
        g.add_input("x", value=10)
        g.add_compute("y", func=lambda d: d["x"] ** 2, dependencies=["x"])
        cache = MemoCache()

        profiler = Profiler(g, cache)
        profiler.enable()
        profiler.profile_propagation(["x"])

        assert len(profiler.records) > 0
        assert profiler.records[0].node_id == "y"
        assert profiler.records[0].duration_ms >= 0

    def test_profiler_tracks_invocation_count(self):
        """Profiler counts how many times each node is computed."""
        g = ComputeGraph()
        inp = g.add_input("x", value=1)
        g.add_compute("y", func=lambda d: d["x"] + 1, dependencies=["x"])
        cache = MemoCache()

        profiler = Profiler(g, cache)
        profiler.enable()

        profiler.profile_propagation(["x"])
        inp.set(2)
        profiler.profile_propagation(["x"])

        assert profiler.invocation_count("y") == 2

    def test_profiler_total_time_is_sum(self):
        """Total time equals sum of individual record times."""
        g = ComputeGraph()
        g.add_input("x", value=1)
        g.add_compute("a", func=lambda d: d["x"] + 1, dependencies=["x"])
        g.add_compute("b", func=lambda d: d["a"] + 1, dependencies=["a"])
        cache = MemoCache()

        profiler = Profiler(g, cache)
        profiler.enable()
        profiler.profile_propagation(["x"])

        individual_sum = sum(r.duration_ms for r in profiler.records)
        assert abs(profiler.total_time_ms() - individual_sum) < 0.001

    def test_profiler_summary_is_nonempty(self):
        """Summary produces readable output after profiling."""
        g = ComputeGraph()
        g.add_input("x", value=1)
        g.add_compute("y", func=lambda d: d["x"], dependencies=["x"])
        cache = MemoCache()

        profiler = Profiler(g, cache)
        profiler.enable()
        profiler.profile_propagation(["x"])

        summary = profiler.summary()
        assert "y" in summary
        assert "ms" in summary


class TestDotExporter:
    """Tests for DOT/Graphviz export."""

    def test_export_produces_valid_dot_structure(self):
        """Exported DOT has digraph wrapper and node definitions."""
        g = ComputeGraph()
        g.add_input("x", value=1)
        g.add_compute("y", func=lambda d: d["x"], dependencies=["x"])
        cache = MemoCache()

        exporter = DotExporter(g, cache)
        dot = exporter.export()

        assert dot.startswith("digraph")
        assert '"x"' in dot
        assert '"y"' in dot
        assert "->" in dot  # Has edges

    def test_export_contains_all_nodes(self):
        """Every node in the graph appears in the DOT output."""
        g = ComputeGraph()
        g.add_input("a")
        g.add_input("b")
        g.add_compute("c", func=lambda d: None, dependencies=["a", "b"])

        exporter = DotExporter(g)
        dot = exporter.export()

        assert '"a"' in dot
        assert '"b"' in dot
        assert '"c"' in dot

    def test_export_subgraph_limits_scope(self):
        """Subgraph export only includes reachable nodes."""
        g = ComputeGraph()
        g.add_input("x")
        g.add_input("y")
        g.add_compute("cx", func=lambda d: None, dependencies=["x"])
        g.add_compute("cy", func=lambda d: None, dependencies=["y"])

        exporter = DotExporter(g)
        dot = exporter.export_subgraph("x")

        assert '"x"' in dot
        assert '"cx"' in dot
        assert '"cy"' not in dot  # Not reachable from x

    def test_show_values_includes_values_in_labels(self):
        """With show_values=True, node labels include current values."""
        g = ComputeGraph()
        g.add_input("x", value=42)
        cache = MemoCache()

        exporter = DotExporter(g, cache)
        exporter.show_values = True
        dot = exporter.export()

        assert "42" in dot


class TestGraphDiff:
    """Tests for graph state comparison."""

    def test_diff_detects_value_changes(self):
        """Diff finds nodes whose values changed between snapshots."""
        g = ComputeGraph()
        inp = g.add_input("x", value=1)
        g.add_compute("y", func=lambda d: d["x"] * 10, dependencies=["x"])
        cache = MemoCache()
        EagerPropagator(g, cache).propagate(["x"])

        snap1 = Snapshot.capture("before", g, cache)

        inp.set(5)
        EagerPropagator(g, cache).propagate(["x"])
        snap2 = Snapshot.capture("after", g, cache)

        diff = GraphDiff.compare_snapshots(snap1, snap2)
        assert diff.has_changes
        assert "x" in diff.changed_nodes
        assert "y" in diff.changed_nodes

    def test_diff_no_changes_when_identical(self):
        """Diff reports no changes between identical snapshots."""
        g = ComputeGraph()
        g.add_input("x", value=1)
        cache = MemoCache()

        snap1 = Snapshot.capture("s1", g, cache)
        snap2 = Snapshot.capture("s2", g, cache)

        diff = GraphDiff.compare_snapshots(snap1, snap2)
        assert not diff.has_changes

    def test_diff_filter_by_type(self):
        """Can filter diff entries by change type."""
        g = ComputeGraph()
        inp = g.add_input("x", value=1)
        cache = MemoCache()

        snap1 = Snapshot.capture("before", g, cache)
        inp.set(2)
        snap2 = Snapshot.capture("after", g, cache)

        diff = GraphDiff.compare_snapshots(snap1, snap2)
        value_changes = diff.filter_by_type(ChangeType.VALUE_CHANGED)
        assert len(value_changes) > 0

    def test_diff_summary_is_readable(self):
        """Summary produces human-readable output."""
        g = ComputeGraph()
        inp = g.add_input("x", value=1)
        cache = MemoCache()

        snap1 = Snapshot.capture("before", g, cache)
        inp.set(99)
        snap2 = Snapshot.capture("after", g, cache)

        diff = GraphDiff.compare_snapshots(snap1, snap2)
        summary = diff.summary()
        assert "x" in summary
        assert "VALUE_CHANGED" in summary
