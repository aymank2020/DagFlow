"""End-to-end integration tests combining multiple DagFlow modules.

These tests verify that modules work together correctly:
- Batch + Validation: constraints checked after batch apply.
- Persistence + Propagation: serialized graph propagates correctly.
- Debug + Transforms: tracer works with transform nodes.
- Full pipeline: source -> transform -> validate -> snapshot.
"""

import pytest
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode, ComputeNode
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.batch.executor import BatchExecutor
from dagflow.batch.transaction import Transaction
from dagflow.persistence.serializer import GraphSerializer
from dagflow.persistence.snapshot import Snapshot, SnapshotStore
from dagflow.persistence.replay import ReplayLog
from dagflow.debug.tracer import ExecutionTracer, EventType
from dagflow.debug.profiler import Profiler
from dagflow.debug.diff import GraphDiff
from dagflow.transforms.aggregate import AggregateNode, AggregateOp
from dagflow.transforms.conditional import ConditionalNode
from dagflow.transforms.map_reduce import MapNode, FilterNode
from dagflow.validation.constraints import (
    ConstraintRegistry,
    RangeConstraint,
    TypeConstraint,
)
from dagflow.validation.integrity import IntegrityChecker


class TestBatchWithValidation:
    """Integration: batch processing with constraint validation."""

    def test_batch_then_validate(self):
        """After batch apply, constraints can detect violations."""
        g = ComputeGraph()
        g.add_input("temp", value=70)
        g.add_compute("alert", func=lambda d: d["temp"] > 100, dependencies=["temp"])
        cache = MemoCache()

        registry = ConstraintRegistry(g, cache)
        registry.add("temp", RangeConstraint("safe_temp", max_value=120))

        executor = BatchExecutor(g, cache)
        executor.apply({"temp": 150})

        violations = registry.validate_all()
        assert len(violations) > 0
        assert violations[0].node_id == "temp"

    def test_transaction_rollback_preserves_valid_state(self):
        """Rollback after constraint violation restores valid state."""
        g = ComputeGraph()
        g.add_input("pressure", value=50)
        cache = MemoCache()

        registry = ConstraintRegistry(g, cache)
        registry.add("pressure", RangeConstraint("safe", max_value=100))

        txn = Transaction(g, cache)
        txn.begin()
        txn.set_input("pressure", 200)
        # Check constraint before committing
        txn.rollback()

        # State should be unchanged
        assert g.get_node("pressure").value == 50
        assert len(registry.validate_all()) == 0


class TestPersistenceWithPropagation:
    """Integration: serialized graphs still propagate correctly."""

    def test_deserialized_graph_propagates(self):
        """Graph deserialized from JSON propagates changes correctly."""
        # Build original graph
        g = ComputeGraph()
        g.add_input("x", value=5)
        double_func = lambda d: d["x"] * 2
        g.add_compute("y", func=double_func, dependencies=["x"])
        cache = MemoCache()
        EagerPropagator(g, cache).propagate(["x"])

        # Serialize
        serializer = GraphSerializer()
        serializer.register_function("double", double_func)
        data = serializer.serialize(g, cache)

        # Deserialize into new graph
        new_graph, new_cache = serializer.deserialize(data)

        # Modify and propagate on new graph
        new_graph.get_node("x").set(10)
        prop = EagerPropagator(new_graph, new_cache)
        result = prop.propagate(["x"])

        assert result.get("y") == 20

    def test_replay_produces_same_snapshots(self):
        """Replaying changes produces same state as original execution."""
        g = ComputeGraph()
        inp = g.add_input("counter", value=0)
        g.add_compute("doubled", func=lambda d: d["counter"] * 2, dependencies=["counter"])
        cache = MemoCache()
        EagerPropagator(g, cache).propagate(["counter"])

        # Record changes
        log = ReplayLog(g)
        log.start_recording()
        log.record_change("counter", 1)
        log.record_change("counter", 2)
        log.record_change("counter", 3)
        log.stop_recording()

        # Capture final state
        prop = EagerPropagator(g, cache)
        prop.propagate(["counter"])
        final_snap = Snapshot.capture("final", g, cache)

        # Replay on fresh graph
        g2 = ComputeGraph()
        g2.add_input("counter", value=0)
        g2.add_compute("doubled", func=lambda d: d["counter"] * 2, dependencies=["counter"])
        cache2 = MemoCache()

        log.replay(g2, cache2)
        replay_snap = Snapshot.capture("replay", g2, cache2)

        # Both should have same values
        assert replay_snap.get_value("counter") == final_snap.get_value("counter")
        assert replay_snap.get_value("doubled") == final_snap.get_value("doubled")


class TestDebugWithTransforms:
    """Integration: debug tools work with transform nodes."""

    def test_tracer_with_aggregate_nodes(self):
        """Tracer correctly records computation of aggregate nodes."""
        g = ComputeGraph()
        g.add_input("a", value=10)
        g.add_input("b", value=20)
        g.add_input("c", value=30)
        cache = MemoCache()

        agg = AggregateNode(g)
        agg.create("total", sources=["a", "b", "c"], op=AggregateOp.SUM)

        tracer = ExecutionTracer(g, cache)
        tracer.start()
        tracer.trace_propagation(["a", "b", "c"])

        assert "total" in tracer.computed_nodes

    def test_profiler_with_conditional_nodes(self):
        """Profiler times conditional node computations."""
        g = ComputeGraph()
        g.add_input("flag", value=True)
        g.add_input("yes", value=100)
        g.add_input("no", value=0)
        cache = MemoCache()

        cond = ConditionalNode(g)
        cond.create("result", condition="flag", then_source="yes", else_source="no")

        profiler = Profiler(g, cache)
        profiler.enable()
        profiler.profile_propagation(["flag", "yes", "no"])

        assert profiler.invocation_count("result") == 1
        assert profiler.total_time_ms() >= 0


class TestFullPipeline:
    """Integration: complete workflow combining all modules."""

    def test_source_transform_validate_snapshot(self):
        """Full pipeline: input -> transform -> validate -> snapshot."""
        g = ComputeGraph()
        g.add_input("raw_scores", value=[85, 92, 78, 95, 88])
        cache = MemoCache()

        # Transform: filter passing scores
        filterer = FilterNode(g)
        filterer.create("passing", source="raw_scores", predicate=lambda x: x >= 80)

        # Transform: compute average of passing scores
        g.add_compute(
            "avg_passing",
            func=lambda d: (
                sum(d["passing"]) / len(d["passing"]) if d["passing"] else 0
            ),
            dependencies=["passing"],
        )

        # Propagate
        prop = EagerPropagator(g, cache)
        prop.propagate(["raw_scores"])

        # Validate
        registry = ConstraintRegistry(g, cache)
        registry.add("avg_passing", RangeConstraint("valid_avg", min_value=0, max_value=100))
        violations = registry.validate_all()
        assert len(violations) == 0

        # Snapshot
        snap = Snapshot.capture("results", g, cache)
        assert snap.get_value("avg_passing") is not None

        # Integrity check
        checker = IntegrityChecker(g)
        assert checker.is_healthy()

    def test_batch_transform_diff(self):
        """Batch update -> transform -> diff shows what changed."""
        g = ComputeGraph()
        g.add_input("price", value=100)
        g.add_input("quantity", value=5)
        cache = MemoCache()

        # Compute total
        g.add_compute(
            "total",
            func=lambda d: d["price"] * d["quantity"],
            dependencies=["price", "quantity"],
        )

        prop = EagerPropagator(g, cache)
        prop.propagate(["price", "quantity"])

        # Snapshot before
        snap_before = Snapshot.capture("before", g, cache)

        # Batch update
        executor = BatchExecutor(g, cache)
        executor.apply({"price": 150, "quantity": 10})

        # Snapshot after
        snap_after = Snapshot.capture("after", g, cache)

        # Diff
        diff = GraphDiff.compare_snapshots(snap_before, snap_after)
        assert diff.has_changes
        assert "price" in diff.changed_nodes
        assert "total" in diff.changed_nodes

    def test_map_filter_reduce_pipeline(self):
        """Map -> Filter -> Reduce pipeline computes correctly."""
        g = ComputeGraph()
        g.add_input("data", value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cache = MemoCache()

        # Map: square each value
        mapper = MapNode(g)
        mapper.create("squared", source="data", transform=lambda x: x ** 2)

        # Filter: keep only values > 25
        filterer = FilterNode(g)
        filterer.create("large", source="squared", predicate=lambda x: x > 25)

        # Reduce: sum the remaining
        from dagflow.transforms.map_reduce import ReduceNode
        reducer = ReduceNode(g)
        reducer.create("total", source="large", reducer=lambda a, b: a + b, initial=0)

        prop = EagerPropagator(g, cache)
        prop.propagate(["data"])

        # squared = [1,4,9,16,25,36,49,64,81,100]
        # large = [36,49,64,81,100]
        # total = 330
        result = cache.get("total")
        assert result is not None
        assert result > 0  # Qualitative: sum of squares > 25 is positive
        # Structural: result should be sum of squares of numbers whose square > 25
        assert result == sum(x**2 for x in range(1, 11) if x**2 > 25)
