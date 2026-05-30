"""Tests for batch processing — transactions, checkpoints, and batch executor.

Tests verify behavioral invariants:
- Transactions are atomic (all-or-nothing).
- Rollback restores previous state.
- Batch executor applies changes correctly.
- Dry-run doesn't modify state.
"""

import pytest
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode, ComputeNode
from dagflow.memo.cache import MemoCache
from dagflow.batch.checkpoint import Checkpoint
from dagflow.batch.transaction import Transaction, TransactionError
from dagflow.batch.executor import BatchExecutor


class TestCheckpoint:
    """Tests for checkpoint save/restore."""

    def test_capture_and_restore_input_values(self):
        """Restoring a checkpoint reverts input values."""
        g = ComputeGraph()
        inp = g.add_input("x", value=10)
        cache = MemoCache()

        cp = Checkpoint(g, cache)
        cp.capture()

        inp.set(999)
        assert inp.value == 999

        cp.restore()
        assert inp.value == 10

    def test_capture_and_restore_compute_state(self):
        """Restoring a checkpoint reverts compute node cached values."""
        g = ComputeGraph()
        g.add_input("x", value=5)
        g.add_compute("y", func=lambda d: d["x"] * 2, dependencies=["x"])
        cache = MemoCache()

        from dagflow.propagation.eager import EagerPropagator
        prop = EagerPropagator(g, cache)
        prop.propagate(["x"])

        cp = Checkpoint(g, cache)
        cp.capture()

        # Change and propagate
        g.get_node("x").set(100)
        prop.propagate(["x"])
        assert cache.get("y") == 200

        # Restore
        cp.restore()
        assert cache.get("y") == 10

    def test_restore_without_capture_raises(self):
        """Restoring without capturing first raises RuntimeError."""
        g = ComputeGraph()
        g.add_input("x")
        cache = MemoCache()
        cp = Checkpoint(g, cache)

        with pytest.raises(RuntimeError):
            cp.restore()

    def test_discard_clears_state(self):
        """After discard, is_captured returns False."""
        g = ComputeGraph()
        g.add_input("x")
        cache = MemoCache()
        cp = Checkpoint(g, cache)
        cp.capture()
        assert cp.is_captured

        cp.discard()
        assert not cp.is_captured


class TestTransaction:
    """Tests for transaction semantics."""

    def test_commit_applies_all_changes(self):
        """Committed transaction applies all buffered changes."""
        g = ComputeGraph()
        g.add_input("a", value=1)
        g.add_input("b", value=2)
        g.add_compute("sum", func=lambda d: d["a"] + d["b"], dependencies=["a", "b"])
        cache = MemoCache()

        with Transaction(g, cache) as txn:
            txn.set_input("a", 10)
            txn.set_input("b", 20)

        assert g.get_node("a").value == 10
        assert g.get_node("b").value == 20

    def test_rollback_reverts_changes(self):
        """Rolled-back transaction leaves graph unchanged."""
        g = ComputeGraph()
        inp = g.add_input("x", value=42)
        cache = MemoCache()

        txn = Transaction(g, cache)
        txn.begin()
        txn.set_input("x", 999)
        txn.rollback()

        assert inp.value == 42

    def test_exception_triggers_rollback(self):
        """Exception inside context manager triggers automatic rollback."""
        g = ComputeGraph()
        inp = g.add_input("x", value=5)
        cache = MemoCache()

        with pytest.raises(ValueError):
            with Transaction(g, cache) as txn:
                txn.set_input("x", 100)
                raise ValueError("simulated error")

        assert inp.value == 5

    def test_nested_begin_raises(self):
        """Starting a transaction while one is active raises."""
        g = ComputeGraph()
        g.add_input("x")
        cache = MemoCache()

        txn = Transaction(g, cache)
        txn.begin()
        with pytest.raises(TransactionError):
            txn.begin()
        txn.rollback()

    def test_set_input_on_compute_node_raises(self):
        """Setting a compute node via transaction raises ValueError."""
        g = ComputeGraph()
        g.add_input("x", value=1)
        g.add_compute("y", func=lambda d: d["x"], dependencies=["x"])
        cache = MemoCache()

        with pytest.raises(ValueError):
            with Transaction(g, cache) as txn:
                txn.set_input("y", 99)


class TestBatchExecutor:
    """Tests for BatchExecutor."""

    def test_apply_multiple_changes(self):
        """Batch apply updates multiple inputs atomically."""
        g = ComputeGraph()
        g.add_input("a", value=0)
        g.add_input("b", value=0)
        g.add_compute("sum", func=lambda d: d["a"] + d["b"], dependencies=["a", "b"])
        cache = MemoCache()

        executor = BatchExecutor(g, cache)
        result = executor.apply({"a": 10, "b": 20})

        assert result.success
        assert g.get_node("a").value == 10
        assert g.get_node("b").value == 20

    def test_apply_skips_unchanged_values(self):
        """Values that haven't changed are reported as skipped."""
        g = ComputeGraph()
        g.add_input("x", value=42)
        cache = MemoCache()

        executor = BatchExecutor(g, cache)
        result = executor.apply({"x": 42})

        assert result.success
        assert "x" in result.skipped

    def test_apply_validated_rejects_invalid(self):
        """Validation failure prevents any changes."""
        g = ComputeGraph()
        g.add_input("x", value=5)
        cache = MemoCache()

        executor = BatchExecutor(g, cache)
        result = executor.apply_validated(
            {"x": -1},
            validator=lambda nid, v: v >= 0,
        )

        assert not result.success
        assert g.get_node("x").value == 5  # Unchanged

    def test_dry_run_does_not_modify_state(self):
        """Dry run shows what would happen without side effects."""
        g = ComputeGraph()
        g.add_input("x", value=1)
        g.add_compute("y", func=lambda d: d["x"] * 10, dependencies=["x"])
        cache = MemoCache()

        from dagflow.propagation.eager import EagerPropagator
        EagerPropagator(g, cache).propagate(["x"])

        executor = BatchExecutor(g, cache)
        result = executor.dry_run({"x": 99})

        # State should be unchanged after dry run
        assert g.get_node("x").value == 1
        assert cache.get("y") == 10

    def test_apply_sequence_stops_on_failure(self):
        """Sequence stops at first failing batch."""
        g = ComputeGraph()
        g.add_input("x", value=0)
        cache = MemoCache()

        executor = BatchExecutor(g, cache)
        results = executor.apply_sequence([
            {"x": 10},
            {"nonexistent": 5},  # This will fail
            {"x": 20},           # Should not be reached
        ])

        assert len(results) == 2
        assert results[0].success
        assert not results[1].success
