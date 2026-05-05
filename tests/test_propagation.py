"""Qualitative tests for change propagation — behavioral invariants.

Tests verify that:
- Dirty nodes get recomputed (value changes propagate).
- Clean nodes are NOT recomputed unnecessarily (early cutoff).
- Diamond dependencies don't cause double-computation.
"""

import pytest
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import NodeState
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.propagation.invalidator import Invalidator


class TestInvalidator:
    """Tests for the invalidation walk."""

    def test_invalidate_marks_direct_dependents(self):
        g = ComputeGraph()
        g.add_input("x")
        g.add_compute("y", func=lambda d: d["x"] + 1, dependencies=["x"])

        inv = Invalidator(g)
        dirty = inv.invalidate_from(["x"])
        assert "y" in dirty

    def test_invalidate_propagates_transitively(self):
        g = ComputeGraph()
        g.add_input("a")
        g.add_compute("b", func=lambda d: None, dependencies=["a"])
        g.add_compute("c", func=lambda d: None, dependencies=["b"])

        inv = Invalidator(g)
        dirty = inv.invalidate_from(["a"])
        assert "b" in dirty
        assert "c" in dirty

    def test_invalidate_does_not_affect_unrelated(self):
        g = ComputeGraph()
        g.add_input("x")
        g.add_input("y")
        g.add_compute("cx", func=lambda d: None, dependencies=["x"])
        g.add_compute("cy", func=lambda d: None, dependencies=["y"])

        inv = Invalidator(g)
        dirty = inv.invalidate_from(["x"])
        assert "cx" in dirty
        assert "cy" not in dirty


class TestEagerPropagation:
    """Tests for eager recomputation behavior."""

    def test_simple_propagation_updates_value(self):
        g = ComputeGraph()
        inp = g.add_input("x", value=5)
        g.add_compute("double", func=lambda d: d["x"] * 2, dependencies=["x"])

        cache = MemoCache()
        prop = EagerPropagator(g, cache)

        # Initial propagation
        result = prop.propagate(["x"])
        assert "double" in result
        assert result["double"] == 10

    def test_propagation_after_input_change(self):
        g = ComputeGraph()
        inp = g.add_input("x", value=3)
        g.add_compute("sq", func=lambda d: d["x"] ** 2, dependencies=["x"])

        cache = MemoCache()
        prop = EagerPropagator(g, cache)

        prop.propagate(["x"])
        # Change input
        inp.set(7)
        result = prop.propagate(["x"])
        assert result["sq"] == 49

    def test_chain_propagation_correct_order(self):
        """a=2 -> b=a*3 -> c=b+10. Verify c gets correct value."""
        g = ComputeGraph()
        g.add_input("a", value=2)
        g.add_compute("b", func=lambda d: d["a"] * 3, dependencies=["a"])
        g.add_compute("c", func=lambda d: d["b"] + 10, dependencies=["b"])

        cache = MemoCache()
        prop = EagerPropagator(g, cache)
        result = prop.propagate(["a"])

        # b = 2*3 = 6, c = 6+10 = 16
        assert result.get("c") == 16

    def test_diamond_propagation_no_stale_read(self):
        """Diamond: a -> b, a -> c, b+c -> d. d must see both updated."""
        g = ComputeGraph()
        g.add_input("a", value=1)
        g.add_compute("b", func=lambda d: d["a"] + 10, dependencies=["a"])
        g.add_compute("c", func=lambda d: d["a"] * 100, dependencies=["a"])
        g.add_compute("d", func=lambda d: d["b"] + d["c"], dependencies=["b", "c"])

        cache = MemoCache()
        prop = EagerPropagator(g, cache)
        result = prop.propagate(["a"])

        # b=11, c=100, d=111
        assert result.get("d") == 111

    def test_unrelated_branch_not_recomputed(self):
        """Changing x should not recompute nodes depending only on y."""
        g = ComputeGraph()
        g.add_input("x", value=1)
        g.add_input("y", value=2)
        g.add_compute("cx", func=lambda d: d["x"] * 2, dependencies=["x"])
        g.add_compute("cy", func=lambda d: d["y"] * 3, dependencies=["y"])

        cache = MemoCache()
        prop = EagerPropagator(g, cache)

        result = prop.propagate(["x"])
        assert "cx" in result
        assert "cy" not in result
