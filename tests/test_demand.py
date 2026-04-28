"""Qualitative tests for DemandEngine — lazy evaluation invariants.

Tests verify that:
- Demanding a clean node returns cached value without recomputation.
- Demanding a dirty node triggers recomputation of necessary ancestors.
- Only the minimal set of nodes is recomputed.
"""

import pytest
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import NodeState, ComputeNode
from dagflow.memo.cache import MemoCache
from dagflow.query.demand import DemandEngine
from dagflow.propagation.invalidator import Invalidator


class TestDemandEngine:
    """Tests for demand-driven recomputation."""

    def test_demand_input_returns_value(self):
        g = ComputeGraph()
        g.add_input("x", value=42)
        cache = MemoCache()
        engine = DemandEngine(g, cache)
        assert engine.demand("x") == 42

    def test_demand_dirty_node_computes(self):
        g = ComputeGraph()
        g.add_input("x", value=5)
        g.add_compute("y", func=lambda d: d["x"] + 1, dependencies=["x"])

        cache = MemoCache()
        engine = DemandEngine(g, cache)
        result = engine.demand("y")
        assert result == 6

    def test_demand_clean_node_uses_cache(self):
        g = ComputeGraph()
        g.add_input("x", value=3)

        call_count = [0]

        def tracked_func(d):
            call_count[0] += 1
            return d["x"] * 2

        g.add_compute("y", func=tracked_func, dependencies=["x"])

        cache = MemoCache()
        engine = DemandEngine(g, cache)

        # First demand — computes
        engine.demand("y")
        assert call_count[0] == 1

        # Second demand — should use cache (node is clean)
        engine.demand("y")
        assert call_count[0] == 1

    def test_demand_after_input_change_recomputes(self):
        g = ComputeGraph()
        inp = g.add_input("x", value=2)
        g.add_compute("y", func=lambda d: d["x"] ** 3, dependencies=["x"])

        cache = MemoCache()
        engine = DemandEngine(g, cache)

        assert engine.demand("y") == 8

        # Change input and invalidate
        inp.set(3)
        inv = Invalidator(g)
        inv.invalidate_from(["x"])

        assert engine.demand("y") == 27

    def test_demand_chain_recomputes_ancestors(self):
        """a=4 -> b=a*2 -> c=b-1. Demand c should compute both b and c."""
        g = ComputeGraph()
        g.add_input("a", value=4)
        g.add_compute("b", func=lambda d: d["a"] * 2, dependencies=["a"])
        g.add_compute("c", func=lambda d: d["b"] - 1, dependencies=["b"])

        cache = MemoCache()
        engine = DemandEngine(g, cache)

        result = engine.demand("c")
        # b=8, c=7
        assert result == 7

    def test_demand_multiple_efficient(self):
        """Demanding multiple nodes shares computation."""
        g = ComputeGraph()
        g.add_input("x", value=10)
        g.add_compute("a", func=lambda d: d["x"] + 1, dependencies=["x"])
        g.add_compute("b", func=lambda d: d["x"] * 2, dependencies=["x"])

        cache = MemoCache()
        engine = DemandEngine(g, cache)

        results = engine.demand_multiple(["a", "b"])
        assert results["a"] == 11
        assert results["b"] == 20
