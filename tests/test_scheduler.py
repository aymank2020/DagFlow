"""Qualitative tests for TopologicalScheduler — ordering invariants.

Tests verify that the schedule respects dependency ordering (a dependency
always appears before its dependent) without checking exact order.
"""

import pytest
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode
from dagflow.scheduler.topo import TopologicalScheduler


class TestSchedulerOrdering:
    """Tests that verify topological ordering properties."""

    def _build_chain(self):
        """a -> b -> c -> d (linear chain)."""
        g = ComputeGraph()
        g.add_input("a")
        g.add_compute("b", func=lambda d: d["a"] * 2, dependencies=["a"])
        g.add_compute("c", func=lambda d: d["b"] + 1, dependencies=["b"])
        g.add_compute("d", func=lambda d: d["c"] ** 2, dependencies=["c"])
        return g

    def _build_diamond(self):
        """a -> b, a -> c, b+c -> d."""
        g = ComputeGraph()
        g.add_input("a")
        g.add_compute("b", func=lambda d: d["a"] + 1, dependencies=["a"])
        g.add_compute("c", func=lambda d: d["a"] * 2, dependencies=["a"])
        g.add_compute("d", func=lambda d: d["b"] + d["c"], dependencies=["b", "c"])
        return g

    def test_chain_respects_order(self):
        g = self._build_chain()
        sched = TopologicalScheduler(g)
        order = sched.schedule({"b", "c", "d"})

        idx_b = order.index("b")
        idx_c = order.index("c")
        idx_d = order.index("d")
        assert idx_b < idx_c < idx_d

    def test_diamond_d_after_both_parents(self):
        g = self._build_diamond()
        sched = TopologicalScheduler(g)
        order = sched.schedule({"b", "c", "d"})

        idx_d = order.index("d")
        idx_b = order.index("b")
        idx_c = order.index("c")
        assert idx_b < idx_d
        assert idx_c < idx_d

    def test_empty_dirty_set_returns_empty(self):
        g = self._build_chain()
        sched = TopologicalScheduler(g)
        assert sched.schedule(set()) == []

    def test_single_node_schedule(self):
        g = self._build_chain()
        sched = TopologicalScheduler(g)
        order = sched.schedule({"c"})
        assert order == ["c"]

    def test_full_schedule_includes_all_compute(self):
        g = self._build_diamond()
        sched = TopologicalScheduler(g)
        order = sched.full_schedule()
        assert set(order) == {"b", "c", "d"}

    def test_priority_tiebreaking(self):
        """Among same-level nodes, lower priority executes first."""
        g = ComputeGraph()
        g.add_input("x")
        g.add_compute("high", func=lambda d: None, dependencies=["x"], priority=10)
        g.add_compute("low", func=lambda d: None, dependencies=["x"], priority=1)

        sched = TopologicalScheduler(g)
        order = sched.schedule({"high", "low"})
        assert order.index("low") < order.index("high")
