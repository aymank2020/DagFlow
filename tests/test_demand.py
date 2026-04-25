"""Demand engine tests."""
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import NodeState
from dagflow.memo.cache import MemoCache
from dagflow.query.demand import DemandEngine
from dagflow.propagation.invalidator import Invalidator

def test_demand_input():
    g = ComputeGraph()
    g.add_input("x", value=42)
    cache = MemoCache()
    engine = DemandEngine(g, cache)
    assert engine.demand("x") == 42

def test_demand_computes():
    g = ComputeGraph()
    g.add_input("x", value=5)
    g.add_compute("y", func=lambda d: d["x"] + 1, dependencies=["x"])
    cache = MemoCache()
    engine = DemandEngine(g, cache)
    assert engine.demand("y") == 6

def test_demand_caches():
    g = ComputeGraph()
    g.add_input("x", value=3)
    call_count = [0]
    def tracked(d):
        call_count[0] += 1
        return d["x"] * 2
    g.add_compute("y", func=tracked, dependencies=["x"])
    cache = MemoCache()
    engine = DemandEngine(g, cache)
    engine.demand("y")
    engine.demand("y")
    assert call_count[0] == 1

def test_demand_after_invalidation():
    g = ComputeGraph()
    inp = g.add_input("x", value=2)
    g.add_compute("y", func=lambda d: d["x"] ** 3, dependencies=["x"])
    cache = MemoCache()
    engine = DemandEngine(g, cache)
    assert engine.demand("y") == 8
    inp.set(3)
    Invalidator(g).invalidate_from(["x"])
    assert engine.demand("y") == 27
