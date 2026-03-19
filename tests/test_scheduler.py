"""Scheduler ordering tests."""
from dagflow.core.graph import ComputeGraph
from dagflow.scheduler.topo import TopologicalScheduler

def test_chain_order():
    g = ComputeGraph()
    g.add_input("a")
    g.add_compute("b", func=lambda d: d["a"] * 2, dependencies=["a"])
    g.add_compute("c", func=lambda d: d["b"] + 1, dependencies=["b"])
    sched = TopologicalScheduler(g)
    order = sched.schedule({"b", "c"})
    assert order.index("b") < order.index("c")

def test_empty_returns_empty():
    g = ComputeGraph()
    g.add_input("x")
    sched = TopologicalScheduler(g)
    assert sched.schedule(set()) == []
