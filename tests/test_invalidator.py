"""Invalidator tests."""
from dagflow.core.graph import ComputeGraph
from dagflow.propagation.invalidator import Invalidator

def test_direct_dependent_marked():
    g = ComputeGraph()
    g.add_input("x")
    g.add_compute("y", func=lambda d: d["x"] + 1, dependencies=["x"])
    inv = Invalidator(g)
    dirty = inv.invalidate_from(["x"])
    assert "y" in dirty

def test_transitive_propagation():
    g = ComputeGraph()
    g.add_input("a")
    g.add_compute("b", func=lambda d: None, dependencies=["a"])
    g.add_compute("c", func=lambda d: None, dependencies=["b"])
    inv = Invalidator(g)
    dirty = inv.invalidate_from(["a"])
    assert "b" in dirty and "c" in dirty

def test_unrelated_not_affected():
    g = ComputeGraph()
    g.add_input("x")
    g.add_input("y")
    g.add_compute("cx", func=lambda d: None, dependencies=["x"])
    g.add_compute("cy", func=lambda d: None, dependencies=["y"])
    inv = Invalidator(g)
    dirty = inv.invalidate_from(["x"])
    assert "cx" in dirty
    assert "cy" not in dirty
