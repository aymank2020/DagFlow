"""Eager propagation tests."""
from dagflow.core.graph import ComputeGraph
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator

def test_simple_propagation():
    g = ComputeGraph()
    g.add_input("x", value=5)
    g.add_compute("double", func=lambda d: d["x"] * 2, dependencies=["x"])
    cache = MemoCache()
    prop = EagerPropagator(g, cache)
    result = prop.propagate(["x"])
    assert result["double"] == 10

def test_chain():
    g = ComputeGraph()
    g.add_input("a", value=2)
    g.add_compute("b", func=lambda d: d["a"] * 3, dependencies=["a"])
    g.add_compute("c", func=lambda d: d["b"] + 10, dependencies=["b"])
    cache = MemoCache()
    prop = EagerPropagator(g, cache)
    result = prop.propagate(["a"])
    assert result["c"] == 16

def test_input_change():
    g = ComputeGraph()
    inp = g.add_input("x", value=3)
    g.add_compute("sq", func=lambda d: d["x"] ** 2, dependencies=["x"])
    cache = MemoCache()
    prop = EagerPropagator(g, cache)
    prop.propagate(["x"])
    inp.set(7)
    result = prop.propagate(["x"])
    assert result["sq"] == 49

def test_diamond_no_stale():
    """Diamond: a -> b, a -> c, b+c -> d."""
    g = ComputeGraph()
    g.add_input("a", value=1)
    g.add_compute("b", func=lambda d: d["a"] + 10, dependencies=["a"])
    g.add_compute("c", func=lambda d: d["a"] * 100, dependencies=["a"])
    g.add_compute("d", func=lambda d: d["b"] + d["c"], dependencies=["b", "c"])
    cache = MemoCache()
    prop = EagerPropagator(g, cache)
    result = prop.propagate(["a"])
    assert result["d"] == 111

def test_unrelated_not_recomputed():
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
