"""Basic graph tests."""
import pytest
from dagflow.core.graph import ComputeGraph

def test_add_input():
    g = ComputeGraph()
    n = g.add_input("x", 10)
    assert n.value == 10
    assert g.node_count == 1

def test_add_compute_links():
    g = ComputeGraph()
    g.add_input("a")
    g.add_compute("b", func=lambda d: d["a"], dependencies=["a"])
    assert "b" in g.get_node("a").dependents

def test_duplicate_raises():
    g = ComputeGraph()
    g.add_input("x")
    with pytest.raises(ValueError):
        g.add_input("x")

def test_missing_dep_raises():
    g = ComputeGraph()
    with pytest.raises(ValueError):
        g.add_compute("c", func=lambda d: None, dependencies=["nope"])
