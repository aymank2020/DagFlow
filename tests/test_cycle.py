"""Cycle detection tests."""
import pytest
from dagflow.core.graph import ComputeGraph, CycleError

def test_self_loop():
    g = ComputeGraph()
    g.add_input("x")
    with pytest.raises(CycleError):
        g.add_edge("x", "x")

def test_indirect_cycle():
    g = ComputeGraph()
    g.add_input("a")
    g.add_compute("b", func=lambda d: None, dependencies=["a"])
    g.add_compute("c", func=lambda d: None, dependencies=["b"])
    with pytest.raises(CycleError):
        g.add_edge("c", "a")

def test_valid_edge_no_error():
    g = ComputeGraph()
    g.add_input("a")
    g.add_input("b")
    g.add_compute("c", func=lambda d: None, dependencies=["a"])
    # b -> c is fine (no cycle)
    g.add_edge("b", "c")
