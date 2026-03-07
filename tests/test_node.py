"""Tests for node types."""
from dagflow.core.node import InputNode, ComputeNode, NodeState

def test_input_generation_bumps():
    n = InputNode(node_id="x", value=1)
    assert n.generation == 0
    n.set(2)
    assert n.generation == 1

def test_input_same_value_no_bump():
    n = InputNode(node_id="x", value=5)
    n.set(5)
    assert n.generation == 0

def test_compute_starts_dirty():
    n = ComputeNode(node_id="c")
    assert n.state == NodeState.DIRTY
