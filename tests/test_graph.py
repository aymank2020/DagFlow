"""Qualitative tests for ComputeGraph — structural properties.

These tests verify graph invariants (acyclicity, connectivity, node existence)
rather than specific numerical outputs. They should pass regardless of
internal algorithm changes.
"""

import pytest
from dagflow.core.graph import ComputeGraph, CycleError
from dagflow.core.node import InputNode, ComputeNode


class TestGraphStructure:
    """Tests that verify structural DAG properties."""

    def test_add_input_creates_node(self):
        g = ComputeGraph()
        node = g.add_input("x", value=10)
        assert node.node_id == "x"
        assert node.value == 10
        assert g.node_count == 1

    def test_add_compute_links_dependencies(self):
        g = ComputeGraph()
        g.add_input("a")
        g.add_input("b")
        g.add_compute("c", func=lambda d: d["a"] + d["b"], dependencies=["a", "b"])

        assert "c" in g.get_dependents("a")
        assert "c" in g.get_dependents("b")
        assert set(g.get_dependencies("c")) == {"a", "b"}

    def test_cycle_detection_self_loop(self):
        g = ComputeGraph()
        g.add_input("x")
        with pytest.raises(CycleError):
            g.add_edge("x", "x")

    def test_cycle_detection_indirect(self):
        g = ComputeGraph()
        g.add_input("a")
        g.add_compute("b", func=lambda d: None, dependencies=["a"])
        g.add_compute("c", func=lambda d: None, dependencies=["b"])
        # c -> a would create a->b->c->a cycle
        with pytest.raises(CycleError):
            g.add_edge("c", "a")

    def test_duplicate_node_raises(self):
        g = ComputeGraph()
        g.add_input("x")
        with pytest.raises(ValueError):
            g.add_input("x")

    def test_missing_dependency_raises(self):
        g = ComputeGraph()
        with pytest.raises(ValueError):
            g.add_compute("c", func=lambda d: None, dependencies=["nonexistent"])

    def test_downstream_traversal_completeness(self):
        """All transitive dependents are reachable."""
        g = ComputeGraph()
        g.add_input("root")
        g.add_compute("mid", func=lambda d: None, dependencies=["root"])
        g.add_compute("leaf", func=lambda d: None, dependencies=["mid"])

        downstream = g.get_all_downstream("root")
        assert "mid" in downstream
        assert "leaf" in downstream

    def test_upstream_traversal_completeness(self):
        """All transitive dependencies are reachable."""
        g = ComputeGraph()
        g.add_input("a")
        g.add_input("b")
        g.add_compute("mid", func=lambda d: None, dependencies=["a"])
        g.add_compute("leaf", func=lambda d: None, dependencies=["mid", "b"])

        upstream = g.get_all_upstream("leaf")
        assert "mid" in upstream
        assert "a" in upstream
        assert "b" in upstream

    def test_diamond_dependency_no_duplicate(self):
        """Diamond shape: A -> B, A -> C, B -> D, C -> D."""
        g = ComputeGraph()
        g.add_input("a")
        g.add_compute("b", func=lambda d: None, dependencies=["a"])
        g.add_compute("c", func=lambda d: None, dependencies=["a"])
        g.add_compute("d", func=lambda d: None, dependencies=["b", "c"])

        # D should appear only once in downstream of A
        downstream = g.get_all_downstream("a")
        assert downstream.count("d") <= 1 or len(set(downstream)) == len(downstream)
