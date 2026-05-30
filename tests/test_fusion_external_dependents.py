"""Regression tests for optimizer fusion around shared intermediates."""

from __future__ import annotations

from typing import Dict, Tuple

import pytest

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode
from dagflow.memo.cache import MemoCache
from dagflow.optimizer.fusion import NodeFuser
from dagflow.propagation.eager import EagerPropagator
from dagflow.scheduler.topo import TopologicalScheduler


def _branched_graph(extra_side: bool = False) -> ComputeGraph:
    graph = ComputeGraph()
    graph.add_input("x", value=2)
    graph.add_compute("a", lambda deps: deps["x"] * 3, ["x"])
    graph.add_compute("b", lambda deps: deps["a"] + 1, ["a"])
    graph.add_compute("c", lambda deps: deps["b"] * 2, ["b"])
    graph.add_compute("side", lambda deps: deps["x"] + deps["b"], ["x", "b"])
    if extra_side:
        graph.add_compute("side2", lambda deps: deps["b"] - deps["x"], ["b", "x"])
        graph.add_compute(
            "out",
            lambda deps: deps["c"] + deps["side"] + deps["side2"],
            ["c", "side", "side2"],
        )
    else:
        graph.add_compute("out", lambda deps: deps["c"] + deps["side"], ["c", "side"])
    return graph


def _simple_chain_graph() -> ComputeGraph:
    graph = ComputeGraph()
    graph.add_input("x", value=2)
    graph.add_compute("a", lambda deps: deps["x"] + 1, ["x"])
    graph.add_compute("b", lambda deps: deps["a"] * 4, ["a"])
    graph.add_compute("c", lambda deps: deps["b"] - 3, ["b"])
    return graph


def _run(graph: ComputeGraph, changed_to: int) -> Tuple[Dict[str, int], MemoCache]:
    cache = MemoCache()
    propagator = EagerPropagator(graph, cache)
    propagator.propagate(["x"])
    input_node = graph.get_node("x")
    input_node.set(changed_to)
    result = propagator.propagate(["x"])
    values = {
        node_id: cache.get(node_id)
        for node_id in graph.all_node_ids
        if isinstance(graph.get_node(node_id), ComputeNode)
    }
    values.update(result)
    return values, cache


def _optimize(graph: ComputeGraph):
    fuser = NodeFuser(graph)
    candidates = fuser.find_candidates()
    return fuser.apply(candidates)


def test_optimized_graph_matches_unoptimized_side_branch():
    reference = _branched_graph()
    optimized = _branched_graph()
    expected, _ = _run(reference, 4)

    _optimize(optimized)
    actual, _ = _run(optimized, 4)

    assert actual["out"] == expected["out"]
    assert actual["side"] == expected["side"]


def test_external_dependent_updates_after_second_input_change():
    optimized = _branched_graph()
    _optimize(optimized)

    first, _ = _run(optimized, 4)
    input_node = optimized.get_node("x")
    input_node.set(5)
    cache = MemoCache()
    propagator = EagerPropagator(optimized, cache)
    propagator.propagate(["x"])

    assert cache.get("side") == 21
    assert cache.get("out") == 53
    assert first["out"] == 43


def test_two_external_dependents_keep_intermediate_value():
    reference = _branched_graph(extra_side=True)
    optimized = _branched_graph(extra_side=True)
    expected, _ = _run(reference, 4)

    _optimize(optimized)
    actual, _ = _run(optimized, 4)

    assert actual["side"] == expected["side"]
    assert actual["side2"] == expected["side2"]
    assert actual["out"] == expected["out"]


def test_downstream_join_uses_recomputed_side_branch():
    optimized = _branched_graph()
    _optimize(optimized)

    actual, cache = _run(optimized, 6)

    assert cache.get("side") == 25
    assert cache.get("out") == 63
    assert actual["out"] == 63


def test_scheduler_after_fusion_has_no_removed_dependencies():
    reference = _branched_graph()
    expected, _ = _run(reference, 4)

    optimized = _branched_graph()
    _optimize(optimized)

    compute_nodes = {
        node_id
        for node_id in optimized.all_node_ids
        if isinstance(optimized.get_node(node_id), ComputeNode)
    }
    for node_id in compute_nodes:
        node = optimized.get_node(node_id)
        assert set(node.dependencies) <= set(optimized.all_node_ids)

    order = TopologicalScheduler(optimized).schedule(compute_nodes)
    assert set(order) == compute_nodes

    # Executing the scheduled optimized graph must reproduce the observable
    # values of the unoptimized graph. A fused chain that dropped an
    # externally observed intermediate would either raise on a missing
    # dependency or yield a diverging value here.
    actual, _ = _run(optimized, 4)
    assert actual["out"] == expected["out"]
    assert actual["side"] == expected["side"]


def test_simple_chain_still_fuses_and_updates():
    graph = _simple_chain_graph()
    before_count = graph.node_count
    result = _optimize(graph)
    values, cache = _run(graph, 4)

    assert result.fused_count >= 1
    assert graph.node_count < before_count
    assert cache.get("fused_a_b_c") == 17 or values.get("c") == 17


def test_dry_run_does_not_modify_branched_graph():
    graph = _branched_graph()
    before_ids = list(graph.all_node_ids)
    result = NodeFuser(graph).apply(dry_run=True)

    assert result.candidates_found >= 1
    assert graph.all_node_ids == before_ids
