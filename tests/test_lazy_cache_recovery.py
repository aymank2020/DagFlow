"""Lazy public API refreshes changed ancestors and recovers evicted entries."""
import pytest

from dagflow.core.graph import ComputeGraph
from dagflow.memo.cache import MemoCache
from dagflow.query.demand import DemandEngine


@pytest.mark.parametrize("multiple", [False, True])
def test_direct_input_change_refreshes_entire_chain(multiple):
    graph = ComputeGraph()
    source = graph.add_input("x", value=2)
    graph.add_compute("double", lambda d: d["x"] * 2, ["x"])
    graph.add_compute("total", lambda d: d["double"] + 1, ["double"])
    engine = DemandEngine(graph, MemoCache())
    demand = lambda: engine.demand_multiple(["total"])["total"] if multiple else engine.demand("total")
    assert demand() == 5
    source.set(3)
    assert demand() == 7


@pytest.mark.parametrize("clear", [False, True])
def test_eviction_recovers_clean_nodes(clear):
    graph = ComputeGraph()
    graph.add_input("x", value=2)
    graph.add_compute("double", lambda d: d["x"] * 2, ["x"])
    graph.add_compute("total", lambda d: d["double"] + 1, ["double"])
    cache = MemoCache()
    engine = DemandEngine(graph, cache)
    assert engine.demand("total") == 5
    if clear:
        cache.clear()
    else:
        cache.invalidate("double")
    assert engine.demand("total") == 5
    assert cache.get("double") == 4


def test_none_results_are_cached_and_unaffected_nodes_are_not_recomputed():
    calls = []
    graph = ComputeGraph()
    graph.add_input("x", value=2)
    graph.add_compute("optional", lambda d: calls.append(d["x"]), ["x"])
    cache = MemoCache()
    engine = DemandEngine(graph, cache)
    assert engine.demand("optional") is None
    assert engine.demand_multiple(["optional"]) == {"optional": None}
    assert calls == [2]


def test_dependency_snapshots_are_immutable_from_callers():
    cache = MemoCache()
    generations = {"x": 1}
    cache.store("result", 42, generations)
    generations["x"] = 2
    snapshot = cache.get_dep_snapshot("result")
    snapshot["x"] = 3
    assert cache.is_valid("result", {"x": 1})
    assert not cache.is_valid("result", {"x": 0})
    assert not cache.is_valid("result", {"x": 1, "y": 1})
