"""Regression tests for transaction rollback with windowed transforms."""

from __future__ import annotations

import pytest

from dagflow.batch.transaction import Transaction
from dagflow.core.graph import ComputeGraph
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.transforms.window import WindowNode


def _prime_sum_window(value: int = 10):
    graph = ComputeGraph()
    source = graph.add_input("temperature", value=value)
    cache = MemoCache()
    windows = WindowNode(graph, window_size=3)

    def guarded_sum(values):
        if values[-1] == 99:
            raise RuntimeError("sensor spike rejected")
        return sum(values)

    windows.create_custom("rolling_total", "temperature", guarded_sum, size=3)
    EagerPropagator(graph, cache).propagate(["temperature"])
    return graph, source, cache, windows


def _failed_transaction(graph: ComputeGraph, cache: MemoCache, node_id: str, value: int) -> None:
    with pytest.raises(RuntimeError, match="sensor spike rejected"):
        with Transaction(graph, cache) as txn:
            txn.set_input(node_id, value)


def test_failed_transaction_restores_window_history():
    graph, _, cache, windows = _prime_sum_window(10)

    _failed_transaction(graph, cache, "temperature", 99)

    assert graph.get_node("temperature").value == 10
    assert cache.get("rolling_total") == 10
    assert windows.get_buffer("rolling_total").values == [10]


def test_later_update_after_failed_transaction_uses_original_window():
    graph, source, cache, windows = _prime_sum_window(10)
    _failed_transaction(graph, cache, "temperature", 99)

    source.set(20)
    EagerPropagator(graph, cache).propagate(["temperature"])

    assert cache.get("rolling_total") == 30
    assert windows.get_buffer("rolling_total").values == [10, 20]


def test_second_commit_after_rollback_uses_pre_transaction_history():
    graph, _, cache, windows = _prime_sum_window(10)
    _failed_transaction(graph, cache, "temperature", 99)

    with Transaction(graph, cache) as txn:
        txn.set_input("temperature", 20)

    assert cache.get("rolling_total") == 30
    assert windows.get_buffer("rolling_total").values == [10, 20]


def test_multiple_window_nodes_restore_independently():
    graph = ComputeGraph()
    source = graph.add_input("temperature", value=4)
    cache = MemoCache()
    windows = WindowNode(graph, window_size=3)
    windows.create_moving_sum("sum_window", "temperature", size=3)

    def guarded_range(values):
        if values[-1] == 99:
            raise RuntimeError("sensor spike rejected")
        return max(values) - min(values)

    windows.create_custom("zz_range_window", "temperature", guarded_range, size=3)
    EagerPropagator(graph, cache).propagate(["temperature"])

    _failed_transaction(graph, cache, "temperature", 99)

    assert source.value == 4
    assert cache.get("sum_window") == 4
    assert cache.get("zz_range_window") == 0
    assert windows.get_buffer("sum_window").values == [4]
    assert windows.get_buffer("zz_range_window").values == [4]


def test_unrelated_committed_window_history_survives_failed_transaction():
    graph = ComputeGraph()
    stable = graph.add_input("stable", value=5)
    volatile = graph.add_input("volatile", value=10)
    cache = MemoCache()
    windows = WindowNode(graph, window_size=3)
    windows.create_moving_sum("stable_sum", "stable", size=3)

    def guarded_sum(values):
        if values[-1] == 99:
            raise RuntimeError("sensor spike rejected")
        return sum(values)

    windows.create_custom("volatile_sum", "volatile", guarded_sum, size=3)
    prop = EagerPropagator(graph, cache)
    prop.propagate(["stable", "volatile"])
    stable.set(6)
    prop.propagate(["stable"])

    _failed_transaction(graph, cache, "volatile", 99)

    assert cache.get("stable_sum") == 11
    assert windows.get_buffer("stable_sum").values == [5, 6]
    assert volatile.value == 10
    assert cache.get("volatile_sum") == 10
    assert windows.get_buffer("volatile_sum").values == [10]


def test_committed_transaction_keeps_new_window_history():
    graph, _, cache, windows = _prime_sum_window(10)

    with Transaction(graph, cache) as txn:
        txn.set_input("temperature", 20)

    assert cache.get("rolling_total") == 30
    assert windows.get_buffer("rolling_total").values == [10, 20]


def test_plain_input_rollback_still_restores_cache():
    graph = ComputeGraph()
    source = graph.add_input("x", value=2)
    graph.add_compute("double", lambda deps: deps["x"] * 2, ["x"])
    cache = MemoCache()
    EagerPropagator(graph, cache).propagate(["x"])

    txn = Transaction(graph, cache)
    txn.begin()
    txn.set_input("x", 7)
    txn.rollback()

    assert source.value == 2
    assert cache.get("double") == 4
