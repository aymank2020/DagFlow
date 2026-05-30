"""Regression tests for vector-clock conflict windows."""

from __future__ import annotations

from dagflow.distributed.sync import SyncProtocol, SyncState


def _protocol_after_old_target_price_write() -> SyncProtocol:
    protocol = SyncProtocol()
    protocol.register_graph("source")
    protocol.register_graph("target")

    protocol.record_mutation("target", "price")
    protocol.synchronize("target", "source", {"price": 90})

    protocol.record_mutation("source", "price")
    first = protocol.synchronize("source", "target", {"price": 100})
    assert first.state == SyncState.IN_SYNC
    assert first.conflicts == set()
    return protocol


def test_old_target_write_before_last_sync_is_not_conflict():
    protocol = _protocol_after_old_target_price_write()

    protocol.record_mutation("target", "quantity")
    protocol.record_mutation("source", "price")
    result = protocol.synchronize("source", "target", {"price": 101})

    assert result.state == SyncState.IN_SYNC
    assert result.conflicts == set()
    assert result.synced_nodes == {"price"}


def test_unrelated_post_sync_target_write_does_not_reactivate_old_node_conflict():
    protocol = _protocol_after_old_target_price_write()

    protocol.record_mutation("target", "inventory_status")
    protocol.record_mutation("source", "price")
    result = protocol.synchronize("source", "target", {"price": 102})

    assert "price" not in result.conflicts
    assert "price" in result.synced_nodes


def test_multi_node_sync_filters_target_mutations_by_node_and_window():
    protocol = _protocol_after_old_target_price_write()

    protocol.record_mutation("target", "quantity")
    protocol.record_mutation("source", "price")
    protocol.record_mutation("source", "tax")
    result = protocol.synchronize("source", "target", {"price": 105, "tax": 8})

    assert result.conflicts == set()
    assert result.synced_nodes == {"price", "tax"}


def test_reverse_direction_uses_its_own_last_sync_window():
    protocol = SyncProtocol()
    protocol.register_graph("left")
    protocol.register_graph("right")

    protocol.record_mutation("left", "price")
    protocol.synchronize("left", "right", {"price": 75})

    protocol.record_mutation("right", "price")
    first = protocol.synchronize("right", "left", {"price": 80})
    assert first.state == SyncState.IN_SYNC
    assert first.conflicts == set()

    protocol.record_mutation("left", "quantity")
    protocol.record_mutation("right", "price")
    result = protocol.synchronize("right", "left", {"price": 81})

    assert result.state == SyncState.IN_SYNC
    assert result.conflicts == set()


def test_sequential_source_updates_do_not_reuse_stale_target_conflict():
    protocol = _protocol_after_old_target_price_write()

    protocol.record_mutation("target", "quantity")
    protocol.record_mutation("source", "price")
    first = protocol.synchronize("source", "target", {"price": 110})

    protocol.record_mutation("source", "price")
    second = protocol.synchronize("source", "target", {"price": 111})

    assert first.conflicts == set()
    assert second.conflicts == set()
    assert second.synced_nodes == {"price"}


def test_real_concurrent_target_write_to_same_node_still_conflicts():
    protocol = _protocol_after_old_target_price_write()

    protocol.record_mutation("target", "price")
    protocol.record_mutation("source", "price")
    result = protocol.synchronize("source", "target", {"price": 120})

    assert result.state == SyncState.DIVERGED
    assert result.conflicts == {"price"}
    assert "price" not in result.synced_nodes


def test_unrelated_target_mutation_count_remains_available_after_sync():
    protocol = _protocol_after_old_target_price_write()

    protocol.record_mutation("target", "quantity")
    protocol.record_mutation("source", "price")
    protocol.synchronize("source", "target", {"price": 130})

    assert protocol.get_mutation_count("target") == 2


def test_check_sync_still_reports_in_sync_after_non_conflicting_sync():
    protocol = _protocol_after_old_target_price_write()

    protocol.record_mutation("target", "quantity")
    protocol.record_mutation("source", "price")
    protocol.synchronize("source", "target", {"price": 140})

    assert protocol.check_sync("source", "target") == SyncState.IN_SYNC
