"""Tests for persistence — serialization, snapshots, and replay.

Tests verify structural/behavioral invariants:
- Serialized graph can be deserialized to equivalent state.
- Snapshots capture values at a point in time.
- Replay reproduces the same sequence of state changes.
"""

import pytest
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode, ComputeNode, NodeState
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.persistence.serializer import GraphSerializer, SerializationError
from dagflow.persistence.snapshot import Snapshot, SnapshotStore
from dagflow.persistence.replay import ReplayLog


class TestSerializer:
    """Tests for graph serialization/deserialization."""

    def _make_graph_and_cache(self):
        """Helper: create a simple graph with computed values."""
        g = ComputeGraph()
        g.add_input("x", value=10)
        g.add_input("y", value=20)

        add_func = lambda d: d["x"] + d["y"]
        g.add_compute("sum", func=add_func, dependencies=["x", "y"])

        cache = MemoCache()
        prop = EagerPropagator(g, cache)
        prop.propagate(["x", "y"])

        return g, cache, add_func

    def test_serialize_deserialize_preserves_structure(self):
        """Deserialized graph has same nodes and edges."""
        g, cache, add_func = self._make_graph_and_cache()

        serializer = GraphSerializer()
        serializer.register_function("add", add_func)

        data = serializer.serialize(g, cache)
        new_graph, new_cache = serializer.deserialize(data)

        assert new_graph.node_count == g.node_count
        assert set(new_graph.all_node_ids) == set(g.all_node_ids)
        assert set(new_graph.get_dependencies("sum")) == {"x", "y"}

    def test_serialize_deserialize_preserves_values(self):
        """Deserialized graph has same input values and cached results."""
        g, cache, add_func = self._make_graph_and_cache()

        serializer = GraphSerializer()
        serializer.register_function("add", add_func)

        data = serializer.serialize(g, cache)
        new_graph, new_cache = serializer.deserialize(data)

        assert new_graph.get_node("x").value == 10
        assert new_graph.get_node("y").value == 20
        assert new_cache.get("sum") == 30

    def test_json_roundtrip(self):
        """JSON serialization roundtrip preserves data."""
        g, cache, add_func = self._make_graph_and_cache()

        serializer = GraphSerializer()
        serializer.register_function("add", add_func)

        json_str = serializer.to_json(g, cache)
        new_graph, new_cache = serializer.from_json(json_str)

        assert new_graph.node_count == 3
        assert new_cache.get("sum") == 30

    def test_unregistered_function_raises(self):
        """Serializing a node with unregistered function raises."""
        g = ComputeGraph()
        g.add_input("x", value=1)
        g.add_compute("y", func=lambda d: d["x"], dependencies=["x"])
        cache = MemoCache()

        serializer = GraphSerializer()
        with pytest.raises(SerializationError):
            serializer.serialize(g, cache)

    def test_missing_function_on_deserialize_raises(self):
        """Deserializing with missing function registry entry raises."""
        data = {
            "version": "1.0",
            "nodes": [
                {"type": "input", "id": "x", "value": 1, "generation": 0},
                {
                    "type": "compute",
                    "id": "y",
                    "function": "unknown_func",
                    "dependencies": ["x"],
                    "priority": 0,
                    "state": "DIRTY",
                    "cached_value": None,
                    "last_valid_generation": {},
                },
            ],
            "cache": {},
        }
        serializer = GraphSerializer()
        with pytest.raises(SerializationError):
            serializer.deserialize(data)


class TestSnapshot:
    """Tests for point-in-time snapshots."""

    def test_snapshot_captures_current_values(self):
        """Snapshot records values at capture time."""
        g = ComputeGraph()
        g.add_input("x", value=42)
        g.add_compute("y", func=lambda d: d["x"] * 2, dependencies=["x"])
        cache = MemoCache()
        EagerPropagator(g, cache).propagate(["x"])

        snap = Snapshot.capture("test", g, cache)

        assert snap.get_value("x") == 42
        assert snap.get_value("y") == 84

    def test_snapshot_is_independent_of_later_changes(self):
        """Snapshot values don't change when graph changes later."""
        g = ComputeGraph()
        inp = g.add_input("x", value=1)
        cache = MemoCache()

        snap = Snapshot.capture("before", g, cache)

        inp.set(999)
        assert snap.get_value("x") == 1  # Snapshot unchanged

    def test_snapshot_diff_detects_changes(self):
        """Diff between two snapshots shows what changed."""
        g = ComputeGraph()
        inp = g.add_input("x", value=10)
        g.add_compute("y", func=lambda d: d["x"] + 1, dependencies=["x"])
        cache = MemoCache()
        EagerPropagator(g, cache).propagate(["x"])

        snap1 = Snapshot.capture("before", g, cache)

        inp.set(20)
        EagerPropagator(g, cache).propagate(["x"])

        snap2 = Snapshot.capture("after", g, cache)

        diff = snap1.diff(snap2)
        assert "x" in diff  # Value changed
        assert "y" in diff  # Computed value changed

    def test_snapshot_store_manages_multiple(self):
        """SnapshotStore can save and retrieve multiple snapshots."""
        g = ComputeGraph()
        g.add_input("x", value=1)
        cache = MemoCache()

        store = SnapshotStore()
        store.save("v1", g, cache)

        g.get_node("x").set(2)
        store.save("v2", g, cache)

        assert store.count == 2
        assert store.get("v1").get_value("x") == 1
        assert store.get("v2").get_value("x") == 2

    def test_snapshot_store_duplicate_name_raises(self):
        """Saving with a duplicate name raises ValueError."""
        g = ComputeGraph()
        g.add_input("x")
        cache = MemoCache()

        store = SnapshotStore()
        store.save("snap", g, cache)
        with pytest.raises(ValueError):
            store.save("snap", g, cache)


class TestReplayLog:
    """Tests for change replay."""

    def test_record_and_replay_produces_same_state(self):
        """Replaying recorded changes produces equivalent state."""
        g = ComputeGraph()
        inp = g.add_input("x", value=0)
        g.add_compute("y", func=lambda d: d["x"] * 2, dependencies=["x"])
        cache = MemoCache()
        EagerPropagator(g, cache).propagate(["x"])

        log = ReplayLog(g)
        log.start_recording()
        log.record_change("x", 5)
        log.record_change("x", 10)
        log.stop_recording()

        assert log.entry_count == 2

        # Replay on a fresh graph
        g2 = ComputeGraph()
        g2.add_input("x", value=0)
        g2.add_compute("y", func=lambda d: d["x"] * 2, dependencies=["x"])
        cache2 = MemoCache()

        log.replay(g2, cache2)
        assert g2.get_node("x").value == 10
        assert cache2.get("y") == 20

    def test_replay_until_partial(self):
        """Replay up to a specific sequence number."""
        g = ComputeGraph()
        g.add_input("x", value=0)
        g.add_compute("y", func=lambda d: d["x"] + 1, dependencies=["x"])
        cache = MemoCache()

        log = ReplayLog(g)
        log.start_recording()
        log.record_change("x", 1)
        log.record_change("x", 2)
        log.record_change("x", 3)
        log.stop_recording()

        # Replay only first 2 changes
        g2 = ComputeGraph()
        g2.add_input("x", value=0)
        g2.add_compute("y", func=lambda d: d["x"] + 1, dependencies=["x"])
        cache2 = MemoCache()

        log.replay_until(g2, cache2, sequence_num=2)
        assert g2.get_node("x").value == 2

    def test_record_without_start_raises(self):
        """Recording without start_recording raises RuntimeError."""
        g = ComputeGraph()
        g.add_input("x", value=0)

        log = ReplayLog(g)
        with pytest.raises(RuntimeError):
            log.record_change("x", 5)

    def test_to_dict_and_from_dict_roundtrip(self):
        """Export/import preserves log entries."""
        g = ComputeGraph()
        g.add_input("x", value=0)

        log = ReplayLog(g)
        log.start_recording()
        log.record_change("x", 10)
        log.record_change("x", 20)
        log.stop_recording()

        exported = log.to_dict()
        restored = ReplayLog.from_dict(exported, g)

        assert restored.entry_count == 2
        assert restored.entries[0].new_value == 10
        assert restored.entries[1].new_value == 20
