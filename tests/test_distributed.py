"""Tests for the distributed multi-graph coordination module.

Tests cover:
- Federation: graph registration, cross-graph edges, propagation
- Sync protocol: vector clocks, synchronization state
- Conflict resolution: policies, merge functions, history
"""

from __future__ import annotations

from typing import Any

import pytest

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode
from dagflow.memo.cache import MemoCache
from dagflow.distributed.federation import (
    CrossGraphEdge,
    FederatedGraph,
    FederationError,
    GraphEndpoint,
)
from dagflow.distributed.sync import (
    SyncBarrier,
    SyncProtocol,
    SyncState,
    VectorClock,
)
from dagflow.distributed.conflict import (
    ConflictPolicy,
    ConflictResolver,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def pricing_graph() -> tuple:
    """Create a pricing graph with cache."""
    g = ComputeGraph()
    g.add_input("base_price", value=100)
    g.add_input("tax_rate", value=0.1)
    g.add_compute("total", lambda d: d["base_price"] * (1 + d["tax_rate"]), ["base_price", "tax_rate"])
    return g, MemoCache()


@pytest.fixture
def inventory_graph() -> tuple:
    """Create an inventory graph with cache."""
    g = ComputeGraph()
    g.add_input("price_input", value=0)
    g.add_input("quantity", value=50)
    g.add_compute("value", lambda d: d["price_input"] * d["quantity"], ["price_input", "quantity"])
    return g, MemoCache()


@pytest.fixture
def federation(pricing_graph, inventory_graph) -> FederatedGraph:
    """Create a federation with two graphs."""
    fed = FederatedGraph()
    pg, pc = pricing_graph
    ig, ic = inventory_graph
    fed.register("pricing", pg, pc)
    fed.register("inventory", ig, ic)
    return fed


# ─── Federation Tests ──────────────────────────────────────────────────


class TestFederatedGraph:
    """Tests for FederatedGraph coordination."""

    def test_register_graph(self, pricing_graph) -> None:
        """Registering a graph creates an endpoint."""
        fed = FederatedGraph()
        g, c = pricing_graph
        endpoint = fed.register("pricing", g, c)

        assert endpoint.name == "pricing"
        assert fed.graph_count == 1

    def test_register_duplicate_raises(self, pricing_graph) -> None:
        """Registering same name twice raises FederationError."""
        fed = FederatedGraph()
        g, c = pricing_graph
        fed.register("pricing", g, c)

        with pytest.raises(FederationError):
            fed.register("pricing", g, c)

    def test_unregister_graph(self, federation: FederatedGraph) -> None:
        """Unregistering removes the graph and its edges."""
        assert federation.graph_count == 2
        result = federation.unregister("pricing")
        assert result is True
        assert federation.graph_count == 1

    def test_add_cross_edge(self, federation: FederatedGraph) -> None:
        """Cross-edge connects nodes between graphs."""
        edge = federation.add_cross_edge(
            "pricing", "total", "inventory", "price_input"
        )
        assert edge.source_graph == "pricing"
        assert edge.target_graph == "inventory"
        assert federation.cross_edge_count == 1

    def test_cross_edge_requires_input_target(
        self, federation: FederatedGraph
    ) -> None:
        """Target of cross-edge must be an InputNode."""
        with pytest.raises(FederationError, match="InputNode"):
            federation.add_cross_edge(
                "pricing", "base_price", "inventory", "value"
            )

    def test_cross_edge_same_graph_raises(
        self, federation: FederatedGraph
    ) -> None:
        """Cross-edge within same graph raises error."""
        with pytest.raises(FederationError, match="same graph"):
            federation.add_cross_edge(
                "pricing", "base_price", "pricing", "tax_rate"
            )

    def test_propagate_within_graph(self, federation: FederatedGraph, pricing_graph) -> None:
        """Propagation within a single graph works."""
        pg, pc = pricing_graph
        # Set input and propagate
        node = pg.get_node("base_price")
        assert isinstance(node, InputNode)
        node.set(200)

        results = federation.propagate("pricing", ["base_price"])
        assert "pricing" in results
        assert "total" in results["pricing"]

    def test_propagate_across_graphs(
        self, federation: FederatedGraph, pricing_graph, inventory_graph
    ) -> None:
        """Changes propagate across cross-graph edges."""
        federation.add_cross_edge(
            "pricing", "total", "inventory", "price_input"
        )

        pg, pc = pricing_graph
        node = pg.get_node("base_price")
        assert isinstance(node, InputNode)
        node.set(200)

        results = federation.propagate("pricing", ["base_price"])

        # Should propagate to inventory graph
        assert "pricing" in results
        assert "inventory" in results

    def test_cross_edge_with_transform(
        self, federation: FederatedGraph, pricing_graph
    ) -> None:
        """Transform function is applied to cross-edge values."""
        federation.add_cross_edge(
            "pricing", "total", "inventory", "price_input",
            transform=lambda v: v * 0.9,  # 10% discount
        )

        pg, pc = pricing_graph
        node = pg.get_node("base_price")
        assert isinstance(node, InputNode)
        node.set(100)

        federation.propagate("pricing", ["base_price"])

        # Check that inventory received the transformed value
        inv_endpoint = federation.get_endpoint("inventory")
        price_node = inv_endpoint.graph.get_node("price_input")
        assert isinstance(price_node, InputNode)
        # Value should be total * 0.9 = 110 * 0.9 = 99.0
        assert price_node.value == pytest.approx(99.0)

    def test_get_cross_edges_from(self, federation: FederatedGraph) -> None:
        """Query cross-edges originating from a graph."""
        federation.add_cross_edge("pricing", "total", "inventory", "price_input")
        edges = federation.get_cross_edges_from("pricing")
        assert len(edges) == 1
        assert edges[0].source_node == "total"


# ─── Sync Protocol Tests ──────────────────────────────────────────────


class TestSyncProtocol:
    """Tests for the synchronization protocol."""

    def test_vector_clock_increment(self) -> None:
        """Incrementing a clock advances its value."""
        clock = VectorClock()
        val = clock.increment("graph_a")
        assert val == 1
        val = clock.increment("graph_a")
        assert val == 2

    def test_vector_clock_dominates(self) -> None:
        """Clock A dominates B if all components >= and at least one >."""
        a = VectorClock(clocks={"x": 2, "y": 1})
        b = VectorClock(clocks={"x": 1, "y": 1})
        assert a.dominates(b)
        assert not b.dominates(a)

    def test_vector_clock_concurrent(self) -> None:
        """Concurrent clocks: neither dominates the other."""
        a = VectorClock(clocks={"x": 2, "y": 1})
        b = VectorClock(clocks={"x": 1, "y": 2})
        assert a.concurrent_with(b)
        assert b.concurrent_with(a)

    def test_vector_clock_merge(self) -> None:
        """Merge takes component-wise maximum."""
        a = VectorClock(clocks={"x": 2, "y": 1})
        b = VectorClock(clocks={"x": 1, "y": 3})
        a.merge(b)
        assert a.get("x") == 2
        assert a.get("y") == 3

    def test_register_and_record_mutation(self) -> None:
        """Recording mutations increments the clock."""
        protocol = SyncProtocol()
        protocol.register_graph("pricing")

        val = protocol.record_mutation("pricing", "base_price")
        assert val == 1
        val = protocol.record_mutation("pricing", "tax_rate")
        assert val == 2

    def test_check_sync_unknown(self) -> None:
        """Unknown graphs return UNKNOWN state."""
        protocol = SyncProtocol()
        state = protocol.check_sync("a", "b")
        assert state == SyncState.UNKNOWN

    def test_check_sync_stale(self) -> None:
        """Unsynced graphs are STALE."""
        protocol = SyncProtocol()
        protocol.register_graph("a")
        protocol.register_graph("b")
        protocol.record_mutation("a")

        state = protocol.check_sync("a", "b")
        assert state == SyncState.STALE

    def test_synchronize_updates_state(self) -> None:
        """Synchronization moves state to IN_SYNC."""
        protocol = SyncProtocol()
        protocol.register_graph("a")
        protocol.register_graph("b")
        protocol.record_mutation("a", "node1")

        result = protocol.synchronize("a", "b", {"node1": 42})
        assert result.state == SyncState.IN_SYNC
        assert "node1" in result.synced_nodes

    def test_barrier_completion(self) -> None:
        """Barrier completes when all participants arrive."""
        protocol = SyncProtocol()
        barrier = protocol.create_barrier({"a", "b", "c"})

        assert not barrier.is_complete
        barrier.arrive("a")
        barrier.arrive("b")
        assert not barrier.is_complete

        completed = barrier.arrive("c")
        assert completed is True
        assert barrier.is_complete
        assert barrier.released

    def test_barrier_pending(self) -> None:
        """Barrier reports pending participants."""
        barrier = SyncBarrier(participants=frozenset({"a", "b", "c"}))
        barrier.arrive("a")
        assert barrier.pending == {"b", "c"}


# ─── Conflict Resolution Tests ────────────────────────────────────────


class TestConflictResolver:
    """Tests for the ConflictResolver."""

    def test_last_writer_wins(self) -> None:
        """LAST_WRITER_WINS picks the value with higher clock."""
        resolver = ConflictResolver(default_policy=ConflictPolicy.LAST_WRITER_WINS)
        result = resolver.resolve(
            "price", "graph_a", 100, "graph_b", 90,
            source_clock=5, target_clock=3,
        )
        assert result == 100  # source has higher clock

    def test_first_writer_wins(self) -> None:
        """FIRST_WRITER_WINS picks the value with lower clock."""
        resolver = ConflictResolver(default_policy=ConflictPolicy.FIRST_WRITER_WINS)
        result = resolver.resolve(
            "price", "graph_a", 100, "graph_b", 90,
            source_clock=5, target_clock=3,
        )
        assert result == 90  # target has lower clock

    def test_highest_value(self) -> None:
        """HIGHEST_VALUE picks the larger value."""
        resolver = ConflictResolver(default_policy=ConflictPolicy.HIGHEST_VALUE)
        result = resolver.resolve("price", "a", 100, "b", 150)
        assert result == 150

    def test_lowest_value(self) -> None:
        """LOWEST_VALUE picks the smaller value."""
        resolver = ConflictResolver(default_policy=ConflictPolicy.LOWEST_VALUE)
        result = resolver.resolve("price", "a", 100, "b", 150)
        assert result == 100

    def test_reject_policy(self) -> None:
        """REJECT returns None."""
        resolver = ConflictResolver(default_policy=ConflictPolicy.REJECT)
        result = resolver.resolve("price", "a", 100, "b", 150)
        assert result is None

    def test_custom_merge_function(self) -> None:
        """Custom merge function combines values."""
        resolver = ConflictResolver()
        resolver.set_merge_function("price", lambda s, t: (s + t) / 2)

        result = resolver.resolve("price", "a", 100, "b", 200)
        assert result == 150.0

    def test_per_node_policy(self) -> None:
        """Per-node policy overrides default."""
        resolver = ConflictResolver(default_policy=ConflictPolicy.LAST_WRITER_WINS)
        resolver.set_node_policy("price", ConflictPolicy.HIGHEST_VALUE)

        result = resolver.resolve(
            "price", "a", 50, "b", 100,
            source_clock=10, target_clock=1,
        )
        # HIGHEST_VALUE should win over LAST_WRITER_WINS
        assert result == 100

    def test_conflict_history(self) -> None:
        """Conflicts are recorded in history."""
        resolver = ConflictResolver()
        resolver.resolve("price", "a", 100, "b", 90)
        resolver.resolve("qty", "a", 10, "b", 20)

        assert len(resolver.history) == 2
        assert resolver.stats.total_conflicts == 2

    def test_resolve_batch(self) -> None:
        """Batch resolution handles multiple conflicts."""
        resolver = ConflictResolver(default_policy=ConflictPolicy.HIGHEST_VALUE)
        conflicts = [
            ("price", "a", 100, "b", 150),
            ("qty", "a", 10, "b", 5),
        ]
        results = resolver.resolve_batch(conflicts)
        assert results["price"] == 150
        assert results["qty"] == 10

    def test_frequent_conflicts(self) -> None:
        """Identifies nodes with frequent conflicts."""
        resolver = ConflictResolver()
        for _ in range(5):
            resolver.resolve("hot_node", "a", 1, "b", 2)
        resolver.resolve("cold_node", "a", 1, "b", 2)

        frequent = resolver.get_frequent_conflicts(min_count=3)
        assert "hot_node" in frequent
        assert "cold_node" not in frequent
