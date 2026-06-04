"""Example: Real-time dashboard with reactive streams.

Demonstrates using the reactive streams layer to build a dashboard
that updates in real-time as input data changes. Shows:
- Stream creation and observation
- Subscription management with groups
- Buffer operator for batched UI updates
- Pause/resume for controlled rendering

This example simulates a financial dashboard that tracks portfolio
values, computes aggregates, and pushes updates to UI components.
"""

from __future__ import annotations

from typing import Any, Dict, List

from dagflow.core.graph import ComputeGraph
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.reactive.stream import EventKind, ReactiveStream, StreamEvent
from dagflow.reactive.subscription import SubscriptionManager, SubscriptionState
from dagflow.reactive.operators import buffer, merge


def build_portfolio_graph() -> tuple[ComputeGraph, MemoCache]:
    """Build a computation graph for portfolio tracking.

    Graph structure:
        stock_price, quantity -> position_value
        bond_price, bond_qty -> bond_value
        position_value, bond_value -> total_portfolio
        total_portfolio, benchmark -> relative_performance
    """
    graph = ComputeGraph()
    cache = MemoCache()

    # Input nodes: market data
    graph.add_input("stock_price", value=150.0)
    graph.add_input("quantity", value=100)
    graph.add_input("bond_price", value=98.5)
    graph.add_input("bond_qty", value=50)
    graph.add_input("benchmark", value=10000.0)

    # Compute nodes: derived values
    graph.add_compute(
        "position_value",
        lambda d: d["stock_price"] * d["quantity"],
        ["stock_price", "quantity"],
    )
    graph.add_compute(
        "bond_value",
        lambda d: d["bond_price"] * d["bond_qty"],
        ["bond_price", "bond_qty"],
    )
    graph.add_compute(
        "total_portfolio",
        lambda d: d["position_value"] + d["bond_value"],
        ["position_value", "bond_value"],
    )
    graph.add_compute(
        "relative_performance",
        lambda d: (d["total_portfolio"] / d["benchmark"] - 1.0) * 100,
        ["total_portfolio", "benchmark"],
    )

    return graph, cache


class DashboardWidget:
    """Simulates a UI widget that displays a value."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.current_value: Any = None
        self.update_count: int = 0
        self.history: List[Any] = []

    def update(self, value: Any) -> None:
        """Update the widget with a new value."""
        self.current_value = value
        self.update_count += 1
        self.history.append(value)

    def __repr__(self) -> str:
        return f"Widget({self.name}={self.current_value})"


class ReactiveDashboard:
    """A reactive dashboard that auto-updates when data changes.

    Demonstrates the reactive streams pattern for building
    responsive UIs on top of the DAG computation engine.
    """

    def __init__(self) -> None:
        self.graph, self.cache = build_portfolio_graph()
        self.propagator = EagerPropagator(self.graph, self.cache)
        self.sub_manager = SubscriptionManager()

        # Create streams for different data categories
        self.price_stream = ReactiveStream(self.graph, buffer_size=100)
        self.price_stream.observe("position_value", "bond_value")

        self.portfolio_stream = ReactiveStream(self.graph, buffer_size=50)
        self.portfolio_stream.observe("total_portfolio", "relative_performance")

        # Merge into a single update stream
        self.all_updates = merge(
            self.price_stream, self.portfolio_stream, graph=self.graph
        )

        # Create widgets
        self.widgets: Dict[str, DashboardWidget] = {
            "position": DashboardWidget("Position Value"),
            "bonds": DashboardWidget("Bond Value"),
            "total": DashboardWidget("Total Portfolio"),
            "performance": DashboardWidget("Performance %"),
        }

        # Wire up subscriptions
        self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        """Connect streams to widgets via managed subscriptions."""
        node_to_widget = {
            "position_value": "position",
            "bond_value": "bonds",
            "total_portfolio": "total",
            "relative_performance": "performance",
        }

        for node_id, widget_name in node_to_widget.items():
            widget = self.widgets[widget_name]
            self.sub_manager.subscribe(
                self.all_updates,
                lambda event, w=widget: w.update(event.value),
                group="dashboard",
                filter_fn=lambda e, nid=node_id: e.node_id == nid,
            )

    def update_market_data(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Push new market data and propagate through the graph.

        Args:
            changes: Dict of {input_node_id: new_value}.

        Returns:
            Dict of recomputed values.
        """
        # Apply input changes
        changed_inputs: List[str] = []
        for node_id, value in changes.items():
            node = self.graph.get_node(node_id)
            node.set(value)
            changed_inputs.append(node_id)

        # Propagate
        recomputed = self.propagator.propagate(changed_inputs)

        # Emit to streams
        self.price_stream.emit_changes(recomputed)
        self.portfolio_stream.emit_changes(recomputed)

        return recomputed

    def pause_updates(self) -> None:
        """Pause dashboard updates (e.g., during batch loading)."""
        self.sub_manager.pause_group("dashboard")

    def resume_updates(self) -> None:
        """Resume dashboard updates."""
        self.sub_manager.resume_group("dashboard")

    def get_widget_states(self) -> Dict[str, Any]:
        """Get current state of all widgets."""
        return {
            name: {"value": w.current_value, "updates": w.update_count}
            for name, w in self.widgets.items()
        }


def main() -> None:
    """Run the reactive dashboard example."""
    dashboard = ReactiveDashboard()

    # Initial computation
    print("=== Initial State ===")
    results = dashboard.update_market_data({"stock_price": 150.0})
    print(f"Recomputed: {results}")
    print(f"Widgets: {dashboard.get_widget_states()}")

    # Market update: stock price rises
    print("\n=== Stock Price Update: $150 -> $165 ===")
    results = dashboard.update_market_data({"stock_price": 165.0})
    print(f"Recomputed: {results}")
    print(f"Widgets: {dashboard.get_widget_states()}")

    # Batch update: multiple changes at once
    print("\n=== Batch Update ===")
    dashboard.pause_updates()
    dashboard.update_market_data({"stock_price": 170.0, "bond_price": 99.0})
    dashboard.resume_updates()
    print(f"Widgets after batch: {dashboard.get_widget_states()}")

    # Show subscription stats
    stats = dashboard.sub_manager.get_group_stats("dashboard")
    print(f"\nSubscription stats: {stats}")


if __name__ == "__main__":
    main()
