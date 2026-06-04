"""Example: Multi-graph ETL pipeline with federation.

Demonstrates using the distributed module to build a federated
ETL (Extract-Transform-Load) pipeline where:
- Extract graph: reads raw data and normalizes it
- Transform graph: applies business rules and computations
- Load graph: aggregates results for output

Shows:
- Federated graph registration
- Cross-graph edges with transforms
- Propagation across graph boundaries
- Sync protocol for coordination
- Conflict resolution for concurrent updates
"""

from __future__ import annotations

from typing import Any, Dict, List

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode
from dagflow.memo.cache import MemoCache
from dagflow.distributed.federation import FederatedGraph
from dagflow.distributed.sync import SyncProtocol, SyncState
from dagflow.distributed.conflict import ConflictPolicy, ConflictResolver


def build_extract_graph() -> tuple[ComputeGraph, MemoCache]:
    """Build the extraction graph that normalizes raw data.

    Simulates reading from multiple data sources and normalizing
    values into a common format.
    """
    graph = ComputeGraph()
    cache = MemoCache()

    # Raw inputs (simulating data source reads)
    graph.add_input("raw_revenue", value=1_000_000)
    graph.add_input("raw_costs", value=750_000)
    graph.add_input("exchange_rate", value=1.0)
    graph.add_input("raw_headcount", value=50)

    # Normalization: convert to base currency
    graph.add_compute(
        "normalized_revenue",
        lambda d: d["raw_revenue"] * d["exchange_rate"],
        ["raw_revenue", "exchange_rate"],
    )
    graph.add_compute(
        "normalized_costs",
        lambda d: d["raw_costs"] * d["exchange_rate"],
        ["raw_costs", "exchange_rate"],
    )
    graph.add_compute(
        "headcount",
        lambda d: max(1, d["raw_headcount"]),  # Ensure at least 1
        ["raw_headcount"],
    )

    return graph, cache


def build_transform_graph() -> tuple[ComputeGraph, MemoCache]:
    """Build the transformation graph that applies business rules.

    Computes derived metrics from normalized data.
    """
    graph = ComputeGraph()
    cache = MemoCache()

    # Inputs (fed from extract graph via cross-edges)
    graph.add_input("revenue", value=0)
    graph.add_input("costs", value=0)
    graph.add_input("employees", value=1)

    # Business rule computations
    graph.add_compute(
        "gross_profit",
        lambda d: d["revenue"] - d["costs"],
        ["revenue", "costs"],
    )
    graph.add_compute(
        "profit_margin",
        lambda d: (d["revenue"] - d["costs"]) / max(d["revenue"], 1) * 100,
        ["revenue", "costs"],
    )
    graph.add_compute(
        "revenue_per_employee",
        lambda d: d["revenue"] / max(d["employees"], 1),
        ["revenue", "employees"],
    )
    graph.add_compute(
        "cost_per_employee",
        lambda d: d["costs"] / max(d["employees"], 1),
        ["costs", "employees"],
    )

    return graph, cache


def build_load_graph() -> tuple[ComputeGraph, MemoCache]:
    """Build the load graph that aggregates for output.

    Combines transformed metrics into final output format.
    """
    graph = ComputeGraph()
    cache = MemoCache()

    # Inputs (fed from transform graph)
    graph.add_input("profit", value=0)
    graph.add_input("margin", value=0)
    graph.add_input("rev_per_emp", value=0)

    # Aggregation and scoring
    graph.add_compute(
        "health_score",
        lambda d: min(100, max(0, d["margin"] * 2 + (d["rev_per_emp"] / 1000))),
        ["margin", "rev_per_emp"],
    )
    graph.add_compute(
        "summary",
        lambda d: {
            "profit": d["profit"],
            "margin_pct": round(d["margin"], 2),
            "health": round(d["profit"] * (d["margin"] / 100), 2) if d["margin"] > 0 else 0,
        },
        ["profit", "margin"],
    )

    return graph, cache


class ETLPipeline:
    """A federated ETL pipeline using DagFlow.

    Coordinates three computation graphs (extract, transform, load)
    with cross-graph dependencies and synchronization.
    """

    def __init__(self) -> None:
        # Build individual graphs
        self.extract_graph, self.extract_cache = build_extract_graph()
        self.transform_graph, self.transform_cache = build_transform_graph()
        self.load_graph, self.load_cache = build_load_graph()

        # Set up federation
        self.federation = FederatedGraph()
        self.federation.register("extract", self.extract_graph, self.extract_cache)
        self.federation.register("transform", self.transform_graph, self.transform_cache)
        self.federation.register("load", self.load_graph, self.load_cache)

        # Cross-graph edges: extract -> transform
        self.federation.add_cross_edge(
            "extract", "normalized_revenue", "transform", "revenue"
        )
        self.federation.add_cross_edge(
            "extract", "normalized_costs", "transform", "costs"
        )
        self.federation.add_cross_edge(
            "extract", "headcount", "transform", "employees"
        )

        # Cross-graph edges: transform -> load
        self.federation.add_cross_edge(
            "transform", "gross_profit", "load", "profit"
        )
        self.federation.add_cross_edge(
            "transform", "profit_margin", "load", "margin"
        )
        self.federation.add_cross_edge(
            "transform", "revenue_per_employee", "load", "rev_per_emp"
        )

        # Sync protocol
        self.sync = SyncProtocol()
        self.sync.register_graph("extract")
        self.sync.register_graph("transform")
        self.sync.register_graph("load")

        # Conflict resolver for concurrent updates
        self.resolver = ConflictResolver(
            default_policy=ConflictPolicy.LAST_WRITER_WINS
        )

    def ingest_data(self, raw_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Ingest raw data and propagate through the entire pipeline.

        Args:
            raw_data: Dict of raw input values for the extract graph.

        Returns:
            Results from all three graphs.
        """
        # Apply raw data to extract graph
        changed: List[str] = []
        for key, value in raw_data.items():
            try:
                node = self.extract_graph.get_node(key)
                if isinstance(node, InputNode):
                    node.set(value)
                    changed.append(key)
            except KeyError:
                continue

        # Record mutation for sync tracking
        self.sync.record_mutation("extract", ",".join(changed))

        # Propagate through federation
        results = self.federation.propagate("extract", changed)

        # Record downstream mutations
        if "transform" in results:
            self.sync.record_mutation("transform")
        if "load" in results:
            self.sync.record_mutation("load")

        return results

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of the pipeline."""
        return {
            "graphs": self.federation.graph_count,
            "cross_edges": self.federation.cross_edge_count,
            "sync_state": {
                "extract->transform": self.sync.check_sync("extract", "transform").name,
                "transform->load": self.sync.check_sync("transform", "load").name,
            },
        }


def main() -> None:
    """Run the distributed ETL pipeline example."""
    pipeline = ETLPipeline()

    print("=== Distributed ETL Pipeline ===")
    print(f"Status: {pipeline.get_pipeline_status()}")

    # Ingest initial data
    print("\n--- Ingesting Q1 Data ---")
    results = pipeline.ingest_data({
        "raw_revenue": 1_200_000,
        "raw_costs": 800_000,
        "exchange_rate": 1.0,
        "raw_headcount": 60,
    })

    for graph_name, values in results.items():
        print(f"\n  [{graph_name}]")
        for node_id, value in values.items():
            if isinstance(value, float):
                print(f"    {node_id}: {value:.2f}")
            else:
                print(f"    {node_id}: {value}")

    # Update with new exchange rate
    print("\n--- Exchange Rate Change: 1.0 -> 1.15 ---")
    results = pipeline.ingest_data({"exchange_rate": 1.15})

    for graph_name, values in results.items():
        print(f"\n  [{graph_name}]")
        for node_id, value in values.items():
            if isinstance(value, float):
                print(f"    {node_id}: {value:.2f}")
            else:
                print(f"    {node_id}: {value}")

    print(f"\nFinal status: {pipeline.get_pipeline_status()}")


if __name__ == "__main__":
    main()
