"""Data pipeline example — ETL-style incremental processing.

Demonstrates DagFlow as an ETL (Extract-Transform-Load) pipeline where
raw data sources feed through transformation stages. When source data
changes, only affected pipeline stages are re-executed.

This example uses:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode)
- memo.cache (MemoCache)
- propagation.eager (EagerPropagator)
- transforms.map_reduce (MapNode, FilterNode, ReduceNode)
- transforms.aggregate (AggregateNode, AggregateOp)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode, ComputeNode
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.transforms.map_reduce import MapNode, FilterNode, ReduceNode
from dagflow.transforms.aggregate import AggregateNode, AggregateOp


class DataPipeline:
    """An ETL-style data pipeline backed by DagFlow.

    Data flows through stages: Source -> Transform -> Aggregate -> Output.
    Each stage is a node in the DAG. When source data changes, only
    downstream stages that are affected get recomputed.

    Usage:
        pipeline = DataPipeline()
        pipeline.add_source("raw_sales", [100, 200, 150, 300])
        pipeline.add_transform("cleaned", "raw_sales",
                              lambda x: [v for v in x if v > 0])
        pipeline.add_aggregation("total", ["cleaned"], AggregateOp.SUM)
        pipeline.execute()
    """

    def __init__(self) -> None:
        self._graph = ComputeGraph()
        self._cache = MemoCache()
        self._propagator = EagerPropagator(self._graph, self._cache)
        self._stages: Dict[str, str] = {}  # node_id -> stage_type

    @property
    def graph(self) -> ComputeGraph:
        """The underlying computation graph."""
        return self._graph

    @property
    def cache(self) -> MemoCache:
        """The underlying memo cache."""
        return self._cache

    def add_source(self, name: str, data: Any) -> None:
        """Add a data source to the pipeline.

        Args:
            name: Source identifier.
            data: Initial data (any type).
        """
        self._graph.add_input(name, value=data)
        self._stages[name] = "source"

    def add_transform(
        self,
        name: str,
        source: str,
        transform: Callable[[Any], Any],
    ) -> None:
        """Add a transformation stage.

        Args:
            name: Stage identifier.
            source: ID of the upstream stage.
            transform: Function to apply to the source data.
        """
        def transform_func(deps: Dict[str, Any]) -> Any:
            return transform(deps[source])

        self._graph.add_compute(name, func=transform_func, dependencies=[source])
        self._stages[name] = "transform"

    def add_join(
        self,
        name: str,
        sources: List[str],
        join_func: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """Add a join stage that combines multiple sources.

        Args:
            name: Stage identifier.
            sources: IDs of upstream stages to join.
            join_func: Function that receives {source_id: data} and returns joined data.
        """
        self._graph.add_compute(name, func=join_func, dependencies=sources)
        self._stages[name] = "join"

    def add_aggregation(
        self,
        name: str,
        source: str,
        agg_func: Callable[[Any], Any],
    ) -> None:
        """Add an aggregation stage.

        Args:
            name: Stage identifier.
            source: ID of the upstream stage.
            agg_func: Function to aggregate the source data.
        """
        def aggregation_func(deps: Dict[str, Any]) -> Any:
            return agg_func(deps[source])

        self._graph.add_compute(name, func=aggregation_func, dependencies=[source])
        self._stages[name] = "aggregation"

    def update_source(self, name: str, data: Any) -> Dict[str, Any]:
        """Update a source and propagate changes through the pipeline.

        Args:
            name: Source identifier.
            data: New data.

        Returns:
            Dict of {stage_id: new_value} for all recomputed stages.
        """
        node = self._graph.get_node(name)
        if not isinstance(node, InputNode):
            raise ValueError(f"'{name}' is not a source")
        node.set(data)
        return self._propagator.propagate([name])

    def execute(self) -> Dict[str, Any]:
        """Execute the full pipeline from all sources.

        Returns:
            Dict of {stage_id: value} for all computed stages.
        """
        sources = self._graph.get_input_nodes()
        return self._propagator.propagate(sources)

    def get_result(self, stage: str) -> Any:
        """Get the current result of a pipeline stage.

        Args:
            stage: Stage identifier.

        Returns:
            The current value at that stage.
        """
        node = self._graph.get_node(stage)
        if isinstance(node, InputNode):
            return node.value
        return self._cache.get(stage)

    def get_stage_type(self, name: str) -> Optional[str]:
        """Get the type of a pipeline stage."""
        return self._stages.get(name)

    @property
    def stage_count(self) -> int:
        """Total number of pipeline stages."""
        return len(self._stages)


def demo_data_pipeline() -> DataPipeline:
    """Create a demo ETL pipeline for sales data processing.

    Pipeline:
        raw_sales -> clean (remove negatives) -> enriched (add tax)
        raw_costs -> clean_costs
        enriched + clean_costs -> profit_calc -> summary

    Returns:
        A configured DataPipeline instance.
    """
    pipeline = DataPipeline()

    # Sources
    pipeline.add_source("raw_sales", [100, 200, -50, 300, 150])
    pipeline.add_source("raw_costs", [80, 120, 60, 180, 90])
    pipeline.add_source("tax_rate", 0.1)

    # Transform: clean sales (remove negatives)
    pipeline.add_transform(
        "clean_sales",
        "raw_sales",
        lambda data: [x for x in data if x > 0],
    )

    # Transform: clean costs
    pipeline.add_transform(
        "clean_costs",
        "raw_costs",
        lambda data: [x for x in data if x > 0],
    )

    # Transform: compute total sales
    pipeline.add_transform(
        "total_sales",
        "clean_sales",
        lambda data: sum(data),
    )

    # Transform: compute total costs
    pipeline.add_transform(
        "total_costs",
        "clean_costs",
        lambda data: sum(data),
    )

    # Join: compute profit
    pipeline.add_join(
        "gross_profit",
        ["total_sales", "total_costs"],
        lambda d: d["total_sales"] - d["total_costs"],
    )

    # Aggregation: apply tax to get net profit
    pipeline.add_join(
        "net_profit",
        ["gross_profit", "tax_rate"],
        lambda d: d["gross_profit"] * (1 - d["tax_rate"]),
    )

    return pipeline
