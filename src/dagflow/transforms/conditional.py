"""Conditional — if-then-else and switch computation nodes.

Provides nodes that select between different computation paths based
on the value of a condition node. Enables dynamic control flow within
the static DAG structure.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode


class ConditionalNode:
    """Factory for creating if-then-else compute nodes.

    A ConditionalNode evaluates a condition and selects between two
    branches (then/else) based on the result.

    Usage:
        cond = ConditionalNode(graph)
        cond.create(
            "result",
            condition="flag",
            then_source="value_a",
            else_source="value_b",
        )
    """

    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def create(
        self,
        node_id: str,
        condition: str,
        then_source: str,
        else_source: str,
        priority: int = 0,
    ) -> ComputeNode:
        """Create an if-then-else node.

        Selects then_source's value if condition is truthy,
        else_source's value otherwise.

        Args:
            node_id: ID for the new node.
            condition: ID of the condition node (truthy/falsy).
            then_source: ID of the node to use when condition is True.
            else_source: ID of the node to use when condition is False.
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        def cond_func(deps: Dict[str, Any]) -> Any:
            if deps.get(condition):
                return deps.get(then_source)
            else:
                return deps.get(else_source)

        dependencies = [condition, then_source, else_source]
        return self._graph.add_compute(
            node_id, func=cond_func, dependencies=dependencies, priority=priority
        )

    def create_with_predicate(
        self,
        node_id: str,
        source: str,
        predicate: Callable[[Any], bool],
        then_transform: Callable[[Any], Any],
        else_transform: Callable[[Any], Any],
        priority: int = 0,
    ) -> ComputeNode:
        """Create a conditional node with inline predicate and transforms.

        Evaluates predicate on the source value, then applies either
        then_transform or else_transform.

        Args:
            node_id: ID for the new node.
            source: ID of the source node.
            predicate: Function to test the source value.
            then_transform: Applied when predicate returns True.
            else_transform: Applied when predicate returns False.
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        def pred_func(deps: Dict[str, Any]) -> Any:
            value = deps.get(source)
            if predicate(value):
                return then_transform(value)
            else:
                return else_transform(value)

        return self._graph.add_compute(
            node_id, func=pred_func, dependencies=[source], priority=priority
        )

    def create_threshold(
        self,
        node_id: str,
        source: str,
        threshold: float,
        above_value: Any,
        below_value: Any,
        priority: int = 0,
    ) -> ComputeNode:
        """Create a threshold-based conditional node.

        Returns above_value if source >= threshold, below_value otherwise.

        Args:
            node_id: ID for the new node.
            source: ID of the numeric source node.
            threshold: The threshold value.
            above_value: Value to produce when source >= threshold.
            below_value: Value to produce when source < threshold.
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        def threshold_func(deps: Dict[str, Any]) -> Any:
            value = deps.get(source)
            if value is not None and value >= threshold:
                return above_value
            return below_value

        return self._graph.add_compute(
            node_id, func=threshold_func, dependencies=[source], priority=priority
        )


class SwitchNode:
    """Factory for creating multi-way switch compute nodes.

    A SwitchNode selects from multiple branches based on the value of
    a selector node, similar to a switch/case statement.

    Usage:
        switch = SwitchNode(graph)
        switch.create(
            "output",
            selector="mode",
            cases={"fast": "fast_result", "slow": "slow_result"},
            default="fallback_result",
        )
    """

    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def create(
        self,
        node_id: str,
        selector: str,
        cases: Dict[Any, str],
        default: Optional[str] = None,
        priority: int = 0,
    ) -> ComputeNode:
        """Create a switch node that selects between multiple sources.

        Args:
            node_id: ID for the new node.
            selector: ID of the node whose value determines the case.
            cases: Mapping of {selector_value: source_node_id}.
            default: ID of the default source node (used when no case matches).
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        # All case sources + selector + default are dependencies
        dependencies = [selector]
        for source_id in cases.values():
            if source_id not in dependencies:
                dependencies.append(source_id)
        if default and default not in dependencies:
            dependencies.append(default)

        case_map = dict(cases)
        default_source = default

        def switch_func(deps: Dict[str, Any]) -> Any:
            selector_value = deps.get(selector)
            target_id = case_map.get(selector_value)
            if target_id is not None:
                return deps.get(target_id)
            elif default_source is not None:
                return deps.get(default_source)
            return None

        return self._graph.add_compute(
            node_id, func=switch_func, dependencies=dependencies, priority=priority
        )

    def create_computed(
        self,
        node_id: str,
        selector: str,
        cases: Dict[Any, Callable[[Dict[str, Any]], Any]],
        sources: List[str],
        default: Optional[Callable[[Dict[str, Any]], Any]] = None,
        priority: int = 0,
    ) -> ComputeNode:
        """Create a switch node with computed case branches.

        Instead of selecting between source nodes, each case executes
        a different computation function over the same set of sources.

        Args:
            node_id: ID for the new node.
            selector: ID of the selector node.
            cases: Mapping of {selector_value: compute_function}.
            sources: IDs of source nodes available to all case functions.
            default: Default computation function.
            priority: Scheduling priority.

        Returns:
            The created ComputeNode.
        """
        dependencies = [selector] + [s for s in sources if s != selector]
        case_map = dict(cases)
        default_func = default

        def computed_switch_func(deps: Dict[str, Any]) -> Any:
            selector_value = deps.get(selector)
            case_func = case_map.get(selector_value)
            if case_func is not None:
                return case_func(deps)
            elif default_func is not None:
                return default_func(deps)
            return None

        return self._graph.add_compute(
            node_id,
            func=computed_switch_func,
            dependencies=dependencies,
            priority=priority,
        )
