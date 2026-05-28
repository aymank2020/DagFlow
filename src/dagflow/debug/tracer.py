"""ExecutionTracer — records computation events for debugging.

Wraps a ComputeGraph's propagation cycle to record which nodes were
computed, in what order, what values they produced, and whether
early-cutoff was applied. The trace can be inspected after propagation
to understand the execution flow.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode, NodeState)
- memo.cache (MemoCache)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode, NodeState
from dagflow.memo.cache import MemoCache


class EventType(Enum):
    """Types of trace events."""

    INPUT_CHANGED = auto()
    NODE_INVALIDATED = auto()
    NODE_SCHEDULED = auto()
    NODE_COMPUTED = auto()
    NODE_CUTOFF = auto()  # Early cutoff — value unchanged


@dataclass
class TraceEvent:
    """A single event in the execution trace.

    Attributes:
        event_type: What kind of event occurred.
        node_id: The node involved.
        timestamp: When the event occurred (monotonic).
        value: The value produced (for COMPUTED events).
        previous_value: The previous value (for COMPUTED events).
        order: Sequential position in the trace.
    """

    event_type: EventType
    node_id: str
    timestamp: float
    value: Any = None
    previous_value: Any = None
    order: int = 0


class ExecutionTracer:
    """Records execution events during graph propagation.

    Usage:
        tracer = ExecutionTracer(graph, cache)
        tracer.start()

        # Change inputs and propagate
        inp.set(42)
        tracer.trace_propagation(["inp"])

        tracer.stop()
        for event in tracer.events:
            print(event)
    """

    def __init__(self, graph: ComputeGraph, cache: MemoCache) -> None:
        self._graph = graph
        self._cache = cache
        self._events: List[TraceEvent] = []
        self._active: bool = False
        self._order_counter: int = 0

    @property
    def is_active(self) -> bool:
        """Whether the tracer is currently recording."""
        return self._active

    @property
    def events(self) -> List[TraceEvent]:
        """All recorded trace events."""
        return list(self._events)

    @property
    def computed_nodes(self) -> List[str]:
        """Node IDs that were computed (in order)."""
        return [
            e.node_id
            for e in self._events
            if e.event_type == EventType.NODE_COMPUTED
        ]

    @property
    def cutoff_nodes(self) -> List[str]:
        """Node IDs where early cutoff was applied."""
        return [
            e.node_id
            for e in self._events
            if e.event_type == EventType.NODE_CUTOFF
        ]

    def start(self) -> None:
        """Begin recording trace events."""
        self._active = True

    def stop(self) -> None:
        """Stop recording trace events."""
        self._active = False

    def clear(self) -> None:
        """Clear all recorded events."""
        self._events.clear()
        self._order_counter = 0

    def trace_propagation(self, changed_inputs: List[str]) -> Dict[str, Any]:
        """Propagate changes while recording trace events.

        This is a tracing wrapper around the propagation logic. It records
        events for each phase: invalidation, scheduling, and computation.

        Args:
            changed_inputs: List of input node IDs that changed.

        Returns:
            Dict of {node_id: new_value} for recomputed nodes.
        """
        if not self._active:
            # If not active, just do a plain propagation
            from dagflow.propagation.eager import EagerPropagator
            prop = EagerPropagator(self._graph, self._cache)
            return prop.propagate(changed_inputs)

        # Record input changes
        for inp_id in changed_inputs:
            self._record(EventType.INPUT_CHANGED, inp_id)

        # Invalidation phase
        from dagflow.propagation.invalidator import Invalidator
        invalidator = Invalidator(self._graph)
        dirty_set = invalidator.invalidate_from(changed_inputs)

        for node_id in dirty_set:
            self._record(EventType.NODE_INVALIDATED, node_id)

        # Scheduling phase
        from dagflow.scheduler.topo import TopologicalScheduler
        scheduler = TopologicalScheduler(self._graph)
        execution_order = scheduler.schedule(dirty_set)

        for node_id in execution_order:
            self._record(EventType.NODE_SCHEDULED, node_id)

        # Execution phase with early-cutoff tracking
        recomputed: Dict[str, Any] = {}
        cutoff_clean: Set[str] = set()

        for node_id in execution_order:
            if node_id in cutoff_clean:
                continue

            node = self._graph.get_node(node_id)
            if not isinstance(node, ComputeNode):
                continue

            # Gather deps
            dep_values: Dict[str, Any] = {}
            dep_gens: Dict[str, int] = {}
            for dep_id in node.dependencies:
                dep_node = self._graph.get_node(dep_id)
                if isinstance(dep_node, InputNode):
                    dep_values[dep_id] = dep_node.value
                    dep_gens[dep_id] = dep_node.generation
                elif isinstance(dep_node, ComputeNode):
                    dep_values[dep_id] = dep_node.cached_value
                    dep_gens[dep_id] = self._cache.get_generation(dep_id)

            # Compute
            old_value = node.cached_value
            new_value = node.func(dep_values)
            node.mark_clean(new_value, dep_gens)
            self._cache.store(node_id, new_value, dep_gens)

            if new_value == old_value and old_value is not None:
                self._record(EventType.NODE_CUTOFF, node_id, new_value, old_value)
                downstream = self._graph.get_all_downstream(node_id)
                for ds_id in downstream:
                    ds_node = self._graph.get_node(ds_id)
                    if isinstance(ds_node, ComputeNode):
                        ds_node.state = NodeState.CLEAN
                        cutoff_clean.add(ds_id)
            else:
                self._record(EventType.NODE_COMPUTED, node_id, new_value, old_value)
                recomputed[node_id] = new_value

        return recomputed

    def _record(
        self,
        event_type: EventType,
        node_id: str,
        value: Any = None,
        previous_value: Any = None,
    ) -> None:
        """Record a single trace event."""
        self._order_counter += 1
        self._events.append(
            TraceEvent(
                event_type=event_type,
                node_id=node_id,
                timestamp=time.monotonic(),
                value=value,
                previous_value=previous_value,
                order=self._order_counter,
            )
        )

    def get_events_for_node(self, node_id: str) -> List[TraceEvent]:
        """Get all events related to a specific node."""
        return [e for e in self._events if e.node_id == node_id]

    def get_execution_order(self) -> List[str]:
        """Get the order in which nodes were actually computed."""
        return self.computed_nodes
