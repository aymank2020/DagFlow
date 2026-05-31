"""Async scheduler for deferred stream processing.

Provides an event-loop-friendly scheduler that processes stream events
in configurable batches, supports priority queues for urgent events,
and integrates with the DAG propagation engine for deferred recomputation.

This module depends on:
- reactive.stream (ReactiveStream, StreamEvent, EventKind)
- core.graph (ComputeGraph)
- propagation.eager (EagerPropagator)
- memo.cache (MemoCache)
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.reactive.stream import EventKind, ReactiveStream, StreamEvent


class Priority(IntEnum):
    """Event processing priority levels."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    IDLE = 4


@dataclass(order=True)
class ScheduledTask:
    """A task queued for deferred execution.

    Attributes:
        priority: Processing priority (lower = sooner).
        sequence: Insertion order for stable sorting within same priority.
        node_id: The node to recompute.
        trigger_time: When this task was scheduled.
        delay: Minimum seconds to wait before execution.
    """

    priority: int
    sequence: int
    node_id: str = field(compare=False)
    trigger_time: float = field(compare=False, default_factory=time.monotonic)
    delay: float = field(compare=False, default=0.0)

    @property
    def ready_time(self) -> float:
        """Earliest time this task can execute."""
        return self.trigger_time + self.delay


class AsyncScheduler:
    """Deferred event scheduler with priority queue and batch processing.

    Processes stream events asynchronously by queuing them and executing
    in priority order during explicit tick() calls. Supports:
    - Priority-based ordering (critical events processed first)
    - Delayed execution (schedule for future processing)
    - Batch processing (process N events per tick)
    - Coalescing (merge multiple updates to same node)

    Usage:
        scheduler = AsyncScheduler(graph, cache)
        scheduler.schedule("node_a", priority=Priority.HIGH)
        scheduler.schedule("node_b", priority=Priority.LOW)
        results = scheduler.tick(max_tasks=10)
    """

    def __init__(
        self,
        graph: ComputeGraph,
        cache: MemoCache,
        coalesce: bool = True,
        default_batch_size: int = 50,
    ) -> None:
        self._graph = graph
        self._cache = cache
        self._propagator = EagerPropagator(graph, cache)
        self._coalesce = coalesce
        self._default_batch_size = default_batch_size
        self._queue: List[ScheduledTask] = []
        self._sequence: int = 0
        self._pending_nodes: Set[str] = set()
        self._processed_count: int = 0
        self._total_delay: float = 0.0
        self._callbacks: List[Callable[[str, Any], None]] = []

    @property
    def pending_count(self) -> int:
        """Number of tasks waiting to be processed."""
        return len(self._queue)

    @property
    def processed_count(self) -> int:
        """Total number of tasks processed since creation."""
        return self._processed_count

    @property
    def average_delay(self) -> float:
        """Average time between scheduling and execution."""
        if self._processed_count == 0:
            return 0.0
        return self._total_delay / self._processed_count

    def schedule(
        self,
        node_id: str,
        priority: Priority = Priority.NORMAL,
        delay: float = 0.0,
    ) -> bool:
        """Schedule a node for deferred recomputation.

        If coalescing is enabled and the node is already pending,
        the existing entry is updated with the higher priority.

        Args:
            node_id: The node to recompute.
            priority: Processing priority.
            delay: Minimum seconds to wait before processing.

        Returns:
            True if the task was added, False if coalesced with existing.
        """
        if self._coalesce and node_id in self._pending_nodes:
            # Update priority if new one is higher (lower number)
            self._update_priority(node_id, priority)
            return False

        self._sequence += 1
        task = ScheduledTask(
            priority=int(priority),
            sequence=self._sequence,
            node_id=node_id,
            delay=delay,
        )
        heapq.heappush(self._queue, task)
        self._pending_nodes.add(node_id)
        return True

    def schedule_batch(
        self,
        node_ids: List[str],
        priority: Priority = Priority.NORMAL,
        delay: float = 0.0,
    ) -> int:
        """Schedule multiple nodes for deferred recomputation.

        Args:
            node_ids: List of node IDs to schedule.
            priority: Processing priority for all nodes.
            delay: Minimum delay for all nodes.

        Returns:
            Number of tasks actually added (excludes coalesced).
        """
        added = 0
        for node_id in node_ids:
            if self.schedule(node_id, priority, delay):
                added += 1
        return added

    def tick(self, max_tasks: Optional[int] = None) -> Dict[str, Any]:
        """Process pending tasks up to max_tasks.

        Executes ready tasks in priority order, triggering propagation
        for each. Tasks with future ready_time are skipped.

        Args:
            max_tasks: Maximum tasks to process. None = default batch size.

        Returns:
            Dict of {node_id: new_value} for all recomputed nodes.
        """
        limit = max_tasks or self._default_batch_size
        now = time.monotonic()
        results: Dict[str, Any] = {}
        processed = 0

        # Collect ready tasks
        ready_tasks: List[ScheduledTask] = []
        deferred: List[ScheduledTask] = []

        while self._queue and processed < limit:
            task = heapq.heappop(self._queue)

            if task.ready_time > now:
                deferred.append(task)
                continue

            ready_tasks.append(task)
            processed += 1

        # Put deferred tasks back
        for task in deferred:
            heapq.heappush(self._queue, task)

        # Execute ready tasks via propagation
        if ready_tasks:
            changed_inputs: List[str] = []
            for task in ready_tasks:
                self._pending_nodes.discard(task.node_id)
                node = self._graph.get_node(task.node_id)
                if node.is_input:
                    changed_inputs.append(task.node_id)

                # Track timing
                actual_delay = now - task.trigger_time
                self._total_delay += actual_delay
                self._processed_count += 1

            # Propagate all changes together for efficiency
            if changed_inputs:
                propagated = self._propagator.propagate(changed_inputs)
                results.update(propagated)

            # Notify callbacks
            for node_id, value in results.items():
                for callback in self._callbacks:
                    callback(node_id, value)

        return results

    def on_complete(self, callback: Callable[[str, Any], None]) -> None:
        """Register a callback invoked when a task completes.

        Args:
            callback: Function(node_id, new_value) called after recomputation.
        """
        self._callbacks.append(callback)

    def cancel(self, node_id: str) -> bool:
        """Cancel a pending task for a node.

        Args:
            node_id: The node whose task should be cancelled.

        Returns:
            True if a task was found and cancelled.
        """
        if node_id not in self._pending_nodes:
            return False

        self._queue = [t for t in self._queue if t.node_id != node_id]
        heapq.heapify(self._queue)
        self._pending_nodes.discard(node_id)
        return True

    def cancel_all(self) -> int:
        """Cancel all pending tasks.

        Returns:
            Number of tasks cancelled.
        """
        count = len(self._queue)
        self._queue.clear()
        self._pending_nodes.clear()
        return count

    def peek(self) -> Optional[ScheduledTask]:
        """Peek at the next task without removing it."""
        return self._queue[0] if self._queue else None

    def drain_ready(self) -> Dict[str, Any]:
        """Process ALL ready tasks regardless of batch size."""
        return self.tick(max_tasks=len(self._queue) or 1)

    def _update_priority(self, node_id: str, new_priority: Priority) -> None:
        """Update priority of an existing queued task if new priority is higher."""
        for i, task in enumerate(self._queue):
            if task.node_id == node_id:
                if int(new_priority) < task.priority:
                    # Replace with higher priority
                    self._queue[i] = ScheduledTask(
                        priority=int(new_priority),
                        sequence=task.sequence,
                        node_id=node_id,
                        trigger_time=task.trigger_time,
                        delay=task.delay,
                    )
                    heapq.heapify(self._queue)
                break
