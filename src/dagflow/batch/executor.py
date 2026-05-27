"""BatchExecutor — process multiple input changes as a single atomic unit.

Provides a higher-level API over Transaction for common batch patterns:
applying a dict of changes, applying changes with validation, and
executing change sequences with automatic rollback on failure.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode)
- memo.cache (MemoCache)
- batch.transaction (Transaction)
- propagation.eager (EagerPropagator)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode
from dagflow.memo.cache import MemoCache
from dagflow.batch.transaction import Transaction


@dataclass
class BatchResult:
    """Result of a batch execution.

    Attributes:
        success: Whether the batch completed without errors.
        recomputed: Dict of {node_id: new_value} for nodes that changed.
        applied_count: Number of input changes that were actually applied.
        skipped: List of node IDs that were skipped (e.g., same value).
        error: Exception if the batch failed, None otherwise.
    """

    success: bool
    recomputed: Dict[str, Any] = field(default_factory=dict)
    applied_count: int = 0
    skipped: List[str] = field(default_factory=list)
    error: Optional[Exception] = None


class BatchExecutor:
    """Executes batches of input changes atomically.

    All changes in a batch either succeed together or fail together.
    On failure, the graph is rolled back to its pre-batch state.

    Usage:
        executor = BatchExecutor(graph, cache)
        result = executor.apply({"x": 10, "y": 20})
        assert result.success
    """

    def __init__(self, graph: ComputeGraph, cache: MemoCache) -> None:
        self._graph = graph
        self._cache = cache

    def apply(self, changes: Dict[str, Any]) -> BatchResult:
        """Apply a dictionary of {input_node_id: new_value} atomically.

        Args:
            changes: Mapping of input node IDs to their new values.

        Returns:
            BatchResult with success status and recomputed values.
        """
        skipped: List[str] = []
        txn = Transaction(self._graph, self._cache)

        try:
            with txn:
                for node_id, value in changes.items():
                    node = self._graph.get_node(node_id)
                    if isinstance(node, InputNode) and node.value == value:
                        skipped.append(node_id)
                        continue
                    txn.set_input(node_id, value)

            # Transaction committed successfully via context manager
            # We need to get the result from the propagation
            # Re-gather what was recomputed by checking cache
            recomputed = self._gather_current_values()
            applied_count = len(changes) - len(skipped)

            return BatchResult(
                success=True,
                recomputed=recomputed,
                applied_count=applied_count,
                skipped=skipped,
            )

        except Exception as e:
            return BatchResult(success=False, error=e)

    def apply_validated(
        self,
        changes: Dict[str, Any],
        validator: Callable[[str, Any], bool],
    ) -> BatchResult:
        """Apply changes with per-value validation.

        The validator is called for each (node_id, value) pair before
        the change is buffered. If any validation fails, the entire
        batch is rejected without modifying the graph.

        Args:
            changes: Mapping of input node IDs to new values.
            validator: Function(node_id, value) -> bool. Returns True if valid.

        Returns:
            BatchResult. On validation failure, success=False with error details.
        """
        # Validate all changes first (fail-fast)
        for node_id, value in changes.items():
            if not validator(node_id, value):
                return BatchResult(
                    success=False,
                    error=ValueError(
                        f"Validation failed for node '{node_id}' with value {value!r}"
                    ),
                )

        return self.apply(changes)

    def apply_sequence(
        self, change_sequence: List[Dict[str, Any]]
    ) -> List[BatchResult]:
        """Apply a sequence of change batches in order.

        Each batch is applied atomically. If a batch fails, subsequent
        batches are NOT applied, and the failed batch is rolled back.

        Args:
            change_sequence: Ordered list of change dicts to apply.

        Returns:
            List of BatchResults, one per batch attempted.
        """
        results: List[BatchResult] = []

        for changes in change_sequence:
            result = self.apply(changes)
            results.append(result)
            if not result.success:
                break  # Stop on first failure

        return results

    def dry_run(self, changes: Dict[str, Any]) -> BatchResult:
        """Simulate applying changes without persisting them.

        Applies the changes, captures the result, then rolls back.
        Useful for previewing what would change.

        Args:
            changes: Mapping of input node IDs to new values.

        Returns:
            BatchResult showing what WOULD happen, without side effects.
        """
        from dagflow.batch.checkpoint import Checkpoint

        cp = Checkpoint(self._graph, self._cache)
        cp.capture()

        try:
            result = self.apply(changes)
            return result
        finally:
            cp.restore()

    def _gather_current_values(self) -> Dict[str, Any]:
        """Gather current cached values for all compute nodes."""
        values: Dict[str, Any] = {}
        for node_id in self._graph.get_compute_nodes():
            cached = self._cache.get(node_id)
            if cached is not None:
                values[node_id] = cached
        return values
