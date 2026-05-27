"""Transaction — context manager for atomic graph mutations.

Provides begin/commit/rollback semantics over a ComputeGraph. Changes
are buffered during the transaction and only applied (propagated) on commit.
If an error occurs or rollback is called, the graph reverts to its
pre-transaction state via Checkpoint.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode)
- memo.cache (MemoCache)
- batch.checkpoint (Checkpoint)
- propagation.eager (EagerPropagator)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode
from dagflow.memo.cache import MemoCache
from dagflow.batch.checkpoint import Checkpoint
from dagflow.propagation.eager import EagerPropagator


@dataclass
class PendingChange:
    """A buffered input change waiting to be applied."""

    node_id: str
    new_value: Any


class TransactionError(Exception):
    """Raised when transaction operations are used incorrectly."""


class Transaction:
    """Atomic transaction over graph input changes.

    Usage as context manager:
        with Transaction(graph, cache) as txn:
            txn.set_input("x", 42)
            txn.set_input("y", 99)
        # On exit: all changes are propagated atomically

    Usage with explicit control:
        txn = Transaction(graph, cache)
        txn.begin()
        txn.set_input("x", 42)
        txn.commit()  # or txn.rollback()

    If an exception occurs inside the context manager, rollback is automatic.
    """

    def __init__(self, graph: ComputeGraph, cache: MemoCache) -> None:
        self._graph = graph
        self._cache = cache
        self._checkpoint = Checkpoint(graph, cache)
        self._propagator = EagerPropagator(graph, cache)
        self._pending: List[PendingChange] = []
        self._active: bool = False
        self._committed: bool = False

    @property
    def is_active(self) -> bool:
        """Whether a transaction is currently in progress."""
        return self._active

    @property
    def pending_count(self) -> int:
        """Number of buffered changes not yet committed."""
        return len(self._pending)

    def begin(self) -> None:
        """Start a new transaction, capturing a checkpoint.

        Raises TransactionError if a transaction is already active.
        """
        if self._active:
            raise TransactionError("Transaction already active")
        self._checkpoint.capture()
        self._pending.clear()
        self._active = True
        self._committed = False

    def set_input(self, node_id: str, value: Any) -> None:
        """Buffer an input change within the transaction.

        The change is NOT applied to the graph until commit().

        Raises:
            TransactionError: If no transaction is active.
            ValueError: If node_id is not an input node.
        """
        if not self._active:
            raise TransactionError("No active transaction — call begin() first")

        node = self._graph.get_node(node_id)
        if not isinstance(node, InputNode):
            raise ValueError(f"Node '{node_id}' is not an input node")

        self._pending.append(PendingChange(node_id=node_id, new_value=value))

    def commit(self) -> Dict[str, Any]:
        """Apply all buffered changes and propagate atomically.

        Returns:
            Dict of {node_id: new_value} for all recomputed nodes.

        Raises:
            TransactionError: If no transaction is active.
        """
        if not self._active:
            raise TransactionError("No active transaction")

        try:
            # Apply all input changes
            changed_inputs: List[str] = []
            for change in self._pending:
                node = self._graph.get_node(change.node_id)
                if isinstance(node, InputNode):
                    old_gen = node.generation
                    node.set(change.new_value)
                    if node.generation > old_gen:
                        changed_inputs.append(change.node_id)

            # Propagate all changes at once
            result = self._propagator.propagate(changed_inputs)
            self._active = False
            self._committed = True
            self._checkpoint.discard()
            return result

        except Exception:
            # On failure, rollback automatically
            self.rollback()
            raise

    def rollback(self) -> None:
        """Revert the graph to its pre-transaction state.

        Raises TransactionError if no transaction is active.
        """
        if not self._active:
            raise TransactionError("No active transaction to rollback")

        self._checkpoint.restore()
        self._pending.clear()
        self._active = False
        self._checkpoint.discard()

    def __enter__(self) -> "Transaction":
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Exception occurred — rollback
            if self._active:
                self.rollback()
            return False  # Re-raise the exception
        else:
            # No exception — commit if still active
            if self._active:
                self.commit()
            return False
