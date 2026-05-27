"""Batch processing — atomic multi-input changes with transaction support.

This module provides:
- BatchExecutor: Process multiple input changes as a single atomic unit.
- Transaction: Context manager for begin/commit/rollback semantics.
- Checkpoint: Save and restore graph state for rollback.
"""

from dagflow.batch.executor import BatchExecutor
from dagflow.batch.transaction import Transaction
from dagflow.batch.checkpoint import Checkpoint

__all__ = ["BatchExecutor", "Transaction", "Checkpoint"]
