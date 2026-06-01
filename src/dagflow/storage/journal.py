"""Write-ahead journal for crash recovery.

Implements a write-ahead log (WAL) that records all mutations before
they are applied to the main storage. On crash recovery, the journal
can replay uncommitted operations to restore consistent state.

This module depends on:
- storage.base (StorageBackend, StorageEntry, StorageError)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple


class JournalState(Enum):
    """State of a journal entry."""

    PENDING = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()
    RECOVERED = auto()


@dataclass
class JournalEntry:
    """A single entry in the write-ahead journal.

    Attributes:
        sequence: Monotonically increasing sequence number.
        operation: Type of operation (put, delete, clear).
        key: The affected storage key (empty for clear).
        value: The new value (for put operations).
        old_value: The previous value (for undo support).
        state: Current state of this journal entry.
        timestamp: When this entry was written.
        checksum: Integrity checksum of the entry data.
        transaction_id: Optional grouping for atomic operations.
    """

    sequence: int
    operation: str
    key: str = ""
    value: Any = None
    old_value: Any = None
    state: JournalState = JournalState.PENDING
    timestamp: float = field(default_factory=time.monotonic)
    checksum: str = ""
    transaction_id: Optional[str] = None

    def compute_checksum(self) -> str:
        """Compute integrity checksum for this entry."""
        data = f"{self.sequence}:{self.operation}:{self.key}:{self.value}"
        return hashlib.md5(data.encode()).hexdigest()[:8]

    def verify(self) -> bool:
        """Verify entry integrity via checksum."""
        if not self.checksum:
            return True  # No checksum to verify
        return self.compute_checksum() == self.checksum


@dataclass
class RecoveryResult:
    """Result of a journal recovery operation.

    Attributes:
        entries_recovered: Number of entries replayed.
        entries_discarded: Entries that failed verification.
        last_sequence: Highest sequence number recovered.
        operations_applied: Breakdown of operations by type.
    """

    entries_recovered: int = 0
    entries_discarded: int = 0
    last_sequence: int = 0
    operations_applied: Dict[str, int] = field(default_factory=dict)


class WriteAheadJournal:
    """Write-ahead journal for crash-safe storage operations.

    All mutations are first written to the journal before being applied
    to the underlying storage. On recovery, pending entries are replayed
    to restore consistent state.

    Usage:
        journal = WriteAheadJournal(max_entries=10000)
        journal.log_put("node_a", 42, old_value=None)
        journal.commit_up_to(journal.last_sequence)

        # On recovery:
        pending = journal.get_pending_entries()
        for entry in pending:
            apply_to_storage(entry)
        journal.mark_recovered(pending)
    """

    def __init__(
        self,
        max_entries: int = 10000,
        enable_checksums: bool = True,
    ) -> None:
        """Initialize the journal.

        Args:
            max_entries: Maximum entries before compaction.
            enable_checksums: Whether to compute integrity checksums.
        """
        self._entries: List[JournalEntry] = []
        self._max_entries = max_entries
        self._enable_checksums = enable_checksums
        self._sequence: int = 0
        self._active_transactions: Dict[str, List[int]] = {}
        self._compaction_count: int = 0

    @property
    def last_sequence(self) -> int:
        """The most recent sequence number."""
        return self._sequence

    @property
    def entry_count(self) -> int:
        """Total entries in the journal."""
        return len(self._entries)

    @property
    def pending_count(self) -> int:
        """Number of uncommitted entries."""
        return sum(1 for e in self._entries if e.state == JournalState.PENDING)

    def log_put(
        self,
        key: str,
        value: Any,
        old_value: Any = None,
        transaction_id: Optional[str] = None,
    ) -> JournalEntry:
        """Log a put operation to the journal.

        Args:
            key: The storage key being written.
            value: The new value.
            old_value: The previous value (for undo).
            transaction_id: Optional transaction grouping.

        Returns:
            The created JournalEntry.
        """
        return self._append_entry("put", key, value, old_value, transaction_id)

    def log_delete(
        self,
        key: str,
        old_value: Any = None,
        transaction_id: Optional[str] = None,
    ) -> JournalEntry:
        """Log a delete operation to the journal.

        Args:
            key: The storage key being deleted.
            old_value: The value being deleted (for undo).
            transaction_id: Optional transaction grouping.

        Returns:
            The created JournalEntry.
        """
        return self._append_entry("delete", key, None, old_value, transaction_id)

    def log_clear(self, transaction_id: Optional[str] = None) -> JournalEntry:
        """Log a clear-all operation to the journal."""
        return self._append_entry("clear", "", None, None, transaction_id)

    def commit_up_to(self, sequence: int) -> int:
        """Mark all entries up to sequence as committed.

        Args:
            sequence: Commit all entries with sequence <= this value.

        Returns:
            Number of entries committed.
        """
        count = 0
        for entry in self._entries:
            if entry.sequence <= sequence and entry.state == JournalState.PENDING:
                entry.state = JournalState.COMMITTED
                count += 1
        return count

    def commit_transaction(self, transaction_id: str) -> int:
        """Commit all entries belonging to a transaction.

        Args:
            transaction_id: The transaction to commit.

        Returns:
            Number of entries committed.
        """
        count = 0
        for entry in self._entries:
            if (
                entry.transaction_id == transaction_id
                and entry.state == JournalState.PENDING
            ):
                entry.state = JournalState.COMMITTED
                count += 1

        self._active_transactions.pop(transaction_id, None)
        return count

    def rollback_transaction(self, transaction_id: str) -> List[JournalEntry]:
        """Roll back all entries in a transaction.

        Returns entries in reverse order for undo application.

        Args:
            transaction_id: The transaction to roll back.

        Returns:
            List of entries to undo, in reverse order.
        """
        to_rollback: List[JournalEntry] = []
        for entry in self._entries:
            if (
                entry.transaction_id == transaction_id
                and entry.state == JournalState.PENDING
            ):
                entry.state = JournalState.ROLLED_BACK
                to_rollback.append(entry)

        self._active_transactions.pop(transaction_id, None)
        return list(reversed(to_rollback))

    def get_pending_entries(self) -> List[JournalEntry]:
        """Get all pending (uncommitted) entries for recovery.

        Returns:
            List of pending entries in sequence order.
        """
        return [e for e in self._entries if e.state == JournalState.PENDING]

    def mark_recovered(self, entries: List[JournalEntry]) -> int:
        """Mark entries as recovered after replay.

        Args:
            entries: Entries that were successfully replayed.

        Returns:
            Number of entries marked.
        """
        count = 0
        recovered_seqs = {e.sequence for e in entries}
        for entry in self._entries:
            if entry.sequence in recovered_seqs:
                entry.state = JournalState.RECOVERED
                count += 1
        return count

    def recover(
        self,
        apply_fn: Callable[[JournalEntry], bool],
    ) -> RecoveryResult:
        """Perform crash recovery by replaying pending entries.

        Args:
            apply_fn: Function that applies a journal entry to storage.
                      Returns True if successful.

        Returns:
            RecoveryResult with recovery statistics.
        """
        result = RecoveryResult()
        pending = self.get_pending_entries()

        for entry in pending:
            # Verify integrity
            if self._enable_checksums and not entry.verify():
                result.entries_discarded += 1
                entry.state = JournalState.ROLLED_BACK
                continue

            # Apply the entry
            if apply_fn(entry):
                entry.state = JournalState.RECOVERED
                result.entries_recovered += 1
                result.last_sequence = max(result.last_sequence, entry.sequence)

                op = entry.operation
                result.operations_applied[op] = (
                    result.operations_applied.get(op, 0) + 1
                )
            else:
                result.entries_discarded += 1
                entry.state = JournalState.ROLLED_BACK

        return result

    def compact(self) -> int:
        """Remove committed and rolled-back entries to free memory.

        Keeps only pending and recovered entries.

        Returns:
            Number of entries removed.
        """
        before = len(self._entries)
        self._entries = [
            e
            for e in self._entries
            if e.state in (JournalState.PENDING, JournalState.RECOVERED)
        ]
        removed = before - len(self._entries)
        self._compaction_count += 1
        return removed

    def get_undo_entries(self, count: int = 1) -> List[JournalEntry]:
        """Get the most recent entries for undo operations.

        Args:
            count: Number of entries to retrieve.

        Returns:
            List of entries in reverse chronological order.
        """
        committed = [e for e in self._entries if e.state == JournalState.COMMITTED]
        return list(reversed(committed[-count:]))

    def _append_entry(
        self,
        operation: str,
        key: str,
        value: Any,
        old_value: Any,
        transaction_id: Optional[str],
    ) -> JournalEntry:
        """Create and append a new journal entry."""
        self._sequence += 1

        entry = JournalEntry(
            sequence=self._sequence,
            operation=operation,
            key=key,
            value=value,
            old_value=old_value,
            transaction_id=transaction_id,
        )

        if self._enable_checksums:
            entry.checksum = entry.compute_checksum()

        self._entries.append(entry)

        # Track transaction membership
        if transaction_id:
            if transaction_id not in self._active_transactions:
                self._active_transactions[transaction_id] = []
            self._active_transactions[transaction_id].append(entry.sequence)

        # Auto-compact if too large
        if len(self._entries) > self._max_entries:
            self.compact()

        return entry
