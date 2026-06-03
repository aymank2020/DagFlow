"""Tests for the persistent storage backends.

Tests cover:
- Abstract interface compliance
- MemoryStore: LRU eviction, capacity, batch operations
- SQLiteStore: persistence, transactions, batch queries
- WriteAheadJournal: logging, recovery, transactions
"""

from __future__ import annotations

import pytest

from dagflow.storage.base import StorageBackend, StorageEntry, StorageStats
from dagflow.storage.memory_store import MemoryStore
from dagflow.storage.sqlite_store import SQLiteStore
from dagflow.storage.journal import (
    JournalEntry,
    JournalState,
    RecoveryResult,
    WriteAheadJournal,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def memory_store() -> MemoryStore:
    """Create an unbounded memory store."""
    return MemoryStore()


@pytest.fixture
def bounded_store() -> MemoryStore:
    """Create a memory store with capacity limit."""
    return MemoryStore(max_entries=5)


@pytest.fixture
def sqlite_store() -> SQLiteStore:
    """Create an in-memory SQLite store."""
    store = SQLiteStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def journal() -> WriteAheadJournal:
    """Create a write-ahead journal."""
    return WriteAheadJournal(max_entries=100)


# ─── MemoryStore Tests ─────────────────────────────────────────────────


class TestMemoryStore:
    """Tests for the in-memory storage backend."""

    def test_put_and_get(self, memory_store: MemoryStore) -> None:
        """Basic put/get round-trip."""
        entry = memory_store.put("node_a", 42, generation=1)
        assert entry.key == "node_a"
        assert entry.value == 42
        assert entry.generation == 1

        retrieved = memory_store.get("node_a")
        assert retrieved is not None
        assert retrieved.value == 42

    def test_get_nonexistent_returns_none(self, memory_store: MemoryStore) -> None:
        """Getting a missing key returns None."""
        assert memory_store.get("missing") is None

    def test_delete(self, memory_store: MemoryStore) -> None:
        """Delete removes an entry."""
        memory_store.put("x", 1)
        assert memory_store.delete("x") is True
        assert memory_store.get("x") is None
        assert memory_store.delete("x") is False  # Already gone

    def test_exists(self, memory_store: MemoryStore) -> None:
        """exists() checks presence without LRU update."""
        memory_store.put("x", 1)
        assert memory_store.exists("x") is True
        assert memory_store.exists("y") is False

    def test_size_and_clear(self, memory_store: MemoryStore) -> None:
        """size() and clear() work correctly."""
        memory_store.put("a", 1)
        memory_store.put("b", 2)
        assert memory_store.size() == 2

        cleared = memory_store.clear()
        assert cleared == 2
        assert memory_store.size() == 0

    def test_keys_returns_all(self, memory_store: MemoryStore) -> None:
        """keys() returns all stored keys."""
        memory_store.put("x", 1)
        memory_store.put("y", 2)
        memory_store.put("z", 3)
        assert set(memory_store.keys()) == {"x", "y", "z"}

    def test_lru_eviction(self, bounded_store: MemoryStore) -> None:
        """LRU eviction removes oldest entry when at capacity."""
        for i in range(5):
            bounded_store.put(f"key_{i}", i)
        assert bounded_store.size() == 5

        # Adding one more should evict key_0
        bounded_store.put("key_5", 5)
        assert bounded_store.size() == 5
        assert bounded_store.get("key_0") is None
        assert bounded_store.eviction_count == 1

    def test_lru_access_updates_order(self, bounded_store: MemoryStore) -> None:
        """Accessing an entry moves it to most-recently-used."""
        for i in range(5):
            bounded_store.put(f"key_{i}", i)

        # Access key_0 to make it most recent
        bounded_store.get("key_0")

        # Now adding a new key should evict key_1 (oldest after key_0 was accessed)
        bounded_store.put("key_5", 5)
        assert bounded_store.get("key_0") is not None  # Still present
        assert bounded_store.get("key_1") is None  # Evicted

    def test_put_many(self, memory_store: MemoryStore) -> None:
        """Batch put stores multiple entries."""
        entries = memory_store.put_many({"a": 1, "b": 2, "c": 3})
        assert len(entries) == 3
        assert memory_store.size() == 3

    def test_get_many(self, memory_store: MemoryStore) -> None:
        """Batch get retrieves multiple entries."""
        memory_store.put("a", 1)
        memory_store.put("b", 2)

        results = memory_store.get_many(["a", "b", "c"])
        assert results["a"] is not None
        assert results["b"] is not None
        assert results["c"] is None

    def test_snapshot_and_restore(self, memory_store: MemoryStore) -> None:
        """Snapshot captures state, restore rebuilds it."""
        memory_store.put("x", 10)
        memory_store.put("y", 20)

        snap = memory_store.snapshot()
        assert snap == {"x": 10, "y": 20}

        memory_store.clear()
        restored = memory_store.restore(snap)
        assert restored == 2
        assert memory_store.get("x").value == 10

    def test_utilization(self, bounded_store: MemoryStore) -> None:
        """Utilization reports fraction of capacity used."""
        assert bounded_store.utilization == 0.0
        bounded_store.put("a", 1)
        assert bounded_store.utilization == pytest.approx(0.2)

    def test_peek_no_lru_update(self, bounded_store: MemoryStore) -> None:
        """peek() retrieves without affecting LRU order."""
        for i in range(5):
            bounded_store.put(f"key_{i}", i)

        # Peek at key_0 (should NOT update LRU)
        entry = bounded_store.peek("key_0")
        assert entry is not None
        assert entry.value == 0

        # key_0 should still be oldest
        assert bounded_store.oldest_key() == "key_0"


# ─── SQLiteStore Tests ─────────────────────────────────────────────────


class TestSQLiteStore:
    """Tests for the SQLite storage backend."""

    def test_put_and_get(self, sqlite_store: SQLiteStore) -> None:
        """Basic put/get with SQLite."""
        sqlite_store.put("node_a", {"result": 42}, generation=1)
        entry = sqlite_store.get("node_a")

        assert entry is not None
        assert entry.value == {"result": 42}
        assert entry.generation == 1

    def test_get_nonexistent(self, sqlite_store: SQLiteStore) -> None:
        """Missing key returns None."""
        assert sqlite_store.get("missing") is None

    def test_delete(self, sqlite_store: SQLiteStore) -> None:
        """Delete removes entry from database."""
        sqlite_store.put("x", 1)
        assert sqlite_store.delete("x") is True
        assert sqlite_store.exists("x") is False

    def test_overwrite(self, sqlite_store: SQLiteStore) -> None:
        """Put with existing key overwrites."""
        sqlite_store.put("x", 1, generation=1)
        sqlite_store.put("x", 2, generation=2)

        entry = sqlite_store.get("x")
        assert entry.value == 2
        assert entry.generation == 2

    def test_keys_and_size(self, sqlite_store: SQLiteStore) -> None:
        """keys() and size() reflect stored entries."""
        sqlite_store.put("a", 1)
        sqlite_store.put("b", 2)

        assert sqlite_store.size() == 2
        assert set(sqlite_store.keys()) == {"a", "b"}

    def test_clear(self, sqlite_store: SQLiteStore) -> None:
        """clear() removes all entries."""
        sqlite_store.put("a", 1)
        sqlite_store.put("b", 2)

        cleared = sqlite_store.clear()
        assert cleared == 2
        assert sqlite_store.size() == 0

    def test_batch_put(self, sqlite_store: SQLiteStore) -> None:
        """put_many stores multiple entries in one transaction."""
        entries = sqlite_store.put_many({"x": 10, "y": 20, "z": 30})
        assert len(entries) == 3
        assert sqlite_store.size() == 3

    def test_batch_get(self, sqlite_store: SQLiteStore) -> None:
        """get_many retrieves multiple entries efficiently."""
        sqlite_store.put_many({"a": 1, "b": 2, "c": 3})
        results = sqlite_store.get_many(["a", "c", "missing"])

        assert results["a"].value == 1
        assert results["c"].value == 3
        assert results["missing"] is None

    def test_serialization_types(self, sqlite_store: SQLiteStore) -> None:
        """Various Python types are serialized correctly."""
        sqlite_store.put("int", 42)
        sqlite_store.put("float", 3.14)
        sqlite_store.put("str", "hello")
        sqlite_store.put("list", [1, 2, 3])
        sqlite_store.put("dict", {"key": "value"})
        sqlite_store.put("null", None)

        assert sqlite_store.get("int").value == 42
        assert sqlite_store.get("float").value == pytest.approx(3.14)
        assert sqlite_store.get("str").value == "hello"
        assert sqlite_store.get("list").value == [1, 2, 3]
        assert sqlite_store.get("dict").value == {"key": "value"}
        assert sqlite_store.get("null").value is None

    def test_stats(self, sqlite_store: SQLiteStore) -> None:
        """Stats track read/write counts."""
        sqlite_store.put("a", 1)
        sqlite_store.get("a")
        sqlite_store.get("b")

        stats = sqlite_store.get_stats()
        assert stats.total_writes == 1
        assert stats.total_reads == 2


# ─── WriteAheadJournal Tests ──────────────────────────────────────────


class TestWriteAheadJournal:
    """Tests for the write-ahead journal."""

    def test_log_put(self, journal: WriteAheadJournal) -> None:
        """Logging a put creates a pending entry."""
        entry = journal.log_put("node_a", 42, old_value=None)
        assert entry.operation == "put"
        assert entry.key == "node_a"
        assert entry.value == 42
        assert entry.state == JournalState.PENDING
        assert journal.pending_count == 1

    def test_log_delete(self, journal: WriteAheadJournal) -> None:
        """Logging a delete records the operation."""
        entry = journal.log_delete("node_a", old_value=42)
        assert entry.operation == "delete"
        assert entry.old_value == 42

    def test_commit_up_to(self, journal: WriteAheadJournal) -> None:
        """Committing marks entries as committed."""
        journal.log_put("a", 1)
        journal.log_put("b", 2)
        seq = journal.log_put("c", 3).sequence

        committed = journal.commit_up_to(seq)
        assert committed == 3
        assert journal.pending_count == 0

    def test_transaction_commit(self, journal: WriteAheadJournal) -> None:
        """Transaction commit marks all entries in the transaction."""
        journal.log_put("a", 1, transaction_id="tx1")
        journal.log_put("b", 2, transaction_id="tx1")
        journal.log_put("c", 3, transaction_id="tx2")

        committed = journal.commit_transaction("tx1")
        assert committed == 2
        assert journal.pending_count == 1  # tx2 still pending

    def test_transaction_rollback(self, journal: WriteAheadJournal) -> None:
        """Rollback marks entries as rolled back and returns undo list."""
        journal.log_put("a", 1, transaction_id="tx1")
        journal.log_put("b", 2, transaction_id="tx1")

        undo_entries = journal.rollback_transaction("tx1")
        assert len(undo_entries) == 2
        # Undo entries are in reverse order
        assert undo_entries[0].key == "b"
        assert undo_entries[1].key == "a"
        assert journal.pending_count == 0

    def test_get_pending_entries(self, journal: WriteAheadJournal) -> None:
        """get_pending_entries returns only uncommitted entries."""
        journal.log_put("a", 1)
        journal.log_put("b", 2)
        journal.commit_up_to(1)

        pending = journal.get_pending_entries()
        assert len(pending) == 1
        assert pending[0].key == "b"

    def test_recovery(self, journal: WriteAheadJournal) -> None:
        """Recovery replays pending entries."""
        journal.log_put("a", 1)
        journal.log_put("b", 2)

        applied: list = []
        result = journal.recover(
            apply_fn=lambda entry: (applied.append(entry.key), True)[1]
        )

        assert result.entries_recovered == 2
        assert set(applied) == {"a", "b"}

    def test_checksum_verification(self) -> None:
        """Entries with valid checksums pass verification."""
        journal = WriteAheadJournal(enable_checksums=True)
        entry = journal.log_put("test", 42)
        assert entry.verify() is True

    def test_compact_removes_committed(self, journal: WriteAheadJournal) -> None:
        """Compaction removes committed and rolled-back entries."""
        journal.log_put("a", 1)
        journal.log_put("b", 2)
        journal.commit_up_to(2)
        journal.log_put("c", 3)  # Still pending

        removed = journal.compact()
        assert removed == 2
        assert journal.entry_count == 1

    def test_sequence_monotonic(self, journal: WriteAheadJournal) -> None:
        """Sequence numbers are monotonically increasing."""
        e1 = journal.log_put("a", 1)
        e2 = journal.log_put("b", 2)
        e3 = journal.log_delete("a")

        assert e1.sequence < e2.sequence < e3.sequence

    def test_undo_entries(self, journal: WriteAheadJournal) -> None:
        """get_undo_entries returns recent committed entries in reverse."""
        journal.log_put("a", 1)
        journal.log_put("b", 2)
        journal.log_put("c", 3)
        journal.commit_up_to(3)

        undo = journal.get_undo_entries(count=2)
        assert len(undo) == 2
        assert undo[0].key == "c"
        assert undo[1].key == "b"
