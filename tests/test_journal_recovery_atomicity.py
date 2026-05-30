"""Regression tests for atomic crash recovery in the write-ahead journal."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set

from dagflow.storage.journal import (
    JournalEntry,
    JournalState,
    WriteAheadJournal,
)


def _make_apply_fn(
    store: Dict[str, object],
    fail_sequences: Optional[Set[int]] = None,
) -> Callable[[JournalEntry], bool]:
    """Build an apply function that mutates ``store`` like real storage would.

    Entries whose sequence is in ``fail_sequences`` report failure so we can
    exercise mid-transaction apply errors. Compensating entries produced during
    rollback are ordinary put/delete operations and revert ``store``.
    """
    fail_sequences = fail_sequences or set()

    def apply_fn(entry: JournalEntry) -> bool:
        if entry.sequence in fail_sequences:
            return False
        if entry.operation == "put":
            store[entry.key] = entry.value
        elif entry.operation == "delete":
            store.pop(entry.key, None)
        elif entry.operation == "clear":
            store.clear()
        return True

    return apply_fn


def _corrupt(entry: JournalEntry) -> None:
    """Tamper an entry's value so its checksum no longer verifies."""
    entry.value = ("corrupted", entry.value)


# --- fail-to-pass: atomicity must hold -------------------------------------

def test_corrupted_entry_aborts_entire_transaction() -> None:
    journal = WriteAheadJournal(enable_checksums=True)
    good = journal.log_put("alpha", 1, old_value=None, transaction_id="tx")
    bad = journal.log_put("beta", 2, old_value=None, transaction_id="tx")
    _corrupt(bad)

    store: Dict[str, object] = {}
    result = journal.recover(_make_apply_fn(store))

    # A corrupt member must prevent the whole transaction from being applied.
    assert store == {}
    assert result.entries_recovered == 0
    assert result.entries_discarded == 2


def test_apply_failure_rolls_back_transaction() -> None:
    journal = WriteAheadJournal(enable_checksums=True)
    journal.log_put("a", 10, old_value=None, transaction_id="tx")
    mid = journal.log_put("b", 20, old_value=None, transaction_id="tx")
    journal.log_put("c", 30, old_value=None, transaction_id="tx")

    store: Dict[str, object] = {}
    result = journal.recover(_make_apply_fn(store, fail_sequences={mid.sequence}))

    # The earlier write must be undone; nothing from the transaction survives.
    assert store == {}
    assert result.entries_recovered == 0


def test_partial_transaction_leaves_no_recovered_entries() -> None:
    journal = WriteAheadJournal(enable_checksums=True)
    first = journal.log_put("x", 1, old_value=None, transaction_id="tx")
    second = journal.log_put("y", 2, old_value=None, transaction_id="tx")
    _corrupt(second)

    journal.recover(_make_apply_fn({}))

    assert first.state == JournalState.ROLLED_BACK
    assert second.state == JournalState.ROLLED_BACK


def test_independent_transaction_recovers_when_another_aborts() -> None:
    journal = WriteAheadJournal(enable_checksums=True)
    journal.log_put("keep_1", 1, old_value=None, transaction_id="ok")
    journal.log_put("keep_2", 2, old_value=None, transaction_id="ok")
    journal.log_put("drop_1", 3, old_value=None, transaction_id="bad")
    doomed = journal.log_put("drop_2", 4, old_value=None, transaction_id="bad")
    _corrupt(doomed)

    store: Dict[str, object] = {}
    result = journal.recover(_make_apply_fn(store))

    assert store == {"keep_1": 1, "keep_2": 2}
    assert result.entries_recovered == 2
    assert result.entries_discarded == 2


def test_recovered_store_matches_committed_transactions_only() -> None:
    journal = WriteAheadJournal(enable_checksums=True)
    journal.log_put("good", 100, old_value=None, transaction_id="t1")
    journal.log_put("gone_a", 1, old_value=None, transaction_id="t2")
    fail = journal.log_put("gone_b", 2, old_value=None, transaction_id="t2")

    store: Dict[str, object] = {}
    journal.recover(_make_apply_fn(store, fail_sequences={fail.sequence}))

    assert store == {"good": 100}


# --- pass-to-pass: existing behaviour must be preserved --------------------

def test_standalone_entries_recover_independently() -> None:
    journal = WriteAheadJournal(enable_checksums=True)
    journal.log_put("a", 1)
    journal.log_put("b", 2)

    store: Dict[str, object] = {}
    result = journal.recover(_make_apply_fn(store))

    assert store == {"a": 1, "b": 2}
    assert result.entries_recovered == 2


def test_standalone_corrupted_entry_only_discards_itself() -> None:
    journal = WriteAheadJournal(enable_checksums=True)
    journal.log_put("a", 1)
    bad = journal.log_put("b", 2)
    journal.log_put("c", 3)
    _corrupt(bad)

    store: Dict[str, object] = {}
    result = journal.recover(_make_apply_fn(store))

    assert store == {"a": 1, "c": 3}
    assert result.entries_recovered == 2
    assert result.entries_discarded == 1


def test_fully_valid_transaction_recovers_completely() -> None:
    journal = WriteAheadJournal(enable_checksums=True)
    journal.log_put("a", 1, old_value=None, transaction_id="tx")
    journal.log_put("b", 2, old_value=None, transaction_id="tx")
    journal.log_put("c", 3, old_value=None, transaction_id="tx")

    store: Dict[str, object] = {}
    result = journal.recover(_make_apply_fn(store))

    assert store == {"a": 1, "b": 2, "c": 3}
    assert result.entries_recovered == 3
