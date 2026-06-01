"""SQLite-backed persistent storage backend.

Provides durable storage using SQLite, suitable for graphs that need
to survive process restarts. Supports transactions, batch operations,
and schema migrations.

This module depends on:
- storage.base (StorageBackend, StorageEntry, StorageStats, StorageError)
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from dagflow.storage.base import (
    StorageBackend,
    StorageEntry,
    StorageError,
    StorageStats,
)


class SQLiteStore(StorageBackend):
    """SQLite-backed persistent storage for DAG state.

    Stores entries as JSON-serialized values in a SQLite database.
    Supports both file-based and in-memory (`:memory:`) databases.

    Usage:
        store = SQLiteStore("dagflow_state.db")
        store.put("node_a", {"result": 42}, generation=3)
        entry = store.get("node_a")

        # In-memory for testing
        store = SQLiteStore(":memory:")
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize the SQLite store.

        Args:
            db_path: Path to the SQLite database file, or ":memory:".
        """
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._total_writes: int = 0
        self._total_reads: int = 0
        self._connect()
        self._ensure_schema()

    def _connect(self) -> None:
        """Establish database connection."""
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                generation INTEGER DEFAULT 0,
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS schema_info (
                version INTEGER PRIMARY KEY
            );
            INSERT OR IGNORE INTO schema_info (version) VALUES (1);
        """)
        self._conn.commit()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database transactions."""
        assert self._conn is not None
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def get(self, key: str) -> Optional[StorageEntry]:
        """Retrieve an entry from SQLite."""
        self._total_reads += 1
        assert self._conn is not None

        cursor = self._conn.execute(
            "SELECT key, value, generation, timestamp, metadata FROM entries WHERE key = ?",
            (key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return StorageEntry(
            key=row["key"],
            value=self._deserialize(row["value"]),
            generation=row["generation"],
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]),
        )

    def put(self, key: str, value: Any, generation: int = 0) -> StorageEntry:
        """Store an entry in SQLite."""
        self._total_writes += 1
        assert self._conn is not None

        now = time.monotonic()
        serialized = self._serialize(value)

        with self._transaction() as cursor:
            cursor.execute(
                """INSERT OR REPLACE INTO entries (key, value, generation, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, serialized, generation, now, "{}"),
            )

        return StorageEntry(
            key=key,
            value=value,
            generation=generation,
            timestamp=now,
        )

    def delete(self, key: str) -> bool:
        """Delete an entry from SQLite."""
        assert self._conn is not None

        with self._transaction() as cursor:
            cursor.execute("DELETE FROM entries WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT 1 FROM entries WHERE key = ?", (key,)
        )
        return cursor.fetchone() is not None

    def keys(self) -> List[str]:
        """Return all keys."""
        assert self._conn is not None
        cursor = self._conn.execute("SELECT key FROM entries ORDER BY key")
        return [row["key"] for row in cursor.fetchall()]

    def clear(self) -> int:
        """Remove all entries."""
        assert self._conn is not None
        cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM entries")
        count = cursor.fetchone()["cnt"]

        with self._transaction() as cur:
            cur.execute("DELETE FROM entries")

        return count

    def size(self) -> int:
        """Return number of entries."""
        assert self._conn is not None
        cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM entries")
        return cursor.fetchone()["cnt"]

    def get_many(self, keys: List[str]) -> Dict[str, Optional[StorageEntry]]:
        """Batch retrieve using a single query."""
        self._total_reads += len(keys)
        assert self._conn is not None

        if not keys:
            return {}

        placeholders = ",".join("?" * len(keys))
        cursor = self._conn.execute(
            f"SELECT key, value, generation, timestamp, metadata FROM entries WHERE key IN ({placeholders})",
            keys,
        )

        results: Dict[str, Optional[StorageEntry]] = {k: None for k in keys}
        for row in cursor.fetchall():
            results[row["key"]] = StorageEntry(
                key=row["key"],
                value=self._deserialize(row["value"]),
                generation=row["generation"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"]),
            )
        return results

    def put_many(
        self, entries: Dict[str, Any], generation: int = 0
    ) -> List[StorageEntry]:
        """Batch store using a single transaction."""
        self._total_writes += len(entries)
        assert self._conn is not None

        now = time.monotonic()
        results: List[StorageEntry] = []

        with self._transaction() as cursor:
            for key, value in entries.items():
                serialized = self._serialize(value)
                cursor.execute(
                    """INSERT OR REPLACE INTO entries (key, value, generation, timestamp, metadata)
                       VALUES (?, ?, ?, ?, ?)""",
                    (key, serialized, generation, now, "{}"),
                )
                results.append(
                    StorageEntry(key=key, value=value, generation=generation, timestamp=now)
                )

        return results

    def get_stats(self) -> StorageStats:
        """Return storage statistics."""
        return StorageStats(
            entry_count=self.size(),
            total_writes=self._total_writes,
            total_reads=self._total_reads,
            cache_hits=0,
            cache_misses=self._total_reads,
        )

    def vacuum(self) -> None:
        """Reclaim unused space in the database file."""
        assert self._conn is not None
        self._conn.execute("VACUUM")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _serialize(self, value: Any) -> str:
        """Serialize a value to JSON string."""
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return json.dumps(str(value))

    def _deserialize(self, data: str) -> Any:
        """Deserialize a JSON string to a value."""
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data

    def __del__(self) -> None:
        """Ensure connection is closed on garbage collection."""
        self.close()
