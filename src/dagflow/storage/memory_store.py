"""In-memory storage backend — fast, non-persistent.

Provides the default storage backend that keeps all data in memory.
Supports optional capacity limits with LRU eviction, and tracks
access statistics for cache optimization.

This module depends on:
- storage.base (StorageBackend, StorageEntry, StorageStats, StorageCapacityError)
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

from dagflow.storage.base import (
    StorageBackend,
    StorageCapacityError,
    StorageEntry,
    StorageStats,
)


class MemoryStore(StorageBackend):
    """In-memory storage with optional LRU eviction.

    Stores entries in an OrderedDict for O(1) access with LRU ordering.
    When capacity is exceeded, the least recently accessed entries are
    evicted automatically.

    Usage:
        store = MemoryStore(max_entries=1000)
        store.put("node_a", 42, generation=1)
        entry = store.get("node_a")
        assert entry.value == 42
    """

    def __init__(self, max_entries: Optional[int] = None) -> None:
        """Initialize the memory store.

        Args:
            max_entries: Maximum entries before LRU eviction. None = unlimited.
        """
        self._data: OrderedDict[str, StorageEntry] = OrderedDict()
        self._max_entries = max_entries
        self._total_writes: int = 0
        self._total_reads: int = 0
        self._eviction_count: int = 0

    def get(self, key: str) -> Optional[StorageEntry]:
        """Retrieve an entry, moving it to most-recently-used position."""
        self._total_reads += 1
        entry = self._data.get(key)
        if entry is not None:
            # Move to end (most recently used)
            self._data.move_to_end(key)
        return entry

    def put(self, key: str, value: Any, generation: int = 0) -> StorageEntry:
        """Store an entry, evicting LRU entries if at capacity."""
        self._total_writes += 1

        # Check capacity before inserting new key
        if (
            self._max_entries is not None
            and key not in self._data
            and len(self._data) >= self._max_entries
        ):
            self._evict_lru()

        entry = StorageEntry(
            key=key,
            value=value,
            generation=generation,
            timestamp=time.monotonic(),
        )
        self._data[key] = entry
        self._data.move_to_end(key)
        return entry

    def delete(self, key: str) -> bool:
        """Remove an entry by key."""
        if key in self._data:
            del self._data[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if a key exists without affecting LRU order."""
        return key in self._data

    def keys(self) -> List[str]:
        """Return all keys in LRU order (oldest first)."""
        return list(self._data.keys())

    def clear(self) -> int:
        """Remove all entries."""
        count = len(self._data)
        self._data.clear()
        return count

    def size(self) -> int:
        """Return current number of entries."""
        return len(self._data)

    def get_many(self, keys: List[str]) -> Dict[str, Optional[StorageEntry]]:
        """Batch retrieve with LRU updates for found entries."""
        results: Dict[str, Optional[StorageEntry]] = {}
        for key in keys:
            results[key] = self.get(key)
        return results

    def put_many(
        self, entries: Dict[str, Any], generation: int = 0
    ) -> List[StorageEntry]:
        """Batch store with single eviction pass."""
        # Pre-evict if needed
        new_keys = [k for k in entries if k not in self._data]
        if self._max_entries is not None:
            space_needed = len(new_keys) - (self._max_entries - len(self._data))
            for _ in range(max(0, space_needed)):
                self._evict_lru()

        results: List[StorageEntry] = []
        for key, value in entries.items():
            entry = self.put(key, value, generation)
            results.append(entry)
        return results

    def get_stats(self) -> StorageStats:
        """Return detailed storage statistics."""
        return StorageStats(
            entry_count=len(self._data),
            total_writes=self._total_writes,
            total_reads=self._total_reads,
            cache_hits=self._total_reads,  # All reads are "hits" in memory
            cache_misses=0,
        )

    @property
    def eviction_count(self) -> int:
        """Number of entries evicted due to capacity limits."""
        return self._eviction_count

    @property
    def capacity(self) -> Optional[int]:
        """Maximum entries allowed, or None if unlimited."""
        return self._max_entries

    @property
    def utilization(self) -> float:
        """Current utilization as fraction of capacity (0.0 to 1.0)."""
        if self._max_entries is None:
            return 0.0
        return len(self._data) / self._max_entries

    def peek(self, key: str) -> Optional[StorageEntry]:
        """Retrieve an entry WITHOUT updating LRU position.

        Useful for inspection without affecting eviction order.
        """
        return self._data.get(key)

    def oldest_key(self) -> Optional[str]:
        """Return the least recently used key, or None if empty."""
        if not self._data:
            return None
        return next(iter(self._data))

    def newest_key(self) -> Optional[str]:
        """Return the most recently used key, or None if empty."""
        if not self._data:
            return None
        return next(reversed(self._data))

    def _evict_lru(self) -> Optional[str]:
        """Evict the least recently used entry.

        Returns:
            The evicted key, or None if store was empty.
        """
        if not self._data:
            return None
        key, _ = self._data.popitem(last=False)
        self._eviction_count += 1
        return key
