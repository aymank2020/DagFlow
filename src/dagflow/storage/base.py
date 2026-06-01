"""Abstract storage backend interface.

Defines the contract that all storage backends must implement,
providing a uniform API for persisting and retrieving graph state
regardless of the underlying storage mechanism.

This module depends on:
- core.node (InputNode, ComputeNode) for type context
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set


class StorageError(Exception):
    """Base exception for storage operations."""


class StorageKeyError(StorageError):
    """Raised when a requested key doesn't exist."""


class StorageCapacityError(StorageError):
    """Raised when storage capacity is exceeded."""


@dataclass(frozen=True)
class StorageEntry:
    """A single entry in the storage backend.

    Attributes:
        key: Unique identifier (typically node_id).
        value: The stored value (serialized or raw).
        generation: Version counter for this entry.
        timestamp: When this entry was last written.
        metadata: Optional extra information about the entry.
        checksum: Optional integrity checksum.
    """

    key: str
    value: Any
    generation: int = 0
    timestamp: float = field(default_factory=time.monotonic)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None

    @property
    def age(self) -> float:
        """Seconds since this entry was written."""
        return time.monotonic() - self.timestamp


@dataclass
class StorageStats:
    """Statistics about storage usage.

    Attributes:
        entry_count: Number of entries stored.
        total_writes: Total write operations performed.
        total_reads: Total read operations performed.
        cache_hits: Number of reads served from cache.
        cache_misses: Number of reads requiring backend access.
    """

    entry_count: int = 0
    total_writes: int = 0
    total_reads: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction (0.0 to 1.0)."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    All storage backends must implement this interface to be usable
    with the DAG computation engine. The interface supports basic
    CRUD operations, batch operations, and iteration.

    Implementations should be thread-safe for concurrent access.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[StorageEntry]:
        """Retrieve an entry by key.

        Args:
            key: The entry identifier.

        Returns:
            StorageEntry if found, None otherwise.
        """

    @abstractmethod
    def put(self, key: str, value: Any, generation: int = 0) -> StorageEntry:
        """Store or update an entry.

        Args:
            key: The entry identifier.
            value: The value to store.
            generation: Version counter for this write.

        Returns:
            The created/updated StorageEntry.
        """

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an entry by key.

        Args:
            key: The entry identifier.

        Returns:
            True if the entry existed and was deleted.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in storage.

        Args:
            key: The entry identifier.

        Returns:
            True if the key exists.
        """

    @abstractmethod
    def keys(self) -> List[str]:
        """Return all keys in storage."""

    @abstractmethod
    def clear(self) -> int:
        """Remove all entries.

        Returns:
            Number of entries removed.
        """

    @abstractmethod
    def size(self) -> int:
        """Return the number of entries in storage."""

    def get_many(self, keys: List[str]) -> Dict[str, Optional[StorageEntry]]:
        """Retrieve multiple entries at once.

        Default implementation calls get() for each key.
        Backends may override for batch optimization.

        Args:
            keys: List of entry identifiers.

        Returns:
            Dict mapping key -> StorageEntry (or None if not found).
        """
        return {key: self.get(key) for key in keys}

    def put_many(
        self, entries: Dict[str, Any], generation: int = 0
    ) -> List[StorageEntry]:
        """Store multiple entries at once.

        Default implementation calls put() for each entry.
        Backends may override for batch optimization.

        Args:
            entries: Dict of {key: value} to store.
            generation: Version counter for all entries.

        Returns:
            List of created StorageEntries.
        """
        results: List[StorageEntry] = []
        for key, value in entries.items():
            entry = self.put(key, value, generation)
            results.append(entry)
        return results

    def delete_many(self, keys: List[str]) -> int:
        """Delete multiple entries at once.

        Args:
            keys: List of entry identifiers to delete.

        Returns:
            Number of entries actually deleted.
        """
        count = 0
        for key in keys:
            if self.delete(key):
                count += 1
        return count

    def get_stats(self) -> StorageStats:
        """Return storage statistics.

        Default implementation returns basic stats.
        Backends should override for accurate tracking.
        """
        return StorageStats(entry_count=self.size())

    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of all stored values.

        Returns:
            Dict of {key: value} for all entries.
        """
        result: Dict[str, Any] = {}
        for key in self.keys():
            entry = self.get(key)
            if entry is not None:
                result[key] = entry.value
        return result

    def restore(self, snapshot: Dict[str, Any], generation: int = 0) -> int:
        """Restore storage from a snapshot.

        Clears existing data and loads from the snapshot.

        Args:
            snapshot: Dict of {key: value} to restore.
            generation: Generation to assign to all entries.

        Returns:
            Number of entries restored.
        """
        self.clear()
        for key, value in snapshot.items():
            self.put(key, value, generation)
        return len(snapshot)
