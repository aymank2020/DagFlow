"""MemoCache — caches computed values with generation tracking.

Provides storage for node results, generation counters for compute nodes,
and staleness detection. The cache is the single source of truth for
"what generation is this compute node at?" — used by the invalidator
for early-cutoff decisions.

This module depends on:
- core.node (ComputeNode) for type checks only
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class MemoCache:
    """Cache for computed node values with generation tracking.

    Each entry stores: (value, generation, dep_generations_snapshot).
    Generation is bumped every time a node produces a NEW value (not same as old).
    """

    def __init__(self) -> None:
        # node_id -> (value, generation, {dep_id: dep_gen_at_compute_time})
        self._entries: Dict[str, Tuple[Any, int, Dict[str, int]]] = {}

    def store(self, node_id: str, value: Any, dep_generations: Dict[str, int]) -> int:
        """Store a computed value. Bumps generation if value changed.

        Args:
            node_id: The compute node ID.
            value: The newly computed value.
            dep_generations: Snapshot of dependency generations at compute time.

        Returns:
            The new generation of this node.
        """
        old_entry = self._entries.get(node_id)
        if old_entry is not None:
            old_value, old_gen, _ = old_entry
            if value == old_value:
                # Same value — keep generation, update dep snapshot
                self._entries[node_id] = (value, old_gen, dict(dep_generations))
                return old_gen
            else:
                new_gen = old_gen + 1
                self._entries[node_id] = (value, new_gen, dict(dep_generations))
                return new_gen
        else:
            # First computation
            self._entries[node_id] = (value, 1, dict(dep_generations))
            return 1

    def get(self, node_id: str) -> Optional[Any]:
        """Retrieve cached value, or None if not cached."""
        entry = self._entries.get(node_id)
        return entry[0] if entry else None

    def get_generation(self, node_id: str) -> int:
        """Get current generation of a node. Returns 0 if never computed."""
        entry = self._entries.get(node_id)
        return entry[1] if entry else 0

    def get_dep_snapshot(self, node_id: str) -> Dict[str, int]:
        """Get the dependency generation snapshot from last computation."""
        entry = self._entries.get(node_id)
        return dict(entry[2]) if entry else {}

    def is_valid(self, node_id: str, current_dep_generations: Dict[str, int]) -> bool:
        """Check if cached value is still valid given current dependency generations.

        Args:
            node_id: The node to check.
            current_dep_generations: Current {dep_id: generation} map.

        Returns:
            True if cache is still valid (no dependency has advanced).
        """
        entry = self._entries.get(node_id)
        if entry is None:
            return False
        _, _, stored_deps = entry
        return stored_deps == current_dep_generations

    def invalidate(self, node_id: str) -> None:
        """Remove a node's cache entry entirely."""
        self._entries.pop(node_id, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._entries)
