"""MemoCache — caches computed values with generation tracking."""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

class MemoCache:
    """Cache for computed node values with generation tracking."""
    def __init__(self) -> None:
        self._entries: Dict[str, Tuple[Any, int, Dict[str, int]]] = {}

    def store(self, node_id: str, value: Any, dep_generations: Dict[str, int]) -> int:
        old_entry = self._entries.get(node_id)
        if old_entry is not None:
            old_value, old_gen, _ = old_entry
            if value == old_value:
                self._entries[node_id] = (value, old_gen, dep_generations)
                return old_gen
            else:
                new_gen = old_gen + 1
                self._entries[node_id] = (value, new_gen, dep_generations)
                return new_gen
        else:
            self._entries[node_id] = (value, 1, dep_generations)
            return 1

    def get(self, node_id: str) -> Optional[Any]:
        entry = self._entries.get(node_id)
        return entry[0] if entry else None

    def get_generation(self, node_id: str) -> int:
        entry = self._entries.get(node_id)
        return entry[1] if entry else 0

    def invalidate(self, node_id: str) -> None:
        self._entries.pop(node_id, None)

    def clear(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)
