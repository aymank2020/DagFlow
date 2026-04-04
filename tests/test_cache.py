"""Qualitative tests for MemoCache — caching invariants.

Tests verify:
- Stored values are retrievable.
- Generation advances only on value change.
- Validity check detects stale entries.
"""

import pytest
from dagflow.memo.cache import MemoCache


class TestMemoCache:
    """Tests for cache behavior."""

    def test_store_and_retrieve(self):
        cache = MemoCache()
        cache.store("n1", 42, {"dep": 1})
        assert cache.get("n1") == 42

    def test_generation_advances_on_new_value(self):
        cache = MemoCache()
        gen1 = cache.store("n1", 10, {"d": 1})
        gen2 = cache.store("n1", 20, {"d": 2})
        assert gen2 > gen1

    def test_generation_stable_on_same_value(self):
        cache = MemoCache()
        gen1 = cache.store("n1", 10, {"d": 1})
        gen2 = cache.store("n1", 10, {"d": 2})
        assert gen2 == gen1

    def test_validity_true_when_deps_unchanged(self):
        cache = MemoCache()
        cache.store("n1", 99, {"a": 3, "b": 5})
        assert cache.is_valid("n1", {"a": 3, "b": 5}) is True

    def test_validity_false_when_dep_advanced(self):
        cache = MemoCache()
        cache.store("n1", 99, {"a": 3, "b": 5})
        assert cache.is_valid("n1", {"a": 4, "b": 5}) is False

    def test_validity_false_for_missing_entry(self):
        cache = MemoCache()
        assert cache.is_valid("nonexistent", {"a": 1}) is False

    def test_invalidate_removes_entry(self):
        cache = MemoCache()
        cache.store("n1", 42, {})
        cache.invalidate("n1")
        assert cache.get("n1") is None
        assert cache.get_generation("n1") == 0

    def test_clear_removes_all(self):
        cache = MemoCache()
        cache.store("a", 1, {})
        cache.store("b", 2, {})
        cache.clear()
        assert cache.size == 0
