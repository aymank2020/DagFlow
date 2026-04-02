"""Cache invariant tests."""
from dagflow.memo.cache import MemoCache

def test_store_and_get():
    c = MemoCache()
    c.store("n1", 42, {"dep": 1})
    assert c.get("n1") == 42

def test_generation_bumps_on_change():
    c = MemoCache()
    g1 = c.store("n1", 10, {"d": 1})
    g2 = c.store("n1", 20, {"d": 2})
    assert g2 > g1

def test_generation_stable_same_value():
    c = MemoCache()
    g1 = c.store("n1", 10, {"d": 1})
    g2 = c.store("n1", 10, {"d": 2})
    assert g2 == g1

def test_invalidate():
    c = MemoCache()
    c.store("n1", 42, {})
    c.invalidate("n1")
    assert c.get("n1") is None
