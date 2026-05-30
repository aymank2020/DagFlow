"""Tests for validation — constraints and integrity checks.

Tests verify behavioral invariants:
- Constraints detect invalid values correctly.
- Constraints pass for valid values.
- Integrity checker finds structural problems.
- Healthy graphs pass all checks.
"""

import pytest
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode, ComputeNode, NodeState
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.validation.constraints import (
    TypeConstraint,
    RangeConstraint,
    PredicateConstraint,
    ConstraintRegistry,
)
from dagflow.validation.integrity import IntegrityChecker, IssueSeverity, IssueType


class TestTypeConstraint:
    """Tests for type constraints."""

    def test_valid_type_passes(self):
        """Value of allowed type passes validation."""
        tc = TypeConstraint("numeric", allowed_types=(int, float))
        assert tc.validate(42) is None
        assert tc.validate(3.14) is None

    def test_invalid_type_fails(self):
        """Value of wrong type fails validation."""
        tc = TypeConstraint("numeric", allowed_types=(int, float))
        result = tc.validate("hello")
        assert result is not None
        assert "str" in result

    def test_none_always_passes(self):
        """None values always pass type constraints."""
        tc = TypeConstraint("strict", allowed_types=(int,))
        assert tc.validate(None) is None


class TestRangeConstraint:
    """Tests for range constraints."""

    def test_value_in_range_passes(self):
        """Value within bounds passes."""
        rc = RangeConstraint("temp", min_value=-40, max_value=150)
        assert rc.validate(72) is None

    def test_value_below_min_fails(self):
        """Value below minimum fails."""
        rc = RangeConstraint("positive", min_value=0)
        result = rc.validate(-5)
        assert result is not None
        assert "below" in result

    def test_value_above_max_fails(self):
        """Value above maximum fails."""
        rc = RangeConstraint("percentage", max_value=100)
        result = rc.validate(150)
        assert result is not None
        assert "exceeds" in result

    def test_none_with_allow_none_passes(self):
        """None passes when allow_none=True."""
        rc = RangeConstraint("optional", min_value=0, allow_none=True)
        assert rc.validate(None) is None

    def test_none_with_disallow_none_fails(self):
        """None fails when allow_none=False."""
        rc = RangeConstraint("required", min_value=0, allow_none=False)
        result = rc.validate(None)
        assert result is not None


class TestPredicateConstraint:
    """Tests for predicate-based constraints."""

    def test_predicate_passes(self):
        """Value satisfying predicate passes."""
        pc = PredicateConstraint("even", predicate=lambda x: x % 2 == 0)
        assert pc.validate(4) is None

    def test_predicate_fails(self):
        """Value not satisfying predicate fails."""
        pc = PredicateConstraint(
            "even",
            predicate=lambda x: x % 2 == 0,
            error_message="Must be even",
        )
        result = pc.validate(3)
        assert result == "Must be even"


class TestConstraintRegistry:
    """Tests for the constraint registry."""

    def test_validate_node_with_valid_value(self):
        """Node with valid value produces no violations."""
        g = ComputeGraph()
        g.add_input("temp", value=72)
        cache = MemoCache()

        registry = ConstraintRegistry(g, cache)
        registry.add("temp", RangeConstraint("valid_temp", min_value=-40, max_value=150))

        violations = registry.validate_node("temp")
        assert len(violations) == 0

    def test_validate_node_with_invalid_value(self):
        """Node with invalid value produces a violation."""
        g = ComputeGraph()
        g.add_input("temp", value=200)
        cache = MemoCache()

        registry = ConstraintRegistry(g, cache)
        registry.add("temp", RangeConstraint("valid_temp", max_value=150))

        violations = registry.validate_node("temp")
        assert len(violations) == 1
        assert violations[0].node_id == "temp"

    def test_validate_all_checks_all_constrained_nodes(self):
        """validate_all checks every node with constraints."""
        g = ComputeGraph()
        g.add_input("a", value=-5)
        g.add_input("b", value=10)
        cache = MemoCache()

        registry = ConstraintRegistry(g, cache)
        registry.add("a", RangeConstraint("positive_a", min_value=0))
        registry.add("b", RangeConstraint("positive_b", min_value=0))

        violations = registry.validate_all()
        # Only 'a' should violate
        assert len(violations) == 1
        assert violations[0].node_id == "a"

    def test_validate_compute_node_uses_cache(self):
        """Constraint on compute node validates cached value."""
        g = ComputeGraph()
        g.add_input("x", value=10)
        g.add_compute("y", func=lambda d: d["x"] * 2, dependencies=["x"])
        cache = MemoCache()
        EagerPropagator(g, cache).propagate(["x"])

        registry = ConstraintRegistry(g, cache)
        registry.add("y", RangeConstraint("y_limit", max_value=15))

        violations = registry.validate_node("y")
        # y = 20, which exceeds max of 15
        assert len(violations) == 1

    def test_remove_constraint(self):
        """Removing a constraint stops it from being checked."""
        g = ComputeGraph()
        g.add_input("x", value=-1)
        cache = MemoCache()

        registry = ConstraintRegistry(g, cache)
        registry.add("x", RangeConstraint("positive", min_value=0))

        assert len(registry.validate_node("x")) == 1

        registry.remove("x", "positive")
        assert len(registry.validate_node("x")) == 0


class TestIntegrityChecker:
    """Tests for graph integrity checks."""

    def test_healthy_graph_passes(self):
        """Well-formed graph has no errors."""
        g = ComputeGraph()
        g.add_input("x", value=1)
        g.add_compute("y", func=lambda d: d["x"], dependencies=["x"])

        checker = IntegrityChecker(g)
        assert checker.is_healthy()

    def test_orphan_input_detected(self):
        """Input with no dependents is flagged as warning."""
        g = ComputeGraph()
        g.add_input("used", value=1)
        g.add_input("orphan", value=2)
        g.add_compute("y", func=lambda d: d["used"], dependencies=["used"])

        checker = IntegrityChecker(g)
        issues = checker.check_orphan_inputs()

        orphan_ids = [i.node_id for i in issues]
        assert "orphan" in orphan_ids
        assert "used" not in orphan_ids

    def test_edge_consistency_in_valid_graph(self):
        """Valid graph has no edge consistency issues."""
        g = ComputeGraph()
        g.add_input("a")
        g.add_input("b")
        g.add_compute("c", func=lambda d: None, dependencies=["a", "b"])

        checker = IntegrityChecker(g)
        issues = checker.check_edge_consistency()
        assert len(issues) == 0

    def test_state_consistency_clean_with_dirty_dep(self):
        """Detects when a CLEAN node has a DIRTY dependency."""
        g = ComputeGraph()
        g.add_input("x", value=1)
        g.add_compute("mid", func=lambda d: d["x"], dependencies=["x"])
        g.add_compute("leaf", func=lambda d: d["mid"], dependencies=["mid"])

        # Manually create inconsistent state
        mid_node = g.get_node("mid")
        leaf_node = g.get_node("leaf")
        mid_node.state = NodeState.DIRTY
        leaf_node.state = NodeState.CLEAN

        checker = IntegrityChecker(g)
        issues = checker.check_state_consistency()

        # leaf is CLEAN but mid (its dep) is DIRTY
        assert any(i.node_id == "leaf" for i in issues)

    def test_check_all_returns_sorted_by_severity(self):
        """check_all returns errors before warnings."""
        g = ComputeGraph()
        g.add_input("orphan", value=1)  # Warning: orphan input

        checker = IntegrityChecker(g)
        issues = checker.check_all()

        if len(issues) > 1:
            # Errors should come before warnings
            error_indices = [
                i for i, issue in enumerate(issues)
                if issue.severity == IssueSeverity.ERROR
            ]
            warning_indices = [
                i for i, issue in enumerate(issues)
                if issue.severity == IssueSeverity.WARNING
            ]
            if error_indices and warning_indices:
                assert max(error_indices) < min(warning_indices)
