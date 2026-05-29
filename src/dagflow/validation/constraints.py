"""Constraints — type and value constraints on node values.

Provides a constraint system that can validate node values against
declared rules. Constraints are registered per-node and checked
after propagation to ensure the graph maintains valid state.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode)
- memo.cache (MemoCache)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode
from dagflow.memo.cache import MemoCache


@dataclass
class ConstraintViolation:
    """A detected constraint violation.

    Attributes:
        node_id: The node that violated the constraint.
        constraint_name: Name of the violated constraint.
        message: Human-readable description of the violation.
        actual_value: The value that caused the violation.
    """

    node_id: str
    constraint_name: str
    message: str
    actual_value: Any = None


class Constraint(ABC):
    """Base class for node value constraints."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Constraint identifier."""
        return self._name

    @abstractmethod
    def validate(self, value: Any) -> Optional[str]:
        """Validate a value against this constraint.

        Args:
            value: The value to validate.

        Returns:
            None if valid, or an error message string if invalid.
        """
        ...


class TypeConstraint(Constraint):
    """Constraint that checks the type of a node's value.

    Usage:
        tc = TypeConstraint("must_be_int", allowed_types=(int, float))
        result = tc.validate("hello")  # Returns error message
    """

    def __init__(self, name: str, allowed_types: Tuple[Type, ...]) -> None:
        super().__init__(name)
        self._allowed_types = allowed_types

    @property
    def allowed_types(self) -> Tuple[Type, ...]:
        """The types that are considered valid."""
        return self._allowed_types

    def validate(self, value: Any) -> Optional[str]:
        """Check if value is one of the allowed types.

        None values are always considered valid (use RangeConstraint
        with allow_none=False to reject None).
        """
        if value is None:
            return None
        if not isinstance(value, self._allowed_types):
            type_names = ", ".join(t.__name__ for t in self._allowed_types)
            return (
                f"Expected type(s) ({type_names}), "
                f"got {type(value).__name__}"
            )
        return None


class RangeConstraint(Constraint):
    """Constraint that checks a numeric value is within bounds.

    Usage:
        rc = RangeConstraint("positive", min_value=0)
        result = rc.validate(-5)  # Returns error message
    """

    def __init__(
        self,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_none: bool = True,
    ) -> None:
        super().__init__(name)
        self._min_value = min_value
        self._max_value = max_value
        self._allow_none = allow_none

    @property
    def min_value(self) -> Optional[float]:
        return self._min_value

    @property
    def max_value(self) -> Optional[float]:
        return self._max_value

    def validate(self, value: Any) -> Optional[str]:
        """Check if value is within the specified range."""
        if value is None:
            if not self._allow_none:
                return "Value cannot be None"
            return None

        if not isinstance(value, (int, float)):
            return f"RangeConstraint requires numeric value, got {type(value).__name__}"

        if self._min_value is not None and value < self._min_value:
            return f"Value {value} is below minimum {self._min_value}"

        if self._max_value is not None and value > self._max_value:
            return f"Value {value} exceeds maximum {self._max_value}"

        return None


class PredicateConstraint(Constraint):
    """Constraint using a custom predicate function.

    Usage:
        pc = PredicateConstraint("even", predicate=lambda x: x % 2 == 0)
    """

    def __init__(
        self,
        name: str,
        predicate: Callable[[Any], bool],
        error_message: str = "Predicate check failed",
    ) -> None:
        super().__init__(name)
        self._predicate = predicate
        self._error_message = error_message

    def validate(self, value: Any) -> Optional[str]:
        """Check if value satisfies the predicate."""
        if value is None:
            return None
        if not self._predicate(value):
            return self._error_message
        return None


class ConstraintRegistry:
    """Manages constraints attached to graph nodes.

    Constraints are registered per-node and can be validated individually
    or across the entire graph.

    Usage:
        registry = ConstraintRegistry(graph, cache)
        registry.add("temperature", RangeConstraint("valid_temp", -50, 150))
        violations = registry.validate_all()
    """

    def __init__(self, graph: ComputeGraph, cache: MemoCache) -> None:
        self._graph = graph
        self._cache = cache
        self._constraints: Dict[str, List[Constraint]] = {}

    def add(self, node_id: str, constraint: Constraint) -> None:
        """Attach a constraint to a node.

        Args:
            node_id: The node to constrain.
            constraint: The constraint to attach.

        Raises:
            KeyError: If node_id doesn't exist in the graph.
        """
        # Verify node exists
        self._graph.get_node(node_id)

        if node_id not in self._constraints:
            self._constraints[node_id] = []
        self._constraints[node_id].append(constraint)

    def remove(self, node_id: str, constraint_name: str) -> bool:
        """Remove a named constraint from a node.

        Returns True if the constraint was found and removed.
        """
        if node_id not in self._constraints:
            return False
        before = len(self._constraints[node_id])
        self._constraints[node_id] = [
            c for c in self._constraints[node_id] if c.name != constraint_name
        ]
        return len(self._constraints[node_id]) < before

    def get_constraints(self, node_id: str) -> List[Constraint]:
        """Get all constraints for a node."""
        return list(self._constraints.get(node_id, []))

    def validate_node(self, node_id: str) -> List[ConstraintViolation]:
        """Validate a single node against its constraints.

        Args:
            node_id: The node to validate.

        Returns:
            List of violations (empty if all constraints pass).
        """
        constraints = self._constraints.get(node_id, [])
        if not constraints:
            return []

        # Get current value
        node = self._graph.get_node(node_id)
        if isinstance(node, InputNode):
            value = node.value
        elif isinstance(node, ComputeNode):
            value = self._cache.get(node_id)
        else:
            return []

        violations: List[ConstraintViolation] = []
        for constraint in constraints:
            error = constraint.validate(value)
            if error is not None:
                violations.append(
                    ConstraintViolation(
                        node_id=node_id,
                        constraint_name=constraint.name,
                        message=error,
                        actual_value=value,
                    )
                )

        return violations

    def validate_all(self) -> List[ConstraintViolation]:
        """Validate all constrained nodes.

        Returns:
            List of all violations across the graph.
        """
        all_violations: List[ConstraintViolation] = []
        for node_id in self._constraints:
            all_violations.extend(self.validate_node(node_id))
        return all_violations

    @property
    def constrained_nodes(self) -> Set[str]:
        """Set of node IDs that have constraints attached."""
        return set(self._constraints.keys())
