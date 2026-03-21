"""Node types for the computation DAG.

Each node has a unique ID, a generation counter (incremented on each value change),
and tracks its upstream dependencies and downstream dependents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, List, Optional


class NodeState(Enum):
    """Lifecycle state of a compute node."""

    CLEAN = auto()      # Value is up-to-date
    DIRTY = auto()      # Needs recomputation
    PENDING = auto()    # Scheduled but not yet executed
    COMPUTING = auto()  # Currently being evaluated


@dataclass
class InputNode:
    """A leaf node whose value is set externally.

    Attributes:
        node_id: Unique identifier.
        value: Current value held by this input.
        generation: Monotonically increasing counter; bumped on every set().
        dependents: List of node IDs that read from this input.
    """

    node_id: str
    value: Any = None
    generation: int = 0
    dependents: List[str] = field(default_factory=list)

    def set(self, new_value: Any) -> int:
        """Update value and bump generation. Returns new generation."""
        if new_value != self.value:
            self.value = new_value
            self.generation += 1
        return self.generation

    @property
    def is_input(self) -> bool:
        return True


@dataclass
class ComputeNode:
    """A derived node that computes its value from dependencies.

    Attributes:
        node_id: Unique identifier.
        func: The computation function. Receives a dict of {dep_id: value}.
        dependencies: List of node IDs this node reads from.
        dependents: List of node IDs that read from this node.
        state: Current lifecycle state.
        cached_value: Last computed value (None if never computed).
        last_valid_generation: Dict mapping dep_id -> generation when last computed.
        priority: Scheduling priority (lower = computed first among same-level).
    """

    node_id: str
    func: Callable[[dict], Any] = field(default=lambda d: None)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    state: NodeState = NodeState.DIRTY
    cached_value: Any = None
    last_valid_generation: dict = field(default_factory=dict)
    priority: int = 0

    def mark_dirty(self) -> None:
        """Transition to DIRTY state if currently CLEAN."""
        if self.state == NodeState.CLEAN:
            self.state = NodeState.DIRTY

    def mark_clean(self, value: Any, dep_generations: dict) -> None:
        """Store computed value and record dependency generations."""
        self.cached_value = value
        self.last_valid_generation = dict(dep_generations)
        self.state = NodeState.CLEAN

    def needs_recompute(self, current_generations: dict) -> bool:
        """Check if any dependency generation has advanced since last compute.

        Args:
            current_generations: Dict of {dep_id: current_generation} for all deps.

        Returns:
            True if recomputation is needed.
        """
        if self.state == NodeState.DIRTY:
            return True
        for dep_id in self.dependencies:
            last_gen = self.last_valid_generation.get(dep_id, -1)
            curr_gen = current_generations.get(dep_id, 0)
            if curr_gen > last_gen:
                return True
        return False

    @property
    def is_input(self) -> bool:
        return False
