"""Node types for the computation DAG."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, List

class NodeState(Enum):
    CLEAN = auto()
    DIRTY = auto()

@dataclass
class InputNode:
    """A leaf node whose value is set externally."""
    node_id: str
    value: Any = None
    generation: int = 0
    dependents: List[str] = field(default_factory=list)

    def set(self, new_value: Any) -> int:
        if new_value != self.value:
            self.value = new_value
            self.generation += 1
        return self.generation

    @property
    def is_input(self) -> bool:
        return True

@dataclass
class ComputeNode:
    """A derived node that computes its value from dependencies."""
    node_id: str
    func: Callable[[dict], Any] = field(default=lambda d: None)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    state: NodeState = NodeState.DIRTY
    cached_value: Any = None

    def mark_dirty(self) -> None:
        if self.state == NodeState.CLEAN:
            self.state = NodeState.DIRTY

    def mark_clean(self, value: Any) -> None:
        self.cached_value = value
        self.state = NodeState.CLEAN

    @property
    def is_input(self) -> bool:
        return False
