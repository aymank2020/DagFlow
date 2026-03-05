"""Node types for the computation DAG."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List

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
