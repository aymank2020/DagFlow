"""Pre/post computation hooks and event system.

Provides a hook manager that allows registering callbacks at various
points in the computation lifecycle: before/after node computation,
on invalidation, on error, and on graph structure changes.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode, InputNode)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set


class HookPhase(Enum):
    """Phases in the computation lifecycle where hooks can fire."""

    PRE_COMPUTE = auto()       # Before a node computes
    POST_COMPUTE = auto()      # After a node computes
    PRE_INVALIDATE = auto()    # Before invalidation propagates
    POST_INVALIDATE = auto()   # After invalidation completes
    ON_ERROR = auto()          # When computation raises an exception
    ON_INPUT_CHANGE = auto()   # When an input value is set
    ON_NODE_ADDED = auto()     # When a new node is registered
    ON_GRAPH_CLEAR = auto()    # When the graph is cleared


@dataclass
class HookContext:
    """Context passed to hook callbacks.

    Provides information about the current computation state
    when a hook fires.

    Attributes:
        phase: Which lifecycle phase triggered this hook.
        node_id: The node involved (if applicable).
        value: The computed/new value (if applicable).
        old_value: The previous value (if applicable).
        elapsed: Computation time in seconds (for POST_COMPUTE).
        error: Exception instance (for ON_ERROR phase).
        metadata: Additional context-specific data.
        cancelled: Set to True by a hook to cancel the operation.
    """

    phase: HookPhase
    node_id: str = ""
    value: Any = None
    old_value: Any = None
    elapsed: float = 0.0
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False

    def cancel(self) -> None:
        """Cancel the current operation (only effective in PRE_ phases)."""
        self.cancelled = True


# Type alias for hook callbacks
HookCallback = Callable[[HookContext], None]


@dataclass
class HookRegistration:
    """A registered hook callback.

    Attributes:
        hook_id: Unique identifier for this registration.
        phase: The phase this hook fires on.
        callback: The function to call.
        priority: Execution order (lower = earlier).
        node_filter: If set, only fire for these node IDs.
        enabled: Whether this hook is currently active.
        fire_count: Number of times this hook has fired.
    """

    hook_id: str
    phase: HookPhase
    callback: HookCallback
    priority: int = 0
    node_filter: Optional[Set[str]] = None
    enabled: bool = True
    fire_count: int = 0


class HookManager:
    """Manages computation lifecycle hooks.

    Allows registering callbacks at various phases of the computation
    lifecycle. Hooks are executed in priority order and can optionally
    filter by node ID.

    Usage:
        hooks = HookManager()
        hooks.on(HookPhase.POST_COMPUTE, lambda ctx: print(f"{ctx.node_id} = {ctx.value}"))
        hooks.on(HookPhase.ON_ERROR, error_handler, priority=0)

        # Fire hooks during computation
        ctx = HookContext(phase=HookPhase.PRE_COMPUTE, node_id="sum")
        hooks.fire(ctx)
    """

    def __init__(self) -> None:
        self._hooks: Dict[HookPhase, List[HookRegistration]] = defaultdict(list)
        self._counter: int = 0
        self._total_fires: int = 0
        self._suppressed: bool = False

    @property
    def total_registrations(self) -> int:
        """Total number of registered hooks across all phases."""
        return sum(len(hooks) for hooks in self._hooks.values())

    @property
    def total_fires(self) -> int:
        """Total number of times hooks have been fired."""
        return self._total_fires

    def on(
        self,
        phase: HookPhase,
        callback: HookCallback,
        priority: int = 10,
        node_filter: Optional[Set[str]] = None,
    ) -> str:
        """Register a hook callback for a lifecycle phase.

        Args:
            phase: The phase to hook into.
            callback: Function to call when the phase fires.
            priority: Execution order (lower = earlier). Default 10.
            node_filter: Optional set of node IDs to filter on.

        Returns:
            Hook ID for later removal.
        """
        self._counter += 1
        hook_id = f"hook_{self._counter}"

        registration = HookRegistration(
            hook_id=hook_id,
            phase=phase,
            callback=callback,
            priority=priority,
            node_filter=node_filter,
        )

        self._hooks[phase].append(registration)
        # Keep sorted by priority
        self._hooks[phase].sort(key=lambda h: h.priority)

        return hook_id

    def off(self, hook_id: str) -> bool:
        """Remove a hook by its ID.

        Returns:
            True if the hook was found and removed.
        """
        for phase_hooks in self._hooks.values():
            for i, reg in enumerate(phase_hooks):
                if reg.hook_id == hook_id:
                    phase_hooks.pop(i)
                    return True
        return False

    def fire(self, context: HookContext) -> HookContext:
        """Fire all hooks registered for the context's phase.

        Hooks are executed in priority order. If a PRE_ hook sets
        context.cancelled = True, subsequent hooks still fire but
        the caller should check context.cancelled.

        Args:
            context: The hook context with phase and data.

        Returns:
            The (possibly modified) context after all hooks have fired.
        """
        if self._suppressed:
            return context

        hooks = self._hooks.get(context.phase, [])
        self._total_fires += 1

        for registration in hooks:
            if not registration.enabled:
                continue

            # Apply node filter
            if (
                registration.node_filter
                and context.node_id
                and context.node_id not in registration.node_filter
            ):
                continue

            registration.callback(context)
            registration.fire_count += 1

        return context

    def enable(self, hook_id: str) -> bool:
        """Enable a previously disabled hook."""
        reg = self._find_registration(hook_id)
        if reg:
            reg.enabled = True
            return True
        return False

    def disable(self, hook_id: str) -> bool:
        """Disable a hook without removing it."""
        reg = self._find_registration(hook_id)
        if reg:
            reg.enabled = False
            return True
        return False

    def suppress_all(self) -> None:
        """Temporarily suppress all hook firing."""
        self._suppressed = True

    def unsuppress_all(self) -> None:
        """Resume hook firing after suppression."""
        self._suppressed = False

    def clear_phase(self, phase: HookPhase) -> int:
        """Remove all hooks for a specific phase.

        Returns:
            Number of hooks removed.
        """
        count = len(self._hooks.get(phase, []))
        self._hooks[phase] = []
        return count

    def clear_all(self) -> int:
        """Remove all registered hooks.

        Returns:
            Total hooks removed.
        """
        total = self.total_registrations
        self._hooks.clear()
        return total

    def get_phase_hooks(self, phase: HookPhase) -> List[HookRegistration]:
        """Get all registrations for a phase (read-only copy)."""
        return list(self._hooks.get(phase, []))

    def get_stats(self) -> Dict[str, Any]:
        """Get hook system statistics."""
        return {
            "total_registrations": self.total_registrations,
            "total_fires": self._total_fires,
            "by_phase": {
                phase.name: len(hooks)
                for phase, hooks in self._hooks.items()
            },
            "suppressed": self._suppressed,
        }

    def _find_registration(self, hook_id: str) -> Optional[HookRegistration]:
        """Find a registration by hook ID."""
        for phase_hooks in self._hooks.values():
            for reg in phase_hooks:
                if reg.hook_id == hook_id:
                    return reg
        return None
