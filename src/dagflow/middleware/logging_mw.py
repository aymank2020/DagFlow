"""Built-in logging middleware for computation tracing.

Provides a configurable logging middleware that records computation
events, timing information, and value changes. Integrates with the
hook system to capture events at all lifecycle phases.

This module depends on:
- middleware.hooks (HookManager, HookPhase, HookContext)
- middleware.plugins (Plugin, PluginMetadata)
- core.graph (ComputeGraph)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from dagflow.core.graph import ComputeGraph
from dagflow.middleware.hooks import HookContext, HookManager, HookPhase
from dagflow.middleware.plugins import Plugin, PluginMetadata, PluginState


class LogLevel(IntEnum):
    """Log severity levels."""

    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


@dataclass
class LogEntry:
    """A single log entry from the middleware.

    Attributes:
        level: Severity level.
        phase: The computation phase that generated this entry.
        node_id: The node involved.
        message: Human-readable log message.
        timestamp: When the entry was created.
        elapsed: Computation time (for POST_COMPUTE entries).
        value: The computed value (if capture_values is enabled).
        extra: Additional context data.
    """

    level: LogLevel
    phase: HookPhase
    node_id: str
    message: str
    timestamp: float = field(default_factory=time.monotonic)
    elapsed: float = 0.0
    value: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


class LoggingMiddleware(Plugin):
    """Logging middleware that captures computation events.

    Records detailed logs of graph computation activity including
    timing, value changes, errors, and invalidation events.

    Usage:
        logging_mw = LoggingMiddleware(
            level=LogLevel.INFO,
            capture_values=True,
            max_entries=1000,
        )
        registry.register(logging_mw)
        registry.activate("dagflow.logging")

        # After computation
        for entry in logging_mw.entries:
            print(f"[{entry.level.name}] {entry.message}")
    """

    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        capture_values: bool = False,
        max_entries: int = 1000,
        node_filter: Optional[Set[str]] = None,
        output_fn: Optional[Callable[[LogEntry], None]] = None,
    ) -> None:
        """Initialize the logging middleware.

        Args:
            level: Minimum log level to capture.
            capture_values: Whether to store computed values in logs.
            max_entries: Maximum log entries to retain.
            node_filter: If set, only log events for these nodes.
            output_fn: Optional callback for real-time log output.
        """
        metadata = PluginMetadata(
            name="dagflow.logging",
            version="1.0.0",
            description="Built-in computation logging middleware",
            tags=["logging", "debugging", "monitoring"],
        )
        super().__init__(metadata)

        self._level = level
        self._capture_values = capture_values
        self._max_entries = max_entries
        self._node_filter = node_filter
        self._output_fn = output_fn
        self._entries: Deque[LogEntry] = deque(maxlen=max_entries)
        self._timing: Dict[str, float] = {}
        self._compute_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)

    @property
    def entries(self) -> List[LogEntry]:
        """All captured log entries."""
        return list(self._entries)

    @property
    def entry_count(self) -> int:
        """Number of entries currently stored."""
        return len(self._entries)

    @property
    def error_count(self) -> int:
        """Total errors logged."""
        return sum(self._error_counts.values())

    def on_load(self, graph: ComputeGraph, hooks: HookManager) -> None:
        """Register hooks for all relevant phases."""
        super().on_load(graph, hooks)

        # Register hooks for each phase we care about
        phases_and_handlers = [
            (HookPhase.PRE_COMPUTE, self._on_pre_compute),
            (HookPhase.POST_COMPUTE, self._on_post_compute),
            (HookPhase.ON_ERROR, self._on_error),
            (HookPhase.ON_INPUT_CHANGE, self._on_input_change),
            (HookPhase.POST_INVALIDATE, self._on_post_invalidate),
        ]

        for phase, handler in phases_and_handlers:
            hook_id = hooks.on(phase, handler, priority=100)
            self.register_hook(hook_id)

    def on_activate(self) -> None:
        """Start logging."""
        self._log(
            LogLevel.INFO,
            HookPhase.ON_NODE_ADDED,
            "__system__",
            "Logging middleware activated",
        )

    def on_deactivate(self) -> None:
        """Stop logging."""
        self._log(
            LogLevel.INFO,
            HookPhase.ON_NODE_ADDED,
            "__system__",
            "Logging middleware deactivated",
        )

    def get_entries_by_level(self, level: LogLevel) -> List[LogEntry]:
        """Filter entries by log level."""
        return [e for e in self._entries if e.level == level]

    def get_entries_by_node(self, node_id: str) -> List[LogEntry]:
        """Get all log entries for a specific node."""
        return [e for e in self._entries if e.node_id == node_id]

    def get_slow_computations(self, threshold: float = 0.01) -> List[LogEntry]:
        """Find computations that exceeded a time threshold.

        Args:
            threshold: Minimum seconds to be considered slow.

        Returns:
            List of log entries for slow computations.
        """
        return [
            e
            for e in self._entries
            if e.phase == HookPhase.POST_COMPUTE and e.elapsed >= threshold
        ]

    def get_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary per node.

        Returns:
            Dict of {node_id: {"count": N, "total": T, "avg": A}}.
        """
        summary: Dict[str, Dict[str, float]] = {}
        timing_data: Dict[str, List[float]] = defaultdict(list)

        for entry in self._entries:
            if entry.phase == HookPhase.POST_COMPUTE and entry.elapsed > 0:
                timing_data[entry.node_id].append(entry.elapsed)

        for node_id, times in timing_data.items():
            summary[node_id] = {
                "count": float(len(times)),
                "total": sum(times),
                "avg": sum(times) / len(times),
                "max": max(times),
            }

        return summary

    def clear(self) -> int:
        """Clear all log entries. Returns count cleared."""
        count = len(self._entries)
        self._entries.clear()
        return count

    def _on_pre_compute(self, ctx: HookContext) -> None:
        """Hook: before node computation."""
        if not self._should_log(ctx.node_id):
            return

        self._timing[ctx.node_id] = time.monotonic()
        self._log(
            LogLevel.DEBUG,
            ctx.phase,
            ctx.node_id,
            f"Computing node '{ctx.node_id}'",
        )

    def _on_post_compute(self, ctx: HookContext) -> None:
        """Hook: after node computation."""
        if not self._should_log(ctx.node_id):
            return

        start_time = self._timing.pop(ctx.node_id, 0.0)
        elapsed = time.monotonic() - start_time if start_time else 0.0
        self._compute_counts[ctx.node_id] += 1

        value = ctx.value if self._capture_values else None
        self._log(
            LogLevel.DEBUG,
            ctx.phase,
            ctx.node_id,
            f"Computed '{ctx.node_id}' in {elapsed:.4f}s",
            elapsed=elapsed,
            value=value,
        )

    def _on_error(self, ctx: HookContext) -> None:
        """Hook: computation error."""
        self._error_counts[ctx.node_id] += 1
        error_msg = str(ctx.error) if ctx.error else "Unknown error"
        self._log(
            LogLevel.ERROR,
            ctx.phase,
            ctx.node_id,
            f"Error in '{ctx.node_id}': {error_msg}",
            extra={"error_type": type(ctx.error).__name__ if ctx.error else ""},
        )

    def _on_input_change(self, ctx: HookContext) -> None:
        """Hook: input value changed."""
        if not self._should_log(ctx.node_id):
            return

        self._log(
            LogLevel.INFO,
            ctx.phase,
            ctx.node_id,
            f"Input '{ctx.node_id}' changed",
            value=ctx.value if self._capture_values else None,
        )

    def _on_post_invalidate(self, ctx: HookContext) -> None:
        """Hook: invalidation completed."""
        count = ctx.metadata.get("invalidated_count", 0)
        self._log(
            LogLevel.DEBUG,
            ctx.phase,
            ctx.node_id,
            f"Invalidation from '{ctx.node_id}' affected {count} nodes",
            extra={"invalidated_count": count},
        )

    def _should_log(self, node_id: str) -> bool:
        """Check if we should log events for this node."""
        if self._node_filter is None:
            return True
        return node_id in self._node_filter

    def _log(
        self,
        level: LogLevel,
        phase: HookPhase,
        node_id: str,
        message: str,
        elapsed: float = 0.0,
        value: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create and store a log entry."""
        if level < self._level:
            return

        entry = LogEntry(
            level=level,
            phase=phase,
            node_id=node_id,
            message=message,
            elapsed=elapsed,
            value=value,
            extra=extra or {},
        )
        self._entries.append(entry)

        if self._output_fn:
            self._output_fn(entry)
