"""Tests for the middleware and plugin system.

Tests cover:
- Hook registration and firing
- Hook phases and context
- Plugin lifecycle management
- Plugin registry operations
- Logging middleware functionality
"""

from __future__ import annotations

from typing import Any, List

import pytest

from dagflow.core.graph import ComputeGraph
from dagflow.middleware.hooks import HookContext, HookManager, HookPhase
from dagflow.middleware.plugins import (
    Plugin,
    PluginMetadata,
    PluginRegistry,
    PluginState,
)
from dagflow.middleware.logging_mw import LoggingMiddleware, LogLevel


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def hook_manager() -> HookManager:
    """Create a fresh hook manager."""
    return HookManager()


@pytest.fixture
def simple_graph() -> ComputeGraph:
    """Create a simple graph for plugin testing."""
    g = ComputeGraph()
    g.add_input("x", value=1)
    g.add_compute("double", lambda d: d["x"] * 2, ["x"])
    return g


@pytest.fixture
def registry(simple_graph: ComputeGraph, hook_manager: HookManager) -> PluginRegistry:
    """Create a plugin registry."""
    return PluginRegistry(simple_graph, hook_manager)


class SamplePlugin(Plugin):
    """A test plugin that tracks lifecycle calls."""

    def __init__(self, name: str = "sample") -> None:
        metadata = PluginMetadata(name=name, version="1.0.0", tags=["test"])
        super().__init__(metadata)
        self.load_called = False
        self.activate_called = False
        self.deactivate_called = False
        self.unload_called = False

    def on_load(self, graph, hooks) -> None:
        super().on_load(graph, hooks)
        self.load_called = True

    def on_activate(self) -> None:
        self.activate_called = True

    def on_deactivate(self) -> None:
        self.deactivate_called = True

    def on_unload(self) -> None:
        super().on_unload()
        self.unload_called = True


class DependentPlugin(Plugin):
    """A plugin that depends on SamplePlugin."""

    def __init__(self) -> None:
        metadata = PluginMetadata(
            name="dependent",
            version="1.0.0",
            dependencies=["sample"],
        )
        super().__init__(metadata)


# ─── Hook Manager Tests ────────────────────────────────────────────────


class TestHookManager:
    """Tests for HookManager."""

    def test_register_hook(self, hook_manager: HookManager) -> None:
        """Registering a hook returns an ID."""
        hook_id = hook_manager.on(HookPhase.PRE_COMPUTE, lambda ctx: None)
        assert hook_id.startswith("hook_")
        assert hook_manager.total_registrations == 1

    def test_fire_hook(self, hook_manager: HookManager) -> None:
        """Firing a hook calls registered callbacks."""
        received: List[HookContext] = []
        hook_manager.on(HookPhase.PRE_COMPUTE, lambda ctx: received.append(ctx))

        ctx = HookContext(phase=HookPhase.PRE_COMPUTE, node_id="test")
        hook_manager.fire(ctx)

        assert len(received) == 1
        assert received[0].node_id == "test"

    def test_priority_ordering(self, hook_manager: HookManager) -> None:
        """Hooks fire in priority order (lower first)."""
        order: List[int] = []
        hook_manager.on(HookPhase.PRE_COMPUTE, lambda ctx: order.append(2), priority=20)
        hook_manager.on(HookPhase.PRE_COMPUTE, lambda ctx: order.append(1), priority=10)
        hook_manager.on(HookPhase.PRE_COMPUTE, lambda ctx: order.append(3), priority=30)

        hook_manager.fire(HookContext(phase=HookPhase.PRE_COMPUTE))
        assert order == [1, 2, 3]

    def test_node_filter(self, hook_manager: HookManager) -> None:
        """Node filter restricts which nodes trigger the hook."""
        received: List[str] = []
        hook_manager.on(
            HookPhase.POST_COMPUTE,
            lambda ctx: received.append(ctx.node_id),
            node_filter={"target_node"},
        )

        hook_manager.fire(HookContext(phase=HookPhase.POST_COMPUTE, node_id="other"))
        hook_manager.fire(HookContext(phase=HookPhase.POST_COMPUTE, node_id="target_node"))

        assert received == ["target_node"]

    def test_remove_hook(self, hook_manager: HookManager) -> None:
        """Removed hooks no longer fire."""
        received: List[str] = []
        hook_id = hook_manager.on(
            HookPhase.PRE_COMPUTE, lambda ctx: received.append("fired")
        )

        hook_manager.fire(HookContext(phase=HookPhase.PRE_COMPUTE))
        assert len(received) == 1

        hook_manager.off(hook_id)
        hook_manager.fire(HookContext(phase=HookPhase.PRE_COMPUTE))
        assert len(received) == 1  # No new fires

    def test_disable_enable_hook(self, hook_manager: HookManager) -> None:
        """Disabled hooks don't fire; re-enabled hooks do."""
        received: List[str] = []
        hook_id = hook_manager.on(
            HookPhase.PRE_COMPUTE, lambda ctx: received.append("fired")
        )

        hook_manager.disable(hook_id)
        hook_manager.fire(HookContext(phase=HookPhase.PRE_COMPUTE))
        assert len(received) == 0

        hook_manager.enable(hook_id)
        hook_manager.fire(HookContext(phase=HookPhase.PRE_COMPUTE))
        assert len(received) == 1

    def test_suppress_all(self, hook_manager: HookManager) -> None:
        """Suppressed manager doesn't fire any hooks."""
        received: List[str] = []
        hook_manager.on(HookPhase.PRE_COMPUTE, lambda ctx: received.append("fired"))

        hook_manager.suppress_all()
        hook_manager.fire(HookContext(phase=HookPhase.PRE_COMPUTE))
        assert len(received) == 0

        hook_manager.unsuppress_all()
        hook_manager.fire(HookContext(phase=HookPhase.PRE_COMPUTE))
        assert len(received) == 1

    def test_cancel_in_pre_hook(self, hook_manager: HookManager) -> None:
        """PRE hooks can cancel operations via context."""
        hook_manager.on(HookPhase.PRE_COMPUTE, lambda ctx: ctx.cancel())

        ctx = HookContext(phase=HookPhase.PRE_COMPUTE, node_id="test")
        result = hook_manager.fire(ctx)
        assert result.cancelled is True

    def test_clear_phase(self, hook_manager: HookManager) -> None:
        """clear_phase removes all hooks for a phase."""
        hook_manager.on(HookPhase.PRE_COMPUTE, lambda ctx: None)
        hook_manager.on(HookPhase.PRE_COMPUTE, lambda ctx: None)
        hook_manager.on(HookPhase.POST_COMPUTE, lambda ctx: None)

        removed = hook_manager.clear_phase(HookPhase.PRE_COMPUTE)
        assert removed == 2
        assert hook_manager.total_registrations == 1

    def test_get_stats(self, hook_manager: HookManager) -> None:
        """Stats report registration and fire counts."""
        hook_manager.on(HookPhase.PRE_COMPUTE, lambda ctx: None)
        hook_manager.fire(HookContext(phase=HookPhase.PRE_COMPUTE))

        stats = hook_manager.get_stats()
        assert stats["total_registrations"] == 1
        assert stats["total_fires"] == 1


# ─── Plugin Registry Tests ─────────────────────────────────────────────


class TestPluginRegistry:
    """Tests for PluginRegistry lifecycle management."""

    def test_register_plugin(self, registry: PluginRegistry) -> None:
        """Registering a plugin adds it to the registry."""
        plugin = SamplePlugin()
        registry.register(plugin)
        assert registry.plugin_count == 1

    def test_register_duplicate_raises(self, registry: PluginRegistry) -> None:
        """Registering same name twice raises ValueError."""
        registry.register(SamplePlugin("dup"))
        with pytest.raises(ValueError):
            registry.register(SamplePlugin("dup"))

    def test_load_plugin(self, registry: PluginRegistry) -> None:
        """Loading a plugin calls on_load."""
        plugin = SamplePlugin()
        registry.register(plugin)
        registry.load("sample")

        assert plugin.load_called
        assert plugin.state == PluginState.LOADED

    def test_activate_plugin(self, registry: PluginRegistry) -> None:
        """Activating a loaded plugin calls on_activate."""
        plugin = SamplePlugin()
        registry.register(plugin)
        registry.load("sample")
        registry.activate("sample")

        assert plugin.activate_called
        assert plugin.state == PluginState.ACTIVE
        assert "sample" in registry.active_plugins

    def test_deactivate_plugin(self, registry: PluginRegistry) -> None:
        """Deactivating calls on_deactivate."""
        plugin = SamplePlugin()
        registry.register(plugin)
        registry.load("sample")
        registry.activate("sample")
        registry.deactivate("sample")

        assert plugin.deactivate_called
        assert plugin.state == PluginState.DEACTIVATED

    def test_unload_plugin(self, registry: PluginRegistry) -> None:
        """Unloading calls on_unload and resets state."""
        plugin = SamplePlugin()
        registry.register(plugin)
        registry.load("sample")
        registry.unload("sample")

        assert plugin.unload_called
        assert plugin.state == PluginState.REGISTERED

    def test_full_lifecycle(self, registry: PluginRegistry) -> None:
        """Full lifecycle: register -> load -> activate -> deactivate -> unload."""
        plugin = SamplePlugin()
        registry.register(plugin)

        assert plugin.state == PluginState.REGISTERED
        registry.load("sample")
        assert plugin.state == PluginState.LOADED
        registry.activate("sample")
        assert plugin.state == PluginState.ACTIVE
        registry.deactivate("sample")
        assert plugin.state == PluginState.DEACTIVATED
        registry.unload("sample")
        assert plugin.state == PluginState.REGISTERED

    def test_dependency_auto_load(self, registry: PluginRegistry) -> None:
        """Loading a plugin auto-loads its dependencies."""
        sample = SamplePlugin()
        dependent = DependentPlugin()
        registry.register(sample)
        registry.register(dependent)

        registry.load("dependent")
        assert sample.load_called  # Auto-loaded

    def test_missing_dependency_raises(self, registry: PluginRegistry) -> None:
        """Loading with missing dependency raises RuntimeError."""
        dependent = DependentPlugin()
        registry.register(dependent)

        with pytest.raises(RuntimeError, match="sample"):
            registry.load("dependent")

    def test_unregister_plugin(self, registry: PluginRegistry) -> None:
        """Unregistering removes plugin completely."""
        plugin = SamplePlugin()
        registry.register(plugin)
        registry.load("sample")

        result = registry.unregister("sample")
        assert result is True
        assert registry.plugin_count == 0

    def test_get_by_tag(self, registry: PluginRegistry) -> None:
        """Find plugins by tag."""
        registry.register(SamplePlugin("a"))
        names = registry.get_by_tag("test")
        assert "a" in names


# ─── Logging Middleware Tests ──────────────────────────────────────────


class TestLoggingMiddleware:
    """Tests for the LoggingMiddleware."""

    def test_creation(self) -> None:
        """Logging middleware initializes correctly."""
        mw = LoggingMiddleware(level=LogLevel.DEBUG)
        assert mw.name == "dagflow.logging"
        assert mw.entry_count == 0

    def test_captures_post_compute(
        self, simple_graph: ComputeGraph, hook_manager: HookManager
    ) -> None:
        """Captures POST_COMPUTE events."""
        mw = LoggingMiddleware(level=LogLevel.DEBUG)
        mw.on_load(simple_graph, hook_manager)
        mw.on_activate()

        # Simulate a post-compute hook
        ctx = HookContext(
            phase=HookPhase.POST_COMPUTE,
            node_id="double",
            value=42,
        )
        hook_manager.fire(ctx)

        assert mw.entry_count >= 1

    def test_captures_errors(
        self, simple_graph: ComputeGraph, hook_manager: HookManager
    ) -> None:
        """Captures ON_ERROR events."""
        mw = LoggingMiddleware(level=LogLevel.DEBUG)
        mw.on_load(simple_graph, hook_manager)
        mw.on_activate()

        ctx = HookContext(
            phase=HookPhase.ON_ERROR,
            node_id="double",
            error=ValueError("test error"),
        )
        hook_manager.fire(ctx)

        errors = mw.get_entries_by_level(LogLevel.ERROR)
        assert len(errors) >= 1

    def test_level_filtering(
        self, simple_graph: ComputeGraph, hook_manager: HookManager
    ) -> None:
        """Only logs at or above configured level."""
        mw = LoggingMiddleware(level=LogLevel.WARNING)
        mw.on_load(simple_graph, hook_manager)
        mw.on_activate()

        # DEBUG event should be filtered
        ctx = HookContext(phase=HookPhase.PRE_COMPUTE, node_id="double")
        hook_manager.fire(ctx)

        # Only the activation INFO message should be filtered too
        assert mw.entry_count == 0

    def test_node_filter(
        self, simple_graph: ComputeGraph, hook_manager: HookManager
    ) -> None:
        """Node filter restricts which nodes are logged."""
        mw = LoggingMiddleware(
            level=LogLevel.DEBUG, node_filter={"double"}
        )
        mw.on_load(simple_graph, hook_manager)
        mw.on_activate()

        hook_manager.fire(
            HookContext(phase=HookPhase.PRE_COMPUTE, node_id="other_node")
        )
        hook_manager.fire(
            HookContext(phase=HookPhase.PRE_COMPUTE, node_id="double")
        )

        entries = mw.get_entries_by_node("double")
        other_entries = mw.get_entries_by_node("other_node")
        assert len(entries) >= 1
        assert len(other_entries) == 0

    def test_clear_entries(
        self, simple_graph: ComputeGraph, hook_manager: HookManager
    ) -> None:
        """clear() removes all log entries."""
        mw = LoggingMiddleware(level=LogLevel.DEBUG)
        mw.on_load(simple_graph, hook_manager)
        mw.on_activate()

        hook_manager.fire(HookContext(phase=HookPhase.PRE_COMPUTE, node_id="x"))
        assert mw.entry_count >= 1

        cleared = mw.clear()
        assert cleared >= 1
        assert mw.entry_count == 0

    def test_output_callback(
        self, simple_graph: ComputeGraph, hook_manager: HookManager
    ) -> None:
        """Output callback receives entries in real-time."""
        output: List[Any] = []
        mw = LoggingMiddleware(
            level=LogLevel.DEBUG,
            output_fn=lambda entry: output.append(entry),
        )
        mw.on_load(simple_graph, hook_manager)
        mw.on_activate()

        hook_manager.fire(HookContext(phase=HookPhase.PRE_COMPUTE, node_id="x"))
        assert len(output) >= 1
