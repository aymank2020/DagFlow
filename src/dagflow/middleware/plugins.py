"""Plugin registry with lifecycle management.

Provides a plugin system that allows extending DagFlow with custom
functionality. Plugins have a defined lifecycle (load, activate,
deactivate, unload) and can register hooks, add nodes, or modify
graph behavior.

This module depends on:
- middleware.hooks (HookManager, HookPhase, HookContext)
- core.graph (ComputeGraph)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type

from dagflow.core.graph import ComputeGraph
from dagflow.middleware.hooks import HookManager


class PluginState(Enum):
    """Lifecycle state of a plugin."""

    REGISTERED = auto()
    LOADED = auto()
    ACTIVE = auto()
    DEACTIVATED = auto()
    ERROR = auto()


@dataclass
class PluginMetadata:
    """Metadata describing a plugin.

    Attributes:
        name: Unique plugin identifier.
        version: Semantic version string.
        description: Human-readable description.
        author: Plugin author.
        dependencies: Other plugins this one requires.
        tags: Categorization tags.
    """

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class Plugin(ABC):
    """Abstract base class for DagFlow plugins.

    Plugins extend the computation engine with custom behavior.
    They have a defined lifecycle and access to the graph and
    hook system.

    Subclass this and implement the lifecycle methods:
        class MyPlugin(Plugin):
            def on_load(self, graph, hooks):
                # Setup
                pass

            def on_activate(self):
                # Start working
                pass
    """

    def __init__(self, metadata: PluginMetadata) -> None:
        self._metadata = metadata
        self._state = PluginState.REGISTERED
        self._graph: Optional[ComputeGraph] = None
        self._hooks: Optional[HookManager] = None
        self._hook_ids: List[str] = []

    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata."""
        return self._metadata

    @property
    def name(self) -> str:
        """Plugin name."""
        return self._metadata.name

    @property
    def state(self) -> PluginState:
        """Current lifecycle state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether the plugin is currently active."""
        return self._state == PluginState.ACTIVE

    def on_load(self, graph: ComputeGraph, hooks: HookManager) -> None:
        """Called when the plugin is loaded into the system.

        Override to perform initialization, register hooks, etc.

        Args:
            graph: The computation graph.
            hooks: The hook manager for registering callbacks.
        """
        self._graph = graph
        self._hooks = hooks

    def on_activate(self) -> None:
        """Called when the plugin is activated.

        Override to start any active processing.
        """

    def on_deactivate(self) -> None:
        """Called when the plugin is deactivated.

        Override to pause processing without full cleanup.
        """

    def on_unload(self) -> None:
        """Called when the plugin is being removed.

        Override to perform cleanup. Hook registrations are
        automatically removed.
        """
        # Auto-remove registered hooks
        if self._hooks:
            for hook_id in self._hook_ids:
                self._hooks.off(hook_id)
        self._hook_ids.clear()

    def register_hook(self, hook_id: str) -> None:
        """Track a hook registration for auto-cleanup."""
        self._hook_ids.append(hook_id)


class PluginRegistry:
    """Central registry for managing plugins.

    Handles plugin registration, lifecycle transitions, dependency
    resolution, and provides querying capabilities.

    Usage:
        registry = PluginRegistry(graph, hooks)
        registry.register(MyPlugin(metadata))
        registry.load("my_plugin")
        registry.activate("my_plugin")
    """

    def __init__(self, graph: ComputeGraph, hooks: HookManager) -> None:
        self._graph = graph
        self._hooks = hooks
        self._plugins: Dict[str, Plugin] = {}
        self._load_order: List[str] = []

    @property
    def plugin_count(self) -> int:
        """Number of registered plugins."""
        return len(self._plugins)

    @property
    def active_plugins(self) -> List[str]:
        """Names of currently active plugins."""
        return [
            name for name, plugin in self._plugins.items()
            if plugin.is_active
        ]

    def register(self, plugin: Plugin) -> bool:
        """Register a plugin with the registry.

        Args:
            plugin: The plugin instance to register.

        Returns:
            True if registered successfully.

        Raises:
            ValueError: If a plugin with the same name exists.
        """
        name = plugin.name
        if name in self._plugins:
            raise ValueError(f"Plugin '{name}' already registered")

        self._plugins[name] = plugin
        return True

    def unregister(self, name: str) -> bool:
        """Remove a plugin from the registry.

        Unloads the plugin first if it's loaded.

        Returns:
            True if the plugin was found and removed.
        """
        plugin = self._plugins.get(name)
        if plugin is None:
            return False

        if plugin.state in (PluginState.ACTIVE, PluginState.LOADED):
            self.unload(name)

        del self._plugins[name]
        if name in self._load_order:
            self._load_order.remove(name)
        return True

    def load(self, name: str) -> bool:
        """Load a plugin, resolving dependencies first.

        Args:
            name: Plugin name to load.

        Returns:
            True if loaded successfully.

        Raises:
            KeyError: If plugin not found.
            RuntimeError: If dependencies are not satisfied.
        """
        plugin = self._get_plugin(name)

        if plugin.state != PluginState.REGISTERED:
            return False

        # Check dependencies
        for dep_name in plugin.metadata.dependencies:
            dep = self._plugins.get(dep_name)
            if dep is None:
                raise RuntimeError(
                    f"Plugin '{name}' requires '{dep_name}' which is not registered"
                )
            if dep.state == PluginState.REGISTERED:
                # Auto-load dependency
                self.load(dep_name)

        try:
            plugin.on_load(self._graph, self._hooks)
            plugin._state = PluginState.LOADED
            self._load_order.append(name)
            return True
        except Exception as e:
            plugin._state = PluginState.ERROR
            return False

    def activate(self, name: str) -> bool:
        """Activate a loaded plugin.

        Args:
            name: Plugin name to activate.

        Returns:
            True if activated successfully.
        """
        plugin = self._get_plugin(name)

        if plugin.state == PluginState.REGISTERED:
            self.load(name)

        if plugin.state != PluginState.LOADED and plugin.state != PluginState.DEACTIVATED:
            return False

        try:
            plugin.on_activate()
            plugin._state = PluginState.ACTIVE
            return True
        except Exception:
            plugin._state = PluginState.ERROR
            return False

    def deactivate(self, name: str) -> bool:
        """Deactivate an active plugin.

        Args:
            name: Plugin name to deactivate.

        Returns:
            True if deactivated successfully.
        """
        plugin = self._get_plugin(name)

        if plugin.state != PluginState.ACTIVE:
            return False

        try:
            plugin.on_deactivate()
            plugin._state = PluginState.DEACTIVATED
            return True
        except Exception:
            plugin._state = PluginState.ERROR
            return False

    def unload(self, name: str) -> bool:
        """Unload a plugin, deactivating first if needed.

        Args:
            name: Plugin name to unload.

        Returns:
            True if unloaded successfully.
        """
        plugin = self._get_plugin(name)

        if plugin.state == PluginState.ACTIVE:
            self.deactivate(name)

        if plugin.state not in (PluginState.LOADED, PluginState.DEACTIVATED):
            return False

        try:
            plugin.on_unload()
            plugin._state = PluginState.REGISTERED
            if name in self._load_order:
                self._load_order.remove(name)
            return True
        except Exception:
            plugin._state = PluginState.ERROR
            return False

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name, or None if not found."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[PluginMetadata]:
        """List metadata for all registered plugins."""
        return [p.metadata for p in self._plugins.values()]

    def get_by_tag(self, tag: str) -> List[str]:
        """Find plugins by tag.

        Args:
            tag: Tag to search for.

        Returns:
            List of plugin names with the specified tag.
        """
        return [
            name
            for name, plugin in self._plugins.items()
            if tag in plugin.metadata.tags
        ]

    def _get_plugin(self, name: str) -> Plugin:
        """Get a plugin or raise KeyError."""
        plugin = self._plugins.get(name)
        if plugin is None:
            raise KeyError(f"Plugin '{name}' not found")
        return plugin
