"""Middleware and plugin system for DAG computation hooks.

Provides extensibility through pre/post computation hooks,
a plugin registry with lifecycle management, and built-in
middleware implementations.
"""

from dagflow.middleware.hooks import HookManager, HookPhase, HookContext
from dagflow.middleware.plugins import PluginRegistry, Plugin, PluginState
from dagflow.middleware.logging_mw import LoggingMiddleware, LogLevel

__all__ = [
    "HookManager",
    "HookPhase",
    "HookContext",
    "PluginRegistry",
    "Plugin",
    "PluginState",
    "LoggingMiddleware",
    "LogLevel",
]
