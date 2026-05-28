"""GraphSerializer — serialize and deserialize computation graphs to JSON.

Converts a ComputeGraph (structure + current values) into a JSON-compatible
dictionary, and reconstructs a graph from that representation. Compute
functions are referenced by name and must be provided via a function registry
during deserialization.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode, NodeState)
- memo.cache (MemoCache)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode, NodeState
from dagflow.memo.cache import MemoCache


class SerializationError(Exception):
    """Raised when serialization or deserialization fails."""


class GraphSerializer:
    """Serialize and deserialize ComputeGraph instances.

    Compute functions cannot be serialized directly, so they are stored
    by name. A function registry maps names back to callables during
    deserialization.

    Usage:
        serializer = GraphSerializer()
        serializer.register_function("add", lambda d: d["a"] + d["b"])

        data = serializer.serialize(graph, cache)
        json_str = serializer.to_json(graph, cache)

        new_graph, new_cache = serializer.deserialize(data)
    """

    def __init__(self) -> None:
        self._function_registry: Dict[str, Callable[[dict], Any]] = {}
        self._function_names: Dict[int, str] = {}  # id(func) -> name

    def register_function(self, name: str, func: Callable[[dict], Any]) -> None:
        """Register a named function for serialization/deserialization.

        Args:
            name: Unique string identifier for this function.
            func: The computation function.
        """
        self._function_registry[name] = func
        self._function_names[id(func)] = name

    def get_function_name(self, func: Callable[[dict], Any]) -> Optional[str]:
        """Look up the registered name for a function."""
        return self._function_names.get(id(func))

    def serialize(self, graph: ComputeGraph, cache: MemoCache) -> Dict[str, Any]:
        """Serialize graph structure and state to a dictionary.

        Args:
            graph: The computation graph to serialize.
            cache: The memo cache with current values.

        Returns:
            JSON-compatible dictionary representation.

        Raises:
            SerializationError: If a compute function has no registered name.
        """
        nodes: List[Dict[str, Any]] = []

        for node_id in graph.all_node_ids:
            node = graph.get_node(node_id)
            if isinstance(node, InputNode):
                nodes.append({
                    "type": "input",
                    "id": node_id,
                    "value": self._serialize_value(node.value),
                    "generation": node.generation,
                })
            elif isinstance(node, ComputeNode):
                func_name = self.get_function_name(node.func)
                if func_name is None:
                    raise SerializationError(
                        f"Compute node '{node_id}' has unregistered function. "
                        f"Register it with serializer.register_function() first."
                    )
                nodes.append({
                    "type": "compute",
                    "id": node_id,
                    "function": func_name,
                    "dependencies": list(node.dependencies),
                    "priority": node.priority,
                    "state": node.state.name,
                    "cached_value": self._serialize_value(node.cached_value),
                    "last_valid_generation": dict(node.last_valid_generation),
                })

        # Serialize cache entries
        cache_data: Dict[str, Any] = {}
        for node_id in graph.get_compute_nodes():
            entry_val = cache.get(node_id)
            if entry_val is not None:
                cache_data[node_id] = {
                    "value": self._serialize_value(entry_val),
                    "generation": cache.get_generation(node_id),
                    "dep_snapshot": cache.get_dep_snapshot(node_id),
                }

        return {
            "version": "1.0",
            "nodes": nodes,
            "cache": cache_data,
        }

    def to_json(self, graph: ComputeGraph, cache: MemoCache, indent: int = 2) -> str:
        """Serialize graph to a JSON string.

        Args:
            graph: The computation graph.
            cache: The memo cache.
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        data = self.serialize(graph, cache)
        return json.dumps(data, indent=indent, default=str)

    def deserialize(self, data: Dict[str, Any]) -> Tuple[ComputeGraph, MemoCache]:
        """Reconstruct a graph and cache from serialized data.

        Args:
            data: Dictionary previously produced by serialize().

        Returns:
            Tuple of (ComputeGraph, MemoCache).

        Raises:
            SerializationError: If function names are not in the registry.
        """
        graph = ComputeGraph()
        cache = MemoCache()

        # First pass: create input nodes
        for node_data in data["nodes"]:
            if node_data["type"] == "input":
                inp = graph.add_input(node_data["id"], value=node_data["value"])
                inp.generation = node_data.get("generation", 0)

        # Second pass: create compute nodes (dependencies must exist)
        for node_data in data["nodes"]:
            if node_data["type"] == "compute":
                func_name = node_data["function"]
                if func_name not in self._function_registry:
                    raise SerializationError(
                        f"Function '{func_name}' not found in registry"
                    )
                func = self._function_registry[func_name]
                node = graph.add_compute(
                    node_data["id"],
                    func=func,
                    dependencies=node_data["dependencies"],
                    priority=node_data.get("priority", 0),
                )
                # Restore state
                state_name = node_data.get("state", "DIRTY")
                node.state = NodeState[state_name]
                node.cached_value = node_data.get("cached_value")
                node.last_valid_generation = node_data.get(
                    "last_valid_generation", {}
                )

        # Restore cache
        for node_id, cache_entry in data.get("cache", {}).items():
            cache.store(
                node_id,
                cache_entry["value"],
                cache_entry.get("dep_snapshot", {}),
            )

        return graph, cache

    def from_json(self, json_str: str) -> Tuple[ComputeGraph, MemoCache]:
        """Deserialize a graph from a JSON string.

        Args:
            json_str: JSON string previously produced by to_json().

        Returns:
            Tuple of (ComputeGraph, MemoCache).
        """
        data = json.loads(json_str)
        return self.deserialize(data)

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Convert a value to a JSON-serializable form.

        Handles common Python types. Complex objects are converted to
        their string representation.
        """
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [GraphSerializer._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {
                str(k): GraphSerializer._serialize_value(v)
                for k, v in value.items()
            }
        # Fallback: use string representation
        return str(value)
