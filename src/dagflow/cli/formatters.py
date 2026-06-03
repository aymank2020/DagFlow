"""Output formatters — table, JSON, and tree display.

Provides configurable output formatting for CLI command results.
Supports table (human-readable), JSON (machine-readable), and
tree (hierarchical) output formats.

This module has no internal dagflow dependencies.
"""

from __future__ import annotations

import json
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class FormatKind(Enum):
    """Available output format types."""

    TABLE = auto()
    JSON = auto()
    TREE = auto()

    @classmethod
    def from_string(cls, value: str) -> "FormatKind":
        """Parse a format kind from string."""
        mapping = {"table": cls.TABLE, "json": cls.JSON, "tree": cls.TREE}
        return mapping.get(value.lower(), cls.TABLE)


def format_output(data: Any, kind: FormatKind) -> str:
    """Format data according to the specified format kind.

    Args:
        data: The data to format (dict, list, or primitive).
        kind: The output format to use.

    Returns:
        Formatted string representation.
    """
    if kind == FormatKind.JSON:
        return JsonFormatter.format(data)
    elif kind == FormatKind.TREE:
        return TreeFormatter.format(data)
    else:
        return TableFormatter.format(data)


class TableFormatter:
    """Formats data as aligned text tables.

    Handles dicts (key-value pairs), lists of dicts (tabular data),
    and nested structures with indentation.
    """

    @staticmethod
    def format(data: Any, indent: int = 0) -> str:
        """Format data as a table.

        Args:
            data: Data to format.
            indent: Current indentation level.

        Returns:
            Formatted table string.
        """
        if isinstance(data, dict):
            return TableFormatter._format_dict(data, indent)
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                return TableFormatter._format_list_of_dicts(data)
            return TableFormatter._format_list(data, indent)
        else:
            return str(data)

    @staticmethod
    def _format_dict(data: Dict[str, Any], indent: int = 0) -> str:
        """Format a dictionary as aligned key-value pairs."""
        if not data:
            return "(empty)"

        prefix = "  " * indent
        # Find max key length for alignment
        max_key_len = max(len(str(k)) for k in data.keys())
        lines: List[str] = []

        for key, value in data.items():
            key_str = str(key).ljust(max_key_len)
            if isinstance(value, dict):
                lines.append(f"{prefix}{key_str} :")
                lines.append(TableFormatter._format_dict(value, indent + 1))
            elif isinstance(value, list) and len(value) > 3:
                lines.append(f"{prefix}{key_str} : [{len(value)} items]")
            else:
                lines.append(f"{prefix}{key_str} : {value}")

        return "\n".join(lines)

    @staticmethod
    def _format_list_of_dicts(data: List[Dict[str, Any]]) -> str:
        """Format a list of dicts as a table with headers."""
        if not data:
            return "(empty)"

        # Collect all keys for columns
        columns: List[str] = []
        for row in data:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)

        # Compute column widths
        widths: Dict[str, int] = {}
        for col in columns:
            widths[col] = max(
                len(col),
                max((len(str(row.get(col, ""))) for row in data), default=0),
            )

        # Build header
        header = " | ".join(col.ljust(widths[col]) for col in columns)
        separator = "-+-".join("-" * widths[col] for col in columns)

        # Build rows
        rows: List[str] = []
        for row in data:
            cells = [str(row.get(col, "")).ljust(widths[col]) for col in columns]
            rows.append(" | ".join(cells))

        return "\n".join([header, separator] + rows)

    @staticmethod
    def _format_list(data: List[Any], indent: int = 0) -> str:
        """Format a simple list."""
        prefix = "  " * indent
        lines = [f"{prefix}- {item}" for item in data]
        return "\n".join(lines)


class JsonFormatter:
    """Formats data as pretty-printed JSON."""

    @staticmethod
    def format(data: Any, indent: int = 2) -> str:
        """Format data as JSON.

        Args:
            data: Data to serialize.
            indent: JSON indentation level.

        Returns:
            Pretty-printed JSON string.
        """
        try:
            return json.dumps(data, indent=indent, default=str)
        except (TypeError, ValueError):
            return json.dumps({"value": str(data)}, indent=indent)


class TreeFormatter:
    """Formats hierarchical data as an ASCII tree.

    Produces tree-like output with box-drawing characters
    for visualizing nested structures.
    """

    PIPE = "│   "
    TEE = "├── "
    LAST = "└── "
    BLANK = "    "

    @staticmethod
    def format(data: Any, prefix: str = "", is_last: bool = True) -> str:
        """Format data as an ASCII tree.

        Args:
            data: Data to format (dict or nested structure).
            prefix: Current line prefix for indentation.
            is_last: Whether this is the last sibling.

        Returns:
            Tree-formatted string.
        """
        if isinstance(data, dict):
            return TreeFormatter._format_dict_tree(data, prefix)
        elif isinstance(data, list):
            return TreeFormatter._format_list_tree(data, prefix)
        else:
            return str(data)

    @staticmethod
    def _format_dict_tree(data: Dict[str, Any], prefix: str = "") -> str:
        """Format a dict as a tree with branches."""
        if not data:
            return "(empty)"

        lines: List[str] = []
        items = list(data.items())

        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            connector = TreeFormatter.LAST if is_last else TreeFormatter.TEE
            child_prefix = prefix + (TreeFormatter.BLANK if is_last else TreeFormatter.PIPE)

            if isinstance(value, dict):
                lines.append(f"{prefix}{connector}{key}:")
                subtree = TreeFormatter._format_dict_tree(value, child_prefix)
                lines.append(subtree)
            elif isinstance(value, list) and value:
                lines.append(f"{prefix}{connector}{key}:")
                subtree = TreeFormatter._format_list_tree(value, child_prefix)
                lines.append(subtree)
            else:
                lines.append(f"{prefix}{connector}{key}: {value}")

        return "\n".join(lines)

    @staticmethod
    def _format_list_tree(data: List[Any], prefix: str = "") -> str:
        """Format a list as tree branches."""
        if not data:
            return f"{prefix}(empty)"

        lines: List[str] = []
        for i, item in enumerate(data):
            is_last = i == len(data) - 1
            connector = TreeFormatter.LAST if is_last else TreeFormatter.TEE

            if isinstance(item, dict):
                lines.append(f"{prefix}{connector}[{i}]:")
                child_prefix = prefix + (TreeFormatter.BLANK if is_last else TreeFormatter.PIPE)
                subtree = TreeFormatter._format_dict_tree(item, child_prefix)
                lines.append(subtree)
            else:
                lines.append(f"{prefix}{connector}{item}")

        return "\n".join(lines)
