"""Tests for the CLI module.

Tests cover:
- Argument parsing
- Command execution (inspect, validate, export, run)
- Output formatters (table, JSON, tree)
- Input value parsing
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from dagflow.core.graph import ComputeGraph
from dagflow.memo.cache import MemoCache
from dagflow.cli.__main__ import create_parser, main, parse_input_values
from dagflow.cli.commands import (
    CommandResult,
    ExportCommand,
    InspectCommand,
    RunCommand,
    ValidateCommand,
)
from dagflow.cli.formatters import (
    FormatKind,
    JsonFormatter,
    TableFormatter,
    TreeFormatter,
    format_output,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def test_graph() -> ComputeGraph:
    """Create a test graph for CLI commands."""
    g = ComputeGraph()
    g.add_input("x", value=5)
    g.add_input("y", value=3)
    g.add_compute("sum", lambda d: d["x"] + d["y"], ["x", "y"])
    g.add_compute("product", lambda d: d["x"] * d["y"], ["x", "y"])
    g.add_compute("result", lambda d: d["sum"] + d["product"], ["sum", "product"])
    return g


# ─── Argument Parser Tests ─────────────────────────────────────────────


class TestArgumentParser:
    """Tests for CLI argument parsing."""

    def test_create_parser(self) -> None:
        """Parser is created without errors."""
        parser = create_parser()
        assert parser is not None

    def test_parse_inspect(self) -> None:
        """Inspect command parses correctly."""
        parser = create_parser()
        args = parser.parse_args(["inspect", "node_a", "--depth", "2"])
        assert args.command == "inspect"
        assert args.target == "node_a"
        assert args.depth == 2

    def test_parse_validate(self) -> None:
        """Validate command parses correctly."""
        parser = create_parser()
        args = parser.parse_args(["validate", "--strict", "--check-orphans"])
        assert args.command == "validate"
        assert args.strict is True
        assert args.check_orphans is True

    def test_parse_export(self) -> None:
        """Export command parses correctly."""
        parser = create_parser()
        args = parser.parse_args(["export", "output.json", "--format", "dot"])
        assert args.command == "export"
        assert args.output == "output.json"
        assert args.export_format == "dot"

    def test_parse_run(self) -> None:
        """Run command parses correctly."""
        parser = create_parser()
        args = parser.parse_args(["run", "-i", "x=10", "-i", "y=20", "--dry-run"])
        assert args.command == "run"
        assert args.input == ["x=10", "y=20"]
        assert args.dry_run is True

    def test_parse_global_options(self) -> None:
        """Global options (format, verbose) parse correctly."""
        parser = create_parser()
        args = parser.parse_args(["--format", "json", "--verbose", "inspect"])
        assert args.format == "json"
        assert args.verbose is True

    def test_no_command_returns_zero(self) -> None:
        """No command prints help and returns 0."""
        result = main([])
        assert result == 0


# ─── Input Value Parsing Tests ─────────────────────────────────────────


class TestInputParsing:
    """Tests for parse_input_values."""

    def test_parse_integer(self) -> None:
        """Integer values are parsed correctly."""
        values = parse_input_values(["x=42"])
        assert values == {"x": 42}
        assert isinstance(values["x"], int)

    def test_parse_float(self) -> None:
        """Float values are parsed correctly."""
        values = parse_input_values(["rate=3.14"])
        assert values == {"rate": pytest.approx(3.14)}
        assert isinstance(values["rate"], float)

    def test_parse_string(self) -> None:
        """Non-numeric values remain as strings."""
        values = parse_input_values(["name=hello"])
        assert values == {"name": "hello"}

    def test_parse_multiple(self) -> None:
        """Multiple inputs are all parsed."""
        values = parse_input_values(["x=1", "y=2.5", "z=abc"])
        assert values == {"x": 1, "y": 2.5, "z": "abc"}

    def test_parse_invalid_format(self) -> None:
        """Missing = raises ValueError."""
        with pytest.raises(ValueError, match="Invalid input format"):
            parse_input_values(["invalid"])

    def test_parse_value_with_equals(self) -> None:
        """Values containing = are handled correctly."""
        values = parse_input_values(["expr=a=b"])
        assert values == {"expr": "a=b"}


# ─── Command Execution Tests ──────────────────────────────────────────


class TestInspectCommand:
    """Tests for InspectCommand."""

    def test_inspect_graph_overview(self, test_graph: ComputeGraph, capsys) -> None:
        """Inspect without target shows graph overview."""
        cmd = InspectCommand(format_kind=FormatKind.JSON)
        result = cmd.execute(graph=test_graph)
        assert result == 0

        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["total_nodes"] == 5
        assert data["input_nodes"] == 2
        assert data["compute_nodes"] == 3

    def test_inspect_specific_node(self, test_graph: ComputeGraph, capsys) -> None:
        """Inspect with target shows node details."""
        cmd = InspectCommand(format_kind=FormatKind.JSON)
        result = cmd.execute(target="sum", graph=test_graph)
        assert result == 0

        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["node_id"] == "sum"
        assert data["is_input"] is False

    def test_inspect_with_values(self, test_graph: ComputeGraph, capsys) -> None:
        """Inspect with show_values includes cached values."""
        cmd = InspectCommand(format_kind=FormatKind.JSON)
        result = cmd.execute(show_values=True, graph=test_graph)
        assert result == 0

        output = capsys.readouterr().out
        data = json.loads(output)
        assert "values" in data


class TestValidateCommand:
    """Tests for ValidateCommand."""

    def test_validate_valid_graph(self, test_graph: ComputeGraph, capsys) -> None:
        """Valid graph passes validation."""
        cmd = ValidateCommand(format_kind=FormatKind.JSON)
        result = cmd.execute(graph=test_graph)
        assert result == 0

        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["valid"] is True

    def test_validate_no_graph(self, capsys) -> None:
        """No graph returns valid with message."""
        cmd = ValidateCommand(format_kind=FormatKind.JSON)
        result = cmd.execute(graph=None)
        assert result == 0


class TestExportCommand:
    """Tests for ExportCommand."""

    def test_export_json(self, test_graph: ComputeGraph, capsys) -> None:
        """Export as JSON succeeds."""
        cmd = ExportCommand(format_kind=FormatKind.JSON)
        result = cmd.execute(
            output_path="test.json",
            export_format="json",
            graph=test_graph,
        )
        assert result == 0

    def test_export_dot(self, test_graph: ComputeGraph, capsys) -> None:
        """Export as DOT succeeds."""
        cmd = ExportCommand(format_kind=FormatKind.JSON)
        result = cmd.execute(
            output_path="test.dot",
            export_format="dot",
            graph=test_graph,
        )
        assert result == 0

    def test_export_no_graph(self, capsys) -> None:
        """Export without graph returns error."""
        cmd = ExportCommand()
        result = cmd.execute(output_path="test.json", graph=None)
        assert result == 1


class TestRunCommand:
    """Tests for RunCommand."""

    def test_run_with_inputs(self, test_graph: ComputeGraph, capsys) -> None:
        """Run with inputs computes results."""
        cache = MemoCache()
        cmd = RunCommand(format_kind=FormatKind.JSON)
        result = cmd.execute(
            inputs={"x": 10, "y": 5},
            graph=test_graph,
            cache=cache,
        )
        assert result == 0

        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["computed"] is True
        assert "sum" in data["results"]
        assert data["results"]["sum"] == 15

    def test_run_dry_run(self, test_graph: ComputeGraph, capsys) -> None:
        """Dry run shows affected nodes without computing."""
        cmd = RunCommand(format_kind=FormatKind.JSON)
        result = cmd.execute(
            inputs={"x": 10},
            dry_run=True,
            graph=test_graph,
        )
        assert result == 0

        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["dry_run"] is True


# ─── Formatter Tests ───────────────────────────────────────────────────


class TestFormatters:
    """Tests for output formatters."""

    def test_json_formatter_dict(self) -> None:
        """JSON formatter handles dicts."""
        data = {"key": "value", "num": 42}
        output = JsonFormatter.format(data)
        parsed = json.loads(output)
        assert parsed == data

    def test_json_formatter_nested(self) -> None:
        """JSON formatter handles nested structures."""
        data = {"outer": {"inner": [1, 2, 3]}}
        output = JsonFormatter.format(data)
        parsed = json.loads(output)
        assert parsed == data

    def test_table_formatter_dict(self) -> None:
        """Table formatter produces aligned key-value output."""
        data = {"name": "test", "value": 42}
        output = TableFormatter.format(data)
        assert "name" in output
        assert "42" in output

    def test_table_formatter_list_of_dicts(self) -> None:
        """Table formatter produces tabular output for list of dicts."""
        data = [
            {"id": "a", "value": 1},
            {"id": "b", "value": 2},
        ]
        output = TableFormatter.format(data)
        assert "id" in output
        assert "value" in output

    def test_table_formatter_empty(self) -> None:
        """Table formatter handles empty dict."""
        output = TableFormatter.format({})
        assert output == "(empty)"

    def test_tree_formatter_dict(self) -> None:
        """Tree formatter produces tree-like output."""
        data = {"root": {"child1": "val1", "child2": "val2"}}
        output = TreeFormatter.format(data)
        assert "root" in output
        assert "child1" in output

    def test_tree_formatter_list(self) -> None:
        """Tree formatter handles lists."""
        data = {"items": ["a", "b", "c"]}
        output = TreeFormatter.format(data)
        assert "items" in output

    def test_format_kind_from_string(self) -> None:
        """FormatKind.from_string parses correctly."""
        assert FormatKind.from_string("json") == FormatKind.JSON
        assert FormatKind.from_string("table") == FormatKind.TABLE
        assert FormatKind.from_string("tree") == FormatKind.TREE
        assert FormatKind.from_string("unknown") == FormatKind.TABLE

    def test_format_output_dispatch(self) -> None:
        """format_output dispatches to correct formatter."""
        data = {"key": "value"}
        json_out = format_output(data, FormatKind.JSON)
        table_out = format_output(data, FormatKind.TABLE)

        # JSON output should be valid JSON
        json.loads(json_out)
        # Table output should not be valid JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(table_out)
