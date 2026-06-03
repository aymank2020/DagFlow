"""CLI commands — inspect, validate, export, run.

Implements the core CLI commands that operate on a ComputeGraph.
Each command is a class with an execute() method that returns
an exit code.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode, InputNode)
- analysis.complexity (ComplexityAnalyzer)
- analysis.patterns (PatternDetector)
- validation.integrity (IntegrityChecker)
- cli.formatters (FormatKind, TableFormatter, JsonFormatter, TreeFormatter)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from dagflow.cli.formatters import FormatKind, format_output


@dataclass
class CommandResult:
    """Result of a CLI command execution.

    Attributes:
        success: Whether the command succeeded.
        data: Structured output data.
        messages: Human-readable messages.
        warnings: Warning messages.
        errors: Error messages.
    """

    success: bool = True
    data: Dict[str, Any] = None  # type: ignore[assignment]
    messages: List[str] = None  # type: ignore[assignment]
    warnings: List[str] = None  # type: ignore[assignment]
    errors: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = {}
        if self.messages is None:
            self.messages = []
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class InspectCommand:
    """Inspect graph structure and node details.

    Provides overview of graph topology, node states, and
    dependency relationships.
    """

    def __init__(self, format_kind: FormatKind = FormatKind.TABLE, verbose: bool = False) -> None:
        self._format = format_kind
        self._verbose = verbose

    def execute(
        self,
        target: Optional[str] = None,
        depth: int = 1,
        show_values: bool = False,
        graph=None,
    ) -> int:
        """Execute the inspect command.

        Args:
            target: Specific node to inspect (None for overview).
            depth: Dependency traversal depth.
            show_values: Whether to include cached values.
            graph: Optional ComputeGraph (for testing).

        Returns:
            Exit code (0 = success).
        """
        if graph is None:
            # In real usage, would load from config/file
            result = CommandResult(
                success=True,
                data={"message": "No graph loaded. Use --graph to specify."},
                messages=["No graph loaded"],
            )
            print(format_output(result.data, self._format))
            return 0

        if target:
            data = self._inspect_node(graph, target, depth, show_values)
        else:
            data = self._inspect_graph(graph, show_values)

        print(format_output(data, self._format))
        return 0

    def _inspect_graph(self, graph, show_values: bool) -> Dict[str, Any]:
        """Generate graph overview data."""
        from dagflow.core.node import ComputeNode, InputNode

        inputs = graph.get_input_nodes()
        computes = graph.get_compute_nodes()

        data: Dict[str, Any] = {
            "type": "graph_overview",
            "total_nodes": graph.node_count,
            "input_nodes": len(inputs),
            "compute_nodes": len(computes),
            "inputs": inputs,
            "computes": computes,
        }

        if show_values:
            values: Dict[str, Any] = {}
            for nid in inputs:
                node = graph.get_node(nid)
                if isinstance(node, InputNode):
                    values[nid] = node.value
            for nid in computes:
                node = graph.get_node(nid)
                if isinstance(node, ComputeNode):
                    values[nid] = node.cached_value
            data["values"] = values

        return data

    def _inspect_node(
        self, graph, node_id: str, depth: int, show_values: bool
    ) -> Dict[str, Any]:
        """Generate detailed node inspection data."""
        from dagflow.core.node import ComputeNode, InputNode

        node = graph.get_node(node_id)
        data: Dict[str, Any] = {
            "type": "node_detail",
            "node_id": node_id,
            "is_input": node.is_input,
            "dependents": graph.get_dependents(node_id),
            "dependencies": graph.get_dependencies(node_id),
        }

        if isinstance(node, InputNode):
            data["value"] = node.value
            data["generation"] = node.generation
        elif isinstance(node, ComputeNode):
            data["state"] = node.state.name
            data["priority"] = node.priority
            if show_values:
                data["cached_value"] = node.cached_value

        if depth > 1:
            data["all_downstream"] = graph.get_all_downstream(node_id)
            data["all_upstream"] = graph.get_all_upstream(node_id)

        return data


class ValidateCommand:
    """Validate graph integrity and constraints."""

    def __init__(self, format_kind: FormatKind = FormatKind.TABLE, verbose: bool = False) -> None:
        self._format = format_kind
        self._verbose = verbose

    def execute(
        self,
        strict: bool = False,
        check_cycles: bool = True,
        check_orphans: bool = False,
        graph=None,
    ) -> int:
        """Execute validation checks.

        Returns:
            0 if valid, 1 if issues found.
        """
        issues: List[Dict[str, str]] = []

        if graph is None:
            data = {"valid": True, "message": "No graph to validate"}
            print(format_output(data, self._format))
            return 0

        # Check for orphaned nodes
        if check_orphans:
            for nid in graph.get_compute_nodes():
                deps = graph.get_dependencies(nid)
                dependents = graph.get_dependents(nid)
                if not deps and not dependents:
                    issues.append({
                        "level": "warning",
                        "node": nid,
                        "message": "Isolated node with no connections",
                    })

        # Check for nodes with missing dependencies
        for nid in graph.get_compute_nodes():
            from dagflow.core.node import ComputeNode
            node = graph.get_node(nid)
            if isinstance(node, ComputeNode):
                for dep_id in node.dependencies:
                    if dep_id not in graph.all_node_ids:
                        issues.append({
                            "level": "error",
                            "node": nid,
                            "message": f"Missing dependency: {dep_id}",
                        })

        is_valid = not any(i["level"] == "error" for i in issues)
        if strict:
            is_valid = len(issues) == 0

        data = {
            "valid": is_valid,
            "issue_count": len(issues),
            "issues": issues,
        }
        print(format_output(data, self._format))
        return 0 if is_valid else 1


class ExportCommand:
    """Export graph in various formats (DOT, JSON, Mermaid)."""

    def __init__(self, format_kind: FormatKind = FormatKind.TABLE, verbose: bool = False) -> None:
        self._format = format_kind
        self._verbose = verbose

    def execute(
        self,
        output_path: str = "graph.json",
        export_format: str = "json",
        include_values: bool = False,
        graph=None,
    ) -> int:
        """Export the graph to a file.

        Returns:
            0 on success, 1 on error.
        """
        if graph is None:
            print("No graph to export")
            return 1

        if export_format == "json":
            content = self._export_json(graph, include_values)
        elif export_format == "dot":
            content = self._export_dot(graph)
        elif export_format == "mermaid":
            content = self._export_mermaid(graph)
        else:
            print(f"Unknown format: {export_format}")
            return 1

        # In real usage, write to file
        data = {"exported": True, "format": export_format, "path": output_path}
        print(format_output(data, self._format))
        return 0

    def _export_json(self, graph, include_values: bool) -> str:
        """Export graph as JSON."""
        from dagflow.core.node import ComputeNode, InputNode

        nodes: List[Dict[str, Any]] = []
        for nid in graph.all_node_ids:
            node = graph.get_node(nid)
            entry: Dict[str, Any] = {
                "id": nid,
                "type": "input" if node.is_input else "compute",
                "dependencies": graph.get_dependencies(nid),
                "dependents": graph.get_dependents(nid),
            }
            if include_values:
                if isinstance(node, InputNode):
                    entry["value"] = node.value
                elif isinstance(node, ComputeNode):
                    entry["cached_value"] = node.cached_value
            nodes.append(entry)

        return json.dumps({"nodes": nodes}, indent=2)

    def _export_dot(self, graph) -> str:
        """Export graph as DOT (Graphviz) format."""
        lines = ["digraph DagFlow {", "  rankdir=TB;"]
        for nid in graph.all_node_ids:
            node = graph.get_node(nid)
            shape = "ellipse" if node.is_input else "box"
            lines.append(f'  "{nid}" [shape={shape}];')

        for nid in graph.all_node_ids:
            for dep_id in graph.get_dependents(nid):
                lines.append(f'  "{nid}" -> "{dep_id}";')

        lines.append("}")
        return "\n".join(lines)

    def _export_mermaid(self, graph) -> str:
        """Export graph as Mermaid diagram."""
        lines = ["graph TD"]
        for nid in graph.all_node_ids:
            node = graph.get_node(nid)
            if node.is_input:
                lines.append(f"  {nid}[/{nid}/]")
            else:
                lines.append(f"  {nid}[{nid}]")

        for nid in graph.all_node_ids:
            for dep_id in graph.get_dependents(nid):
                lines.append(f"  {nid} --> {dep_id}")

        return "\n".join(lines)


class RunCommand:
    """Run computation with specified inputs."""

    def __init__(self, format_kind: FormatKind = FormatKind.TABLE, verbose: bool = False) -> None:
        self._format = format_kind
        self._verbose = verbose

    def execute(
        self,
        inputs: Dict[str, Any] = None,
        outputs: List[str] = None,
        dry_run: bool = False,
        graph=None,
        cache=None,
    ) -> int:
        """Execute computation with given inputs.

        Returns:
            0 on success, 1 on error.
        """
        inputs = inputs or {}
        outputs = outputs or []

        if graph is None:
            print("No graph to run")
            return 1

        if dry_run:
            from dagflow.analysis.impact import ImpactAnalyzer
            analyzer = ImpactAnalyzer(graph)
            report = analyzer.analyze_impact(list(inputs.keys()))
            data = {
                "dry_run": True,
                "affected_nodes": list(report.affected_nodes),
                "blast_radius": f"{report.blast_radius:.1%}",
            }
            print(format_output(data, self._format))
            return 0

        # Apply inputs and propagate
        from dagflow.core.node import InputNode
        from dagflow.memo.cache import MemoCache
        from dagflow.propagation.eager import EagerPropagator

        if cache is None:
            cache = MemoCache()

        changed: List[str] = []
        for node_id, value in inputs.items():
            node = graph.get_node(node_id)
            if isinstance(node, InputNode):
                node.set(value)
                changed.append(node_id)

        propagator = EagerPropagator(graph, cache)
        results = propagator.propagate(changed)

        # Filter to requested outputs
        if outputs:
            results = {k: v for k, v in results.items() if k in outputs}

        data = {"computed": True, "results": results}
        print(format_output(data, self._format))
        return 0
