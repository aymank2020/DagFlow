"""DotExporter — export computation graph to DOT/Graphviz format.

Generates DOT language output for visualization of the computation DAG.
Supports customization of node colors based on state, edge styles,
and optional value annotations.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode, NodeState)
- memo.cache (MemoCache)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode, NodeState
from dagflow.memo.cache import MemoCache


# Default color scheme for node states
_STATE_COLORS: Dict[NodeState, str] = {
    NodeState.CLEAN: "#90EE90",     # Light green
    NodeState.DIRTY: "#FFB6C1",     # Light pink
    NodeState.PENDING: "#FFFACD",   # Lemon chiffon
    NodeState.COMPUTING: "#87CEEB",  # Sky blue
}

_INPUT_COLOR = "#ADD8E6"  # Light blue


class DotExporter:
    """Exports a ComputeGraph to DOT format for Graphviz visualization.

    Usage:
        exporter = DotExporter(graph, cache)
        dot_string = exporter.export()
        # Write to file: open("graph.dot", "w").write(dot_string)

    Customization:
        exporter.show_values = True   # Annotate nodes with current values
        exporter.show_states = True   # Color nodes by state
        exporter.highlight = {"x"}    # Highlight specific nodes
    """

    def __init__(
        self,
        graph: ComputeGraph,
        cache: Optional[MemoCache] = None,
    ) -> None:
        self._graph = graph
        self._cache = cache
        self.show_values: bool = False
        self.show_states: bool = True
        self.highlight: Set[str] = set()
        self.graph_name: str = "DagFlow"
        self.rankdir: str = "TB"  # Top-to-bottom layout

    def export(self) -> str:
        """Generate DOT format string for the entire graph.

        Returns:
            A valid DOT language string.
        """
        lines: List[str] = []
        lines.append(f'digraph "{self.graph_name}" {{')
        lines.append(f"  rankdir={self.rankdir};")
        lines.append('  node [shape=box, style=filled, fontname="Helvetica"];')
        lines.append("")

        # Emit node definitions
        for node_id in self._graph.all_node_ids:
            node_def = self._format_node(node_id)
            lines.append(f"  {node_def}")

        lines.append("")

        # Emit edges
        for node_id in self._graph.all_node_ids:
            for dep_id in self._graph.get_dependents(node_id):
                edge_attrs = self._format_edge(node_id, dep_id)
                lines.append(f'  "{node_id}" -> "{dep_id}"{edge_attrs};')

        lines.append("}")
        return "\n".join(lines)

    def export_subgraph(self, root_id: str) -> str:
        """Export only the subgraph reachable downstream from root_id.

        Args:
            root_id: The starting node.

        Returns:
            DOT string for the subgraph.
        """
        downstream = set(self._graph.get_all_downstream(root_id))
        downstream.add(root_id)

        lines: List[str] = []
        lines.append(f'digraph "{self.graph_name}_sub" {{')
        lines.append(f"  rankdir={self.rankdir};")
        lines.append('  node [shape=box, style=filled, fontname="Helvetica"];')
        lines.append("")

        for node_id in downstream:
            node_def = self._format_node(node_id)
            lines.append(f"  {node_def}")

        lines.append("")

        for node_id in downstream:
            for dep_id in self._graph.get_dependents(node_id):
                if dep_id in downstream:
                    edge_attrs = self._format_edge(node_id, dep_id)
                    lines.append(f'  "{node_id}" -> "{dep_id}"{edge_attrs};')

        lines.append("}")
        return "\n".join(lines)

    def _format_node(self, node_id: str) -> str:
        """Format a single node definition in DOT syntax."""
        node = self._graph.get_node(node_id)
        attrs: Dict[str, str] = {}

        # Label
        label_parts = [node_id]
        if self.show_values:
            value = self._get_display_value(node_id)
            if value is not None:
                label_parts.append(f"= {value}")
        attrs["label"] = "\\n".join(label_parts)

        # Color
        if node_id in self.highlight:
            attrs["fillcolor"] = "#FFD700"  # Gold for highlighted
            attrs["penwidth"] = "2"
        elif isinstance(node, InputNode):
            attrs["fillcolor"] = _INPUT_COLOR
            attrs["shape"] = "ellipse"
        elif isinstance(node, ComputeNode) and self.show_states:
            attrs["fillcolor"] = _STATE_COLORS.get(node.state, "#FFFFFF")

        # Format attributes string
        attr_str = ", ".join(f'{k}="{v}"' for k, v in attrs.items())
        return f'"{node_id}" [{attr_str}];'

    def _format_edge(self, from_id: str, to_id: str) -> str:
        """Format edge attributes."""
        if from_id in self.highlight or to_id in self.highlight:
            return ' [color="red", penwidth=2]'
        return ""

    def _get_display_value(self, node_id: str) -> Optional[str]:
        """Get a display-friendly value for a node."""
        node = self._graph.get_node(node_id)
        if isinstance(node, InputNode):
            return self._truncate_value(node.value)
        elif self._cache is not None:
            cached = self._cache.get(node_id)
            if cached is not None:
                return self._truncate_value(cached)
        return None

    @staticmethod
    def _truncate_value(value: Any, max_len: int = 20) -> str:
        """Truncate a value's string representation for display."""
        s = repr(value)
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s
