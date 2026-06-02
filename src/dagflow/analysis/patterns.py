"""Pattern detection — identify structural patterns in the DAG.

Detects common graph patterns that have performance implications:
- Diamond patterns (shared dependencies reconverging)
- Linear chains (sequential bottlenecks)
- Fan-out bottlenecks (single node feeding many)
- Fan-in aggregation points
- Isolated subgraphs

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode, InputNode)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode


class PatternKind(Enum):
    """Classification of detected graph patterns."""

    DIAMOND = auto()        # Two paths from A reconverge at B
    LINEAR_CHAIN = auto()   # Sequential nodes with no branching
    FAN_OUT = auto()        # Single node with many dependents
    FAN_IN = auto()         # Single node with many dependencies
    ISOLATED = auto()       # Disconnected subgraph
    BOTTLENECK = auto()     # Node on all paths between regions


@dataclass
class GraphPattern:
    """A detected structural pattern in the graph.

    Attributes:
        kind: The type of pattern detected.
        nodes: Set of node IDs involved in this pattern.
        root: The root/source node of the pattern (if applicable).
        sink: The sink/convergence node (if applicable).
        severity: How impactful this pattern is (0.0 to 1.0).
        description: Human-readable description of the pattern.
    """

    kind: PatternKind
    nodes: FrozenSet[str]
    root: Optional[str] = None
    sink: Optional[str] = None
    severity: float = 0.0
    description: str = ""

    @property
    def size(self) -> int:
        """Number of nodes in this pattern."""
        return len(self.nodes)


class PatternDetector:
    """Detects structural patterns in a computation graph.

    Scans the graph for known patterns that may indicate performance
    issues or optimization opportunities.

    Usage:
        detector = PatternDetector(graph)
        patterns = detector.detect_all()
        diamonds = detector.find_diamonds()
    """

    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def detect_all(self) -> List[GraphPattern]:
        """Run all pattern detectors and return combined results.

        Returns:
            List of all detected patterns, sorted by severity.
        """
        patterns: List[GraphPattern] = []
        patterns.extend(self.find_diamonds())
        patterns.extend(self.find_linear_chains())
        patterns.extend(self.find_fan_out_bottlenecks())
        patterns.extend(self.find_fan_in_aggregations())

        # Sort by severity descending
        patterns.sort(key=lambda p: p.severity, reverse=True)
        return patterns

    def find_diamonds(self) -> List[GraphPattern]:
        """Detect diamond patterns where paths diverge and reconverge.

        A diamond exists when node A has multiple paths to node B,
        meaning B depends on A through more than one intermediate path.
        Diamonds can cause redundant recomputation if not handled.

        Returns:
            List of detected diamond patterns.
        """
        diamonds: List[GraphPattern] = []
        compute_nodes = self._graph.get_compute_nodes()

        for node_id in compute_nodes:
            deps = self._graph.get_dependencies(node_id)
            if len(deps) < 2:
                continue

            # Check if any two dependencies share a common ancestor
            for i in range(len(deps)):
                ancestors_i = set(self._graph.get_all_upstream(deps[i]))
                ancestors_i.add(deps[i])

                for j in range(i + 1, len(deps)):
                    ancestors_j = set(self._graph.get_all_upstream(deps[j]))
                    ancestors_j.add(deps[j])

                    common = ancestors_i & ancestors_j
                    if common:
                        # Found a diamond: common ancestor -> two paths -> node_id
                        # Pick the closest common ancestor
                        root = self._find_closest_ancestor(common, node_id)
                        involved = {node_id, deps[i], deps[j]}
                        if root:
                            involved.add(root)

                        pattern = GraphPattern(
                            kind=PatternKind.DIAMOND,
                            nodes=frozenset(involved),
                            root=root,
                            sink=node_id,
                            severity=min(1.0, len(involved) / 10.0),
                            description=(
                                f"Diamond: paths from {root} reconverge at {node_id}"
                            ),
                        )
                        diamonds.append(pattern)

        # Deduplicate by sink node
        seen_sinks: Set[str] = set()
        unique: List[GraphPattern] = []
        for d in diamonds:
            if d.sink and d.sink not in seen_sinks:
                seen_sinks.add(d.sink)
                unique.append(d)

        return unique

    def find_linear_chains(self, min_length: int = 3) -> List[GraphPattern]:
        """Detect linear chains (sequential bottlenecks).

        A linear chain is a sequence of nodes where each has exactly
        one dependency and one dependent within the chain.

        Args:
            min_length: Minimum chain length to report.

        Returns:
            List of detected chain patterns.
        """
        chains: List[GraphPattern] = []
        visited: Set[str] = set()

        for node_id in self._graph.get_compute_nodes():
            if node_id in visited:
                continue

            chain = self._trace_chain(node_id, visited)
            if len(chain) >= min_length:
                visited.update(chain)
                pattern = GraphPattern(
                    kind=PatternKind.LINEAR_CHAIN,
                    nodes=frozenset(chain),
                    root=chain[0],
                    sink=chain[-1],
                    severity=min(1.0, len(chain) / 20.0),
                    description=(
                        f"Linear chain of {len(chain)} nodes: "
                        f"{chain[0]} -> ... -> {chain[-1]}"
                    ),
                )
                chains.append(pattern)

        return chains

    def find_fan_out_bottlenecks(self, threshold: int = 3) -> List[GraphPattern]:
        """Detect nodes with high fan-out (many dependents).

        These nodes are potential bottlenecks because a change to them
        triggers recomputation of many downstream nodes.

        Args:
            threshold: Minimum dependents to be considered a bottleneck.

        Returns:
            List of detected fan-out patterns.
        """
        patterns: List[GraphPattern] = []

        for node_id in self._graph.all_node_ids:
            dependents = self._graph.get_dependents(node_id)
            if len(dependents) >= threshold:
                involved = {node_id} | set(dependents)
                pattern = GraphPattern(
                    kind=PatternKind.FAN_OUT,
                    nodes=frozenset(involved),
                    root=node_id,
                    severity=min(1.0, len(dependents) / 10.0),
                    description=(
                        f"Fan-out: {node_id} feeds {len(dependents)} dependents"
                    ),
                )
                patterns.append(pattern)

        return patterns

    def find_fan_in_aggregations(self, threshold: int = 3) -> List[GraphPattern]:
        """Detect nodes with high fan-in (many dependencies).

        These are aggregation points that combine many inputs.

        Args:
            threshold: Minimum dependencies to be considered aggregation.

        Returns:
            List of detected fan-in patterns.
        """
        patterns: List[GraphPattern] = []

        for node_id in self._graph.get_compute_nodes():
            deps = self._graph.get_dependencies(node_id)
            if len(deps) >= threshold:
                involved = {node_id} | set(deps)
                pattern = GraphPattern(
                    kind=PatternKind.FAN_IN,
                    nodes=frozenset(involved),
                    sink=node_id,
                    severity=min(1.0, len(deps) / 10.0),
                    description=(
                        f"Fan-in: {node_id} aggregates {len(deps)} dependencies"
                    ),
                )
                patterns.append(pattern)

        return patterns

    def summarize(self) -> Dict[PatternKind, int]:
        """Get a count of each pattern type in the graph.

        Returns:
            Dict mapping PatternKind -> count.
        """
        patterns = self.detect_all()
        summary: Dict[PatternKind, int] = defaultdict(int)
        for p in patterns:
            summary[p.kind] += 1
        return dict(summary)

    def _trace_chain(self, start: str, visited: Set[str]) -> List[str]:
        """Trace a linear chain starting from a node."""
        chain = [start]
        current = start

        while True:
            node = self._graph.get_node(current)
            if not isinstance(node, ComputeNode):
                break

            dependents = self._graph.get_dependents(current)
            if len(dependents) != 1:
                break

            next_id = dependents[0]
            if next_id in visited:
                break

            next_node = self._graph.get_node(next_id)
            if not isinstance(next_node, ComputeNode):
                break
            if len(next_node.dependencies) != 1:
                break

            chain.append(next_id)
            current = next_id

        return chain

    def _find_closest_ancestor(
        self, candidates: Set[str], target: str
    ) -> Optional[str]:
        """Find the candidate closest to target in the graph."""
        if not candidates:
            return None

        # BFS backward from target
        queue: deque = deque([target])
        visited: Set[str] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current in candidates and current != target:
                return current

            for dep in self._graph.get_dependencies(current):
                if dep not in visited:
                    queue.append(dep)

        return next(iter(candidates), None)
