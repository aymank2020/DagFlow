"""Integrity — structural integrity checks for the computation graph.

Detects structural problems in the graph such as orphan nodes (no
dependents and not an output), unreachable nodes, inconsistent edge
references, and state anomalies.

This module depends on:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode, NodeState)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode, NodeState


class IssueSeverity(Enum):
    """Severity level of an integrity issue."""

    WARNING = auto()   # Potential problem, graph still functional
    ERROR = auto()     # Definite problem that may cause incorrect results


class IssueType(Enum):
    """Types of integrity issues."""

    ORPHAN_INPUT = auto()        # Input with no dependents
    UNREACHABLE_COMPUTE = auto()  # Compute node not reachable from any input
    MISSING_DEPENDENCY = auto()   # Node references a dependency that doesn't exist
    INCONSISTENT_EDGE = auto()    # Edge in adjacency but not in node's dep list
    DIRTY_WITH_NO_PATH = auto()   # Dirty node with no path to a changed input
    STALE_GENERATION = auto()     # Generation counter inconsistency


@dataclass
class IntegrityIssue:
    """A detected integrity issue.

    Attributes:
        issue_type: Classification of the issue.
        severity: How serious the issue is.
        node_id: The node involved.
        message: Human-readable description.
        related_nodes: Other nodes involved in the issue.
    """

    issue_type: IssueType
    severity: IssueSeverity
    node_id: str
    message: str
    related_nodes: List[str]


class IntegrityChecker:
    """Performs structural integrity checks on a ComputeGraph.

    Usage:
        checker = IntegrityChecker(graph)
        issues = checker.check_all()
        for issue in issues:
            print(f"[{issue.severity.name}] {issue.message}")
    """

    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def check_all(self) -> List[IntegrityIssue]:
        """Run all integrity checks and return all issues found.

        Returns:
            List of IntegrityIssue objects, sorted by severity.
        """
        issues: List[IntegrityIssue] = []
        issues.extend(self.check_orphan_inputs())
        issues.extend(self.check_unreachable_nodes())
        issues.extend(self.check_edge_consistency())
        issues.extend(self.check_state_consistency())

        # Sort: errors first, then warnings
        issues.sort(key=lambda i: (0 if i.severity == IssueSeverity.ERROR else 1))
        return issues

    def check_orphan_inputs(self) -> List[IntegrityIssue]:
        """Find input nodes that have no dependents.

        An orphan input is never read by any compute node, which usually
        indicates a configuration error.

        Returns:
            List of issues for orphan inputs.
        """
        issues: List[IntegrityIssue] = []

        for node_id in self._graph.get_input_nodes():
            dependents = self._graph.get_dependents(node_id)
            if not dependents:
                issues.append(
                    IntegrityIssue(
                        issue_type=IssueType.ORPHAN_INPUT,
                        severity=IssueSeverity.WARNING,
                        node_id=node_id,
                        message=f"Input node '{node_id}' has no dependents",
                        related_nodes=[],
                    )
                )

        return issues

    def check_unreachable_nodes(self) -> List[IntegrityIssue]:
        """Find compute nodes not reachable from any input node.

        A compute node should be transitively reachable from at least one
        input. If not, it can never receive fresh data.

        Returns:
            List of issues for unreachable nodes.
        """
        # BFS from all inputs to find reachable compute nodes
        reachable: Set[str] = set()
        queue: deque = deque()

        for input_id in self._graph.get_input_nodes():
            queue.append(input_id)
            reachable.add(input_id)

        while queue:
            current = queue.popleft()
            for dep_id in self._graph.get_dependents(current):
                if dep_id not in reachable:
                    reachable.add(dep_id)
                    queue.append(dep_id)

        # Check which compute nodes are not reachable
        issues: List[IntegrityIssue] = []
        for node_id in self._graph.get_compute_nodes():
            if node_id not in reachable:
                issues.append(
                    IntegrityIssue(
                        issue_type=IssueType.UNREACHABLE_COMPUTE,
                        severity=IssueSeverity.ERROR,
                        node_id=node_id,
                        message=(
                            f"Compute node '{node_id}' is not reachable "
                            f"from any input node"
                        ),
                        related_nodes=[],
                    )
                )

        return issues

    def check_edge_consistency(self) -> List[IntegrityIssue]:
        """Verify that edge references are consistent.

        Checks that:
        - Every dependency listed in a ComputeNode exists in the graph.
        - Every dependent relationship is bidirectional.

        Returns:
            List of issues for inconsistent edges.
        """
        issues: List[IntegrityIssue] = []

        for node_id in self._graph.get_compute_nodes():
            node = self._graph.get_node(node_id)
            if not isinstance(node, ComputeNode):
                continue

            for dep_id in node.dependencies:
                # Check dependency exists
                try:
                    self._graph.get_node(dep_id)
                except KeyError:
                    issues.append(
                        IntegrityIssue(
                            issue_type=IssueType.MISSING_DEPENDENCY,
                            severity=IssueSeverity.ERROR,
                            node_id=node_id,
                            message=(
                                f"Node '{node_id}' depends on '{dep_id}' "
                                f"which doesn't exist"
                            ),
                            related_nodes=[dep_id],
                        )
                    )
                    continue

                # Check bidirectional consistency
                dependents_of_dep = self._graph.get_dependents(dep_id)
                if node_id not in dependents_of_dep:
                    issues.append(
                        IntegrityIssue(
                            issue_type=IssueType.INCONSISTENT_EDGE,
                            severity=IssueSeverity.ERROR,
                            node_id=node_id,
                            message=(
                                f"Node '{node_id}' lists '{dep_id}' as dependency, "
                                f"but '{dep_id}' doesn't list '{node_id}' as dependent"
                            ),
                            related_nodes=[dep_id],
                        )
                    )

        return issues

    def check_state_consistency(self) -> List[IntegrityIssue]:
        """Check for state anomalies in compute nodes.

        Detects cases where a node is CLEAN but has DIRTY dependencies,
        which would indicate a propagation bug.

        Returns:
            List of issues for state inconsistencies.
        """
        issues: List[IntegrityIssue] = []

        for node_id in self._graph.get_compute_nodes():
            node = self._graph.get_node(node_id)
            if not isinstance(node, ComputeNode):
                continue

            if node.state == NodeState.CLEAN:
                # Check if any dependency is dirty
                for dep_id in node.dependencies:
                    dep_node = self._graph.get_node(dep_id)
                    if (
                        isinstance(dep_node, ComputeNode)
                        and dep_node.state == NodeState.DIRTY
                    ):
                        issues.append(
                            IntegrityIssue(
                                issue_type=IssueType.DIRTY_WITH_NO_PATH,
                                severity=IssueSeverity.WARNING,
                                node_id=node_id,
                                message=(
                                    f"Node '{node_id}' is CLEAN but dependency "
                                    f"'{dep_id}' is DIRTY"
                                ),
                                related_nodes=[dep_id],
                            )
                        )

        return issues

    def is_healthy(self) -> bool:
        """Quick check: does the graph have any ERROR-level issues?

        Returns:
            True if no errors found (warnings are acceptable).
        """
        issues = self.check_all()
        return not any(i.severity == IssueSeverity.ERROR for i in issues)
