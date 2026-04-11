"""Invalidator — marks downstream nodes dirty after an input change.

Performs a BFS/DFS walk from changed nodes, marking dependents as DIRTY.
Implements "early cutoff": if a node's dependencies haven't actually changed
their *value* (generation unchanged), propagation stops at that node.

This module depends on:
- core.graph (ComputeGraph) for traversal
- core.node (NodeState, ComputeNode) for state transitions
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode, NodeState


class Invalidator:
    """Walks the DAG downstream from changed inputs, marking nodes dirty.

    Supports early-cutoff optimization: if a compute node is re-evaluated
    and produces the same value as before, its dependents are NOT invalidated.
    """

    def __init__(self, graph: ComputeGraph) -> None:
        self._graph = graph

    def invalidate_from(self, changed_ids: List[str]) -> Set[str]:
        """Mark all downstream compute nodes as DIRTY starting from changed_ids.

        Uses BFS to ensure parents are invalidated before children.
        Returns the set of node IDs that were marked dirty.

        Args:
            changed_ids: List of node IDs whose values have changed.

        Returns:
            Set of node IDs that were newly marked DIRTY.
        """
        newly_dirty: Set[str] = set()
        queue: deque = deque()

        # Seed: immediate dependents of changed nodes
        for cid in changed_ids:
            for dep_id in self._graph.get_dependents(cid):
                if dep_id not in newly_dirty:
                    queue.append(dep_id)
                    newly_dirty.add(dep_id)

        # BFS propagation
        while queue:
            current_id = queue.popleft()
            node = self._graph.get_node(current_id)
            if isinstance(node, ComputeNode):
                node.mark_dirty()

            # Propagate to dependents
            for dep_id in self._graph.get_dependents(current_id):
                if dep_id not in newly_dirty:
                    newly_dirty.add(dep_id)
                    queue.append(dep_id)

        return newly_dirty

    def invalidate_selective(
        self, changed_ids: List[str], generation_map: Dict[str, int]
    ) -> Set[str]:
        """Selective invalidation with early-cutoff awareness.

        Only invalidates a node if at least one of its direct dependencies
        has a generation newer than what the node last saw. This prevents
        unnecessary propagation through diamond dependencies.

        Args:
            changed_ids: Node IDs that changed.
            generation_map: Current {node_id: generation} for all nodes.

        Returns:
            Set of node IDs that were marked dirty.
        """
        newly_dirty: Set[str] = set()
        queue: deque = deque()

        # Seed with immediate dependents
        for cid in changed_ids:
            for dep_id in self._graph.get_dependents(cid):
                queue.append(dep_id)

        while queue:
            current_id = queue.popleft()
            if current_id in newly_dirty:
                continue

            node = self._graph.get_node(current_id)
            if not isinstance(node, ComputeNode):
                continue

            # Check if any dependency actually has a newer generation
            should_dirty = False
            for dep_id in node.dependencies:
                current_gen = generation_map.get(dep_id, 0)
                last_seen = node.last_valid_generation.get(dep_id, -1)
                if current_gen > last_seen:
                    should_dirty = True
                    break

            if should_dirty:
                node.mark_dirty()
                newly_dirty.add(current_id)
                # Propagate further
                for dep_id in self._graph.get_dependents(current_id):
                    if dep_id not in newly_dirty:
                        queue.append(dep_id)

        return newly_dirty
