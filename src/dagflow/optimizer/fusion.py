"""Node fusion — merge sequential compute nodes into single nodes.

Identifies chains of compute nodes where each has exactly one dependent
and one dependency (linear chains), and fuses them into a single node
that executes the composed function. This reduces scheduling overhead
and improves cache locality.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode, InputNode)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode


@dataclass
class FusionCandidate:
    """A chain of nodes eligible for fusion.

    Attributes:
        chain: Ordered list of node IDs forming the linear chain.
        head_deps: Dependencies of the first node in the chain.
        tail_dependents: Dependents of the last node in the chain.
        estimated_savings: Estimated overhead reduction from fusion.
    """

    chain: List[str]
    head_deps: List[str] = field(default_factory=list)
    tail_dependents: List[str] = field(default_factory=list)
    estimated_savings: float = 0.0

    @property
    def length(self) -> int:
        """Number of nodes in the chain."""
        return len(self.chain)

    @property
    def is_valid(self) -> bool:
        """A chain must have at least 2 nodes to be worth fusing."""
        return self.length >= 2


@dataclass
class FusionResult:
    """Result of a fusion optimization pass.

    Attributes:
        fused_count: Number of fusion operations performed.
        nodes_removed: Total nodes eliminated by fusion.
        candidates_found: Number of fusible chains identified.
        candidates_skipped: Chains skipped (too short or constrained).
        fused_node_ids: IDs of the new fused nodes created.
    """

    fused_count: int = 0
    nodes_removed: int = 0
    candidates_found: int = 0
    candidates_skipped: int = 0
    fused_node_ids: List[str] = field(default_factory=list)


class NodeFuser:
    """Identifies and fuses linear chains of compute nodes.

    A linear chain is a sequence [A, B, C] where:
    - A's only dependent is B
    - B's only dependency is A, and only dependent is C
    - C's only dependency is B

    After fusion, the chain becomes a single node that computes
    C(B(A(inputs))) in one step.

    Usage:
        fuser = NodeFuser(graph)
        candidates = fuser.find_candidates()
        result = fuser.apply(candidates)
    """

    def __init__(
        self,
        graph: ComputeGraph,
        min_chain_length: int = 2,
        max_chain_length: int = 10,
    ) -> None:
        self._graph = graph
        self._min_chain_length = max(2, min_chain_length)
        self._max_chain_length = max_chain_length

    def find_candidates(self) -> List[FusionCandidate]:
        """Identify all linear chains eligible for fusion.

        Walks the graph looking for sequences of compute nodes where
        each interior node has exactly one upstream and one downstream
        connection within the chain.

        Returns:
            List of FusionCandidate objects, longest chains first.
        """
        visited: Set[str] = set()
        candidates: List[FusionCandidate] = []

        for node_id in self._graph.get_compute_nodes():
            if node_id in visited:
                continue

            chain = self._extend_chain_forward(node_id, visited)
            if len(chain) >= self._min_chain_length:
                # Trim to max length
                chain = chain[: self._max_chain_length]
                head_node = self._graph.get_node(chain[0])
                tail_node = self._graph.get_node(chain[-1])

                candidate = FusionCandidate(
                    chain=chain,
                    head_deps=(
                        head_node.dependencies
                        if isinstance(head_node, ComputeNode)
                        else []
                    ),
                    tail_dependents=(
                        tail_node.dependents
                        if isinstance(tail_node, ComputeNode)
                        else []
                    ),
                    estimated_savings=self._estimate_savings(chain),
                )
                candidates.append(candidate)

            visited.update(chain)

        # Sort by chain length descending (fuse longest chains first)
        candidates.sort(key=lambda c: c.length, reverse=True)
        return candidates

    def apply(
        self,
        candidates: Optional[List[FusionCandidate]] = None,
        dry_run: bool = False,
    ) -> FusionResult:
        """Apply fusion to identified candidates.

        Creates a composed function for each chain and registers it as
        a new node, removing the intermediate nodes from the graph.

        Args:
            candidates: Chains to fuse. If None, finds them automatically.
            dry_run: If True, only count what would be fused without modifying.

        Returns:
            FusionResult with statistics about the operation.
        """
        if candidates is None:
            candidates = self.find_candidates()

        result = FusionResult(candidates_found=len(candidates))

        for candidate in candidates:
            if not candidate.is_valid:
                result.candidates_skipped += 1
                continue

            if dry_run:
                result.fused_count += 1
                result.nodes_removed += candidate.length - 1
                continue

            fused_id = self._fuse_chain(candidate)
            if fused_id:
                result.fused_count += 1
                result.nodes_removed += candidate.length - 1
                result.fused_node_ids.append(fused_id)
            else:
                result.candidates_skipped += 1

        return result

    def compose_functions(
        self, chain: List[str]
    ) -> Callable[[Dict[str, Any]], Any]:
        """Create a composed function from a chain of node functions.

        The composed function feeds the output of each node as input
        to the next, using the chain's first node's dependencies as
        the initial input.

        Args:
            chain: Ordered list of node IDs to compose.

        Returns:
            A function that computes the entire chain in one call.
        """
        functions: List[Tuple[str, Callable]] = []
        for node_id in chain:
            node = self._graph.get_node(node_id)
            if isinstance(node, ComputeNode):
                functions.append((node_id, node.func))

        def composed(inputs: Dict[str, Any]) -> Any:
            current_value = inputs
            for i, (nid, func) in enumerate(functions):
                if i == 0:
                    current_value = func(inputs)
                else:
                    # Feed previous output as the sole dependency value
                    prev_id = chain[i - 1]
                    current_value = func({prev_id: current_value})
            return current_value

        return composed

    def _extend_chain_forward(
        self, start_id: str, visited: Set[str]
    ) -> List[str]:
        """Extend a chain without crossing externally observed intermediates."""
        chain = [start_id]
        current = start_id

        while True:
            node = self._graph.get_node(current)
            if not isinstance(node, ComputeNode):
                break

            dependents = [
                d for d in node.dependents if d not in visited
            ]

            # A fused node replaces the chain tail. We may only absorb the
            # next node when the current node is observed by exactly one
            # downstream node; otherwise another branch still reads this
            # intermediate value and fusing through it would drop it.
            if len(dependents) != 1:
                break

            next_id = dependents[0]
            next_node = self._graph.get_node(next_id)

            if not isinstance(next_node, ComputeNode):
                break
            if len(next_node.dependencies) != 1:
                break

            chain.append(next_id)
            current = next_id

            if len(chain) >= self._max_chain_length:
                break

        return chain

    def _fuse_chain(self, candidate: FusionCandidate) -> Optional[str]:
        """Fuse a chain into its tail node, removing the interior nodes.

        The fused node keeps the tail's identity so any external branch that
        still observes the tail value keeps reading the same node. Chain
        selection guarantees every interior node has exactly one in-chain
        dependent, so removing the interior drops no externally observed
        edge. Returns the fused node ID or None.
        """
        chain = candidate.chain
        if len(chain) < 2:
            return None

        tail_id = chain[-1]
        tail_node = self._graph.get_node(tail_id)
        if not isinstance(tail_node, ComputeNode):
            return None

        # Compose the chain functions before mutating any node so the
        # closure captures the original tail function, not the new one.
        composed_fn = self.compose_functions(chain)
        head_deps = list(candidate.head_deps)

        # Remove the interior nodes (everything except the tail).
        for node_id in chain[:-1]:
            try:
                self._graph.remove_node(node_id)
            except KeyError:
                pass

        # Re-point the tail at the head's dependencies and let it compute
        # the whole composed chain in one step. The tail's own dependents
        # are left untouched, so their stored references remain valid.
        tail_node.func = composed_fn
        tail_node.dependencies = list(head_deps)
        self._graph._reverse[tail_id] = set(head_deps)
        for dep_id in head_deps:
            self._graph._adjacency[dep_id].add(tail_id)
            dep_node = self._graph.get_node(dep_id)
            if tail_id not in dep_node.dependents:
                dep_node.dependents.append(tail_id)

        return tail_id

    def _estimate_savings(self, chain: List[str]) -> float:
        """Estimate scheduling overhead saved by fusing a chain.

        Each eliminated node saves approximately one scheduling decision
        and one cache lookup. Longer chains save more.
        """
        # Base cost per node: scheduling + cache lookup + function call overhead
        per_node_overhead = 0.001  # 1ms estimated
        # Fused node still has one function call
        return (len(chain) - 1) * per_node_overhead
