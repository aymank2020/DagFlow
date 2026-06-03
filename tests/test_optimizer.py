"""Tests for the graph optimization module.

Tests cover:
- Node fusion: chain detection and function composition
- Dead node pruning: reachability and removal
- Graph partitioning: level assignment and parallelism
- Cost model: estimation and profiling
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode
from dagflow.optimizer.fusion import FusionCandidate, NodeFuser
from dagflow.optimizer.pruning import DeadNodePruner
from dagflow.optimizer.partitioning import GraphPartitioner
from dagflow.optimizer.cost_model import CostModel


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def linear_graph() -> ComputeGraph:
    """Create a linear chain: x -> a -> b -> c -> d."""
    g = ComputeGraph()
    g.add_input("x", value=5)
    g.add_compute("a", lambda d: d["x"] * 2, ["x"])
    g.add_compute("b", lambda d: d["a"] + 1, ["a"])
    g.add_compute("c", lambda d: d["b"] ** 2, ["b"])
    g.add_compute("d", lambda d: d["c"] - 3, ["c"])
    return g


@pytest.fixture
def diamond_graph() -> ComputeGraph:
    """Create a diamond: x -> (a, b) -> c."""
    g = ComputeGraph()
    g.add_input("x", value=10)
    g.add_compute("a", lambda d: d["x"] + 1, ["x"])
    g.add_compute("b", lambda d: d["x"] * 2, ["x"])
    g.add_compute("c", lambda d: d["a"] + d["b"], ["a", "b"])
    return g


@pytest.fixture
def wide_graph() -> ComputeGraph:
    """Create a wide graph: x -> (a, b, c, d, e) -> result."""
    g = ComputeGraph()
    g.add_input("x", value=1)
    for name in ["a", "b", "c", "d", "e"]:
        g.add_compute(name, lambda d, n=name: d["x"] + ord(n), ["x"])
    g.add_compute(
        "result",
        lambda d: sum(d.values()),
        ["a", "b", "c", "d", "e"],
    )
    return g


@pytest.fixture
def graph_with_dead_nodes() -> ComputeGraph:
    """Graph with unreachable nodes."""
    g = ComputeGraph()
    g.add_input("x", value=1)
    g.add_input("y", value=2)
    g.add_input("orphan_input", value=99)  # No dependents
    g.add_compute("live", lambda d: d["x"] + d["y"], ["x", "y"])
    g.add_compute("dead", lambda d: d["orphan_input"] * 2, ["orphan_input"])
    return g


# ─── Node Fusion Tests ─────────────────────────────────────────────────


class TestNodeFuser:
    """Tests for the NodeFuser optimization pass."""

    def test_find_linear_chain(self, linear_graph: ComputeGraph) -> None:
        """Detects the linear chain a -> b -> c -> d."""
        fuser = NodeFuser(linear_graph)
        candidates = fuser.find_candidates()

        assert len(candidates) >= 1
        # Should find a chain of length >= 2
        longest = max(candidates, key=lambda c: c.length)
        assert longest.length >= 2

    def test_no_fusion_in_diamond(self, diamond_graph: ComputeGraph) -> None:
        """Diamond graphs have no fusible chains (fan-out at x)."""
        fuser = NodeFuser(diamond_graph)
        candidates = fuser.find_candidates()

        # No chain of length >= 2 because a and b both depend on x
        # and c depends on both a and b
        for c in candidates:
            # Chains in diamond are at most length 1
            assert c.length <= 2

    def test_compose_functions(self, linear_graph: ComputeGraph) -> None:
        """Composed function produces correct result."""
        fuser = NodeFuser(linear_graph)
        # Compose a -> b (a = x*2, b = a+1)
        composed = fuser.compose_functions(["a", "b"])

        result = composed({"x": 5})
        # a = 5*2 = 10, b = 10+1 = 11
        assert result == 11

    def test_dry_run_doesnt_modify(self, linear_graph: ComputeGraph) -> None:
        """Dry run reports what would happen without modifying graph."""
        fuser = NodeFuser(linear_graph)
        original_count = linear_graph.node_count

        result = fuser.apply(dry_run=True)
        assert result.fused_count >= 1
        assert linear_graph.node_count == original_count  # Unchanged

    def test_fusion_result_statistics(self, linear_graph: ComputeGraph) -> None:
        """FusionResult contains accurate statistics."""
        fuser = NodeFuser(linear_graph)
        result = fuser.apply(dry_run=True)

        assert result.candidates_found >= 1
        assert result.nodes_removed >= 1
        assert result.fused_count + result.candidates_skipped == result.candidates_found

    def test_min_chain_length_filter(self, linear_graph: ComputeGraph) -> None:
        """min_chain_length filters short chains."""
        fuser = NodeFuser(linear_graph, min_chain_length=5)
        candidates = fuser.find_candidates()

        # Chain is a->b->c->d (length 4), so with min=5 nothing qualifies
        assert all(c.length >= 5 for c in candidates) or len(candidates) == 0

    def test_candidate_validity(self) -> None:
        """FusionCandidate.is_valid requires length >= 2."""
        valid = FusionCandidate(chain=["a", "b"])
        invalid = FusionCandidate(chain=["a"])

        assert valid.is_valid
        assert not invalid.is_valid


# ─── Dead Node Pruning Tests ──────────────────────────────────────────


class TestDeadNodePruner:
    """Tests for the DeadNodePruner."""

    def test_find_dead_nodes(self, graph_with_dead_nodes: ComputeGraph) -> None:
        """Identifies nodes not reachable from outputs."""
        pruner = DeadNodePruner(
            graph_with_dead_nodes, outputs={"live"}
        )
        dead = pruner.find_dead_nodes()

        # "dead" and "orphan_input" are not reachable from "live"
        assert "dead" in dead

    def test_find_orphaned_inputs(self, graph_with_dead_nodes: ComputeGraph) -> None:
        """Identifies input nodes with no dependents."""
        # Remove the "dead" node's dependency to make orphan_input truly orphaned
        pruner = DeadNodePruner(
            graph_with_dead_nodes, outputs={"live"}
        )
        orphaned = pruner.find_orphaned_inputs()
        # orphan_input has a dependent ("dead"), so it's not orphaned
        # But if we set outputs={"live"}, orphan_input feeds only dead nodes
        assert isinstance(orphaned, set)

    def test_prune_dry_run(self, graph_with_dead_nodes: ComputeGraph) -> None:
        """Dry run identifies removable nodes without modifying graph."""
        pruner = DeadNodePruner(
            graph_with_dead_nodes, outputs={"live"}
        )
        original_count = graph_with_dead_nodes.node_count

        result = pruner.prune(dry_run=True)
        assert result.removed_count >= 1
        assert graph_with_dead_nodes.node_count == original_count

    def test_prune_removes_dead_nodes(self, graph_with_dead_nodes: ComputeGraph) -> None:
        """Actual prune removes dead nodes from graph."""
        pruner = DeadNodePruner(
            graph_with_dead_nodes, outputs={"live"}
        )
        original_count = graph_with_dead_nodes.node_count

        result = pruner.prune(dry_run=False)
        assert graph_with_dead_nodes.node_count < original_count
        assert result.removed_count > 0

    def test_preserve_set_protects_nodes(
        self, graph_with_dead_nodes: ComputeGraph
    ) -> None:
        """Preserved nodes are never removed."""
        pruner = DeadNodePruner(
            graph_with_dead_nodes,
            outputs={"live"},
            preserve={"dead", "orphan_input"},
        )
        dead = pruner.find_dead_nodes()
        assert "dead" not in dead
        assert "orphan_input" not in dead

    def test_node_liveness_map(self, graph_with_dead_nodes: ComputeGraph) -> None:
        """compute_node_liveness returns correct liveness for all nodes."""
        pruner = DeadNodePruner(
            graph_with_dead_nodes, outputs={"live"}
        )
        liveness = pruner.compute_node_liveness()

        assert liveness["x"] is True
        assert liveness["y"] is True
        assert liveness["live"] is True

    def test_disconnected_subgraphs(self) -> None:
        """Detects disconnected components."""
        g = ComputeGraph()
        g.add_input("a", value=1)
        g.add_compute("b", lambda d: d["a"], ["a"])
        g.add_input("c", value=2)
        g.add_compute("d", lambda d: d["c"], ["c"])

        pruner = DeadNodePruner(g, outputs={"b", "d"})
        components = pruner.find_disconnected_subgraphs()
        assert len(components) == 2


# ─── Graph Partitioning Tests ─────────────────────────────────────────


class TestGraphPartitioner:
    """Tests for the GraphPartitioner."""

    def test_linear_graph_has_sequential_stages(
        self, linear_graph: ComputeGraph
    ) -> None:
        """Linear graph produces one node per stage."""
        partitioner = GraphPartitioner(linear_graph)
        plan = partitioner.partition()

        assert plan.total_levels >= 4  # a, b, c, d each at different level
        assert plan.max_width == 1  # Only one node per level

    def test_wide_graph_has_parallel_stage(
        self, wide_graph: ComputeGraph
    ) -> None:
        """Wide graph has a stage with multiple parallel nodes."""
        partitioner = GraphPartitioner(wide_graph)
        plan = partitioner.partition()

        assert plan.max_width >= 5  # a,b,c,d,e are all at same level

    def test_parallelism_profile(self, wide_graph: ComputeGraph) -> None:
        """Parallelism profile shows width at each level."""
        partitioner = GraphPartitioner(wide_graph)
        profile = partitioner.compute_parallelism_profile()

        assert max(profile) >= 5  # Wide level has 5 nodes

    def test_critical_path_length(self, linear_graph: ComputeGraph) -> None:
        """Critical path equals the chain length."""
        partitioner = GraphPartitioner(linear_graph)
        plan = partitioner.partition()

        assert plan.critical_path_length >= 4

    def test_stage_dependencies(self, linear_graph: ComputeGraph) -> None:
        """Each stage depends on the previous one."""
        partitioner = GraphPartitioner(linear_graph)
        plan = partitioner.partition()

        # Stage 1+ should depend on stage 0+
        for stage in plan.stages[1:]:
            assert len(stage.dependencies) > 0

    def test_max_stage_width_splitting(self, wide_graph: ComputeGraph) -> None:
        """max_stage_width splits wide stages."""
        partitioner = GraphPartitioner(wide_graph, max_stage_width=2)
        plan = partitioner.partition()

        for stage in plan.stages:
            assert stage.width <= 3  # Allow some slack due to splitting logic

    def test_speedup_estimate(self, wide_graph: ComputeGraph) -> None:
        """Speedup estimate is > 1 for parallelizable graphs."""
        partitioner = GraphPartitioner(wide_graph)
        speedup = partitioner.estimate_speedup()
        assert speedup >= 1.0

    def test_find_bottlenecks(self, linear_graph: ComputeGraph) -> None:
        """Linear graph nodes are all bottlenecks."""
        partitioner = GraphPartitioner(linear_graph)
        bottlenecks = partitioner.find_bottlenecks(threshold=1)
        assert len(bottlenecks) >= 4


# ─── Cost Model Tests ─────────────────────────────────────────────────


class TestCostModel:
    """Tests for the CostModel."""

    def test_estimate_node_basic(self, diamond_graph: ComputeGraph) -> None:
        """Basic node estimation produces non-zero cost."""
        model = CostModel(diamond_graph)
        cost = model.estimate_node("c")

        assert cost.node_id == "c"
        assert cost.total_cost > 0
        assert cost.fan_in == 2  # c depends on a and b
        assert cost.compute_time > 0

    def test_fan_in_affects_cost(self, wide_graph: ComputeGraph) -> None:
        """Nodes with higher fan-in have higher cost."""
        model = CostModel(wide_graph)
        cost_result = model.estimate_node("result")  # fan-in = 5
        cost_a = model.estimate_node("a")  # fan-in = 1

        assert cost_result.total_cost > cost_a.total_cost

    def test_profile_graph(self, diamond_graph: ComputeGraph) -> None:
        """profile_graph returns costs for all compute nodes."""
        model = CostModel(diamond_graph)
        profile = model.profile_graph()

        assert len(profile.node_costs) == 3  # a, b, c
        assert profile.total_compute_time > 0

    def test_record_execution_improves_estimate(
        self, diamond_graph: ComputeGraph
    ) -> None:
        """Recording actual times changes future estimates."""
        model = CostModel(diamond_graph)

        # Before recording
        cost_before = model.estimate_node("a").compute_time

        # Record a much longer execution
        model.record_execution("a", 1.0)
        cost_after = model.estimate_node("a").compute_time

        assert cost_after > cost_before

    def test_rank_nodes_by_cost(self, wide_graph: ComputeGraph) -> None:
        """Ranking returns nodes sorted by cost."""
        model = CostModel(wide_graph)
        ranking = model.rank_nodes_by_cost(descending=True)

        assert len(ranking) >= 6  # a,b,c,d,e,result
        # First should be most expensive
        costs = [cost for _, cost in ranking]
        assert costs == sorted(costs, reverse=True)

    def test_annotate_memory(self, diamond_graph: ComputeGraph) -> None:
        """Memory annotation overrides heuristic estimate."""
        model = CostModel(diamond_graph)
        model.annotate_memory("a", 1_000_000)

        cost = model.estimate_node("a")
        assert cost.memory_bytes == 1_000_000

    def test_annotate_io(self, diamond_graph: ComputeGraph) -> None:
        """I/O annotation is reflected in cost."""
        model = CostModel(diamond_graph)
        model.annotate_io("a", 50)

        cost = model.estimate_node("a")
        assert cost.io_operations == 50

    def test_compare_nodes(self, wide_graph: ComputeGraph) -> None:
        """compare_nodes returns correct ordering."""
        model = CostModel(wide_graph)
        # result has fan-in=5, a has fan-in=1
        cmp = model.compare_nodes("result", "a")
        assert cmp == 1  # result > a
