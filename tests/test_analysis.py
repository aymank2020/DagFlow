"""Tests for the static analysis module.

Tests cover:
- Complexity metrics: depth, width, fan-in/fan-out, density
- Pattern detection: diamonds, chains, fan-out, fan-in
- Impact analysis: blast radius, propagation, sensitivity
"""

from __future__ import annotations

import pytest

from dagflow.core.graph import ComputeGraph
from dagflow.analysis.complexity import ComplexityAnalyzer, ComplexityMetrics
from dagflow.analysis.patterns import PatternDetector, PatternKind
from dagflow.analysis.impact import ImpactAnalyzer


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def linear_graph() -> ComputeGraph:
    """x -> a -> b -> c -> d (depth=4, width=1)."""
    g = ComputeGraph()
    g.add_input("x", value=1)
    g.add_compute("a", lambda d: d["x"] * 2, ["x"])
    g.add_compute("b", lambda d: d["a"] + 1, ["a"])
    g.add_compute("c", lambda d: d["b"] ** 2, ["b"])
    g.add_compute("d", lambda d: d["c"] - 1, ["c"])
    return g


@pytest.fixture
def diamond_graph() -> ComputeGraph:
    """x -> (a, b) -> c (diamond pattern)."""
    g = ComputeGraph()
    g.add_input("x", value=10)
    g.add_compute("a", lambda d: d["x"] + 1, ["x"])
    g.add_compute("b", lambda d: d["x"] * 2, ["x"])
    g.add_compute("c", lambda d: d["a"] + d["b"], ["a", "b"])
    return g


@pytest.fixture
def wide_graph() -> ComputeGraph:
    """x -> (a, b, c, d, e) -> result."""
    g = ComputeGraph()
    g.add_input("x", value=1)
    for name in ["a", "b", "c", "d", "e"]:
        g.add_compute(name, lambda d, n=name: d["x"] + ord(n), ["x"])
    g.add_compute("result", lambda d: sum(d.values()), ["a", "b", "c", "d", "e"])
    return g


@pytest.fixture
def multi_input_graph() -> ComputeGraph:
    """Multiple inputs with varying impact."""
    g = ComputeGraph()
    g.add_input("x", value=1)
    g.add_input("y", value=2)
    g.add_input("z", value=3)
    g.add_compute("a", lambda d: d["x"] + d["y"], ["x", "y"])
    g.add_compute("b", lambda d: d["x"] * 2, ["x"])
    g.add_compute("c", lambda d: d["a"] + d["b"], ["a", "b"])
    g.add_compute("d", lambda d: d["z"] + 1, ["z"])
    return g


# ─── Complexity Analyzer Tests ─────────────────────────────────────────


class TestComplexityAnalyzer:
    """Tests for ComplexityAnalyzer metrics."""

    def test_linear_graph_depth(self, linear_graph: ComputeGraph) -> None:
        """Linear graph has depth equal to chain length + 1 (for input)."""
        analyzer = ComplexityAnalyzer(linear_graph)
        metrics = analyzer.analyze()

        assert metrics.depth >= 4  # At least 4 levels (input + 4 compute)
        assert metrics.width == 1  # Only one node per level

    def test_linear_graph_is_linear(self, linear_graph: ComputeGraph) -> None:
        """Linear graph is detected as linear."""
        analyzer = ComplexityAnalyzer(linear_graph)
        metrics = analyzer.analyze()
        assert metrics.is_linear

    def test_wide_graph_width(self, wide_graph: ComputeGraph) -> None:
        """Wide graph has width >= 5."""
        analyzer = ComplexityAnalyzer(wide_graph)
        metrics = analyzer.analyze()

        assert metrics.width >= 5
        assert metrics.is_wide  # Wider than deep

    def test_node_counts(self, diamond_graph: ComputeGraph) -> None:
        """Metrics report correct node counts."""
        analyzer = ComplexityAnalyzer(diamond_graph)
        metrics = analyzer.analyze()

        assert metrics.total_nodes == 4  # x, a, b, c
        assert metrics.input_count == 1
        assert metrics.compute_count == 3

    def test_fan_in_metrics(self, diamond_graph: ComputeGraph) -> None:
        """Fan-in metrics are computed correctly."""
        analyzer = ComplexityAnalyzer(diamond_graph)
        metrics = analyzer.analyze()

        assert metrics.max_fan_in == 2  # c depends on a and b
        assert metrics.avg_fan_in > 0

    def test_fan_out_metrics(self, wide_graph: ComputeGraph) -> None:
        """Fan-out metrics reflect the wide structure."""
        analyzer = ComplexityAnalyzer(wide_graph)
        metrics = analyzer.analyze()

        assert metrics.max_fan_out >= 5  # x feeds 5 nodes

    def test_density(self, linear_graph: ComputeGraph) -> None:
        """Density is between 0 and 1."""
        analyzer = ComplexityAnalyzer(linear_graph)
        metrics = analyzer.analyze()

        assert 0.0 <= metrics.density <= 1.0

    def test_fan_in_distribution(self, diamond_graph: ComputeGraph) -> None:
        """Fan-in distribution counts nodes per fan-in value."""
        analyzer = ComplexityAnalyzer(diamond_graph)
        dist = analyzer.fan_in_distribution()

        assert 0 in dist  # x has fan-in 0
        assert 1 in dist  # a, b have fan-in 1
        assert 2 in dist  # c has fan-in 2

    def test_level_profile(self, wide_graph: ComputeGraph) -> None:
        """Level profile shows node count at each depth."""
        analyzer = ComplexityAnalyzer(wide_graph)
        profile = analyzer.level_profile()

        assert len(profile) >= 2
        assert max(profile) >= 5  # Wide level

    def test_parallelism_ratio(self, wide_graph: ComputeGraph) -> None:
        """Wide graphs have high parallelism ratio."""
        analyzer = ComplexityAnalyzer(wide_graph)
        metrics = analyzer.analyze()

        assert metrics.parallelism_ratio > 1.0


# ─── Pattern Detection Tests ──────────────────────────────────────────


class TestPatternDetector:
    """Tests for PatternDetector."""

    def test_detect_diamond(self, diamond_graph: ComputeGraph) -> None:
        """Detects diamond pattern in diamond graph."""
        detector = PatternDetector(diamond_graph)
        diamonds = detector.find_diamonds()

        assert len(diamonds) >= 1
        assert diamonds[0].kind == PatternKind.DIAMOND
        assert diamonds[0].sink == "c"

    def test_detect_linear_chain(self, linear_graph: ComputeGraph) -> None:
        """Detects linear chain in sequential graph."""
        detector = PatternDetector(linear_graph)
        chains = detector.find_linear_chains(min_length=3)

        assert len(chains) >= 1
        assert chains[0].kind == PatternKind.LINEAR_CHAIN
        assert chains[0].size >= 3

    def test_detect_fan_out(self, wide_graph: ComputeGraph) -> None:
        """Detects fan-out bottleneck at input node."""
        detector = PatternDetector(wide_graph)
        fan_outs = detector.find_fan_out_bottlenecks(threshold=3)

        assert len(fan_outs) >= 1
        assert fan_outs[0].kind == PatternKind.FAN_OUT
        assert fan_outs[0].root == "x"

    def test_detect_fan_in(self, wide_graph: ComputeGraph) -> None:
        """Detects fan-in aggregation at result node."""
        detector = PatternDetector(wide_graph)
        fan_ins = detector.find_fan_in_aggregations(threshold=3)

        assert len(fan_ins) >= 1
        assert fan_ins[0].kind == PatternKind.FAN_IN
        assert fan_ins[0].sink == "result"

    def test_detect_all_returns_sorted(self, diamond_graph: ComputeGraph) -> None:
        """detect_all returns patterns sorted by severity."""
        detector = PatternDetector(diamond_graph)
        patterns = detector.detect_all()

        if len(patterns) >= 2:
            severities = [p.severity for p in patterns]
            assert severities == sorted(severities, reverse=True)

    def test_summarize(self, wide_graph: ComputeGraph) -> None:
        """summarize returns count per pattern kind."""
        detector = PatternDetector(wide_graph)
        summary = detector.summarize()

        assert isinstance(summary, dict)
        assert PatternKind.FAN_OUT in summary

    def test_no_false_diamonds_in_linear(self, linear_graph: ComputeGraph) -> None:
        """Linear graph has no diamond patterns."""
        detector = PatternDetector(linear_graph)
        diamonds = detector.find_diamonds()
        assert len(diamonds) == 0


# ─── Impact Analysis Tests ─────────────────────────────────────────────


class TestImpactAnalyzer:
    """Tests for ImpactAnalyzer."""

    def test_single_input_impact(self, multi_input_graph: ComputeGraph) -> None:
        """Changing x affects a, b, and c."""
        analyzer = ImpactAnalyzer(multi_input_graph)
        report = analyzer.analyze_impact(["x"])

        assert "a" in report.affected_nodes
        assert "b" in report.affected_nodes
        assert "c" in report.affected_nodes
        assert "d" not in report.affected_nodes  # d depends only on z

    def test_isolated_input_impact(self, multi_input_graph: ComputeGraph) -> None:
        """Changing z only affects d."""
        analyzer = ImpactAnalyzer(multi_input_graph)
        report = analyzer.analyze_impact(["z"])

        assert report.affected_nodes == {"d"}
        assert report.affected_count == 1

    def test_blast_radius(self, multi_input_graph: ComputeGraph) -> None:
        """Blast radius is fraction of affected nodes."""
        analyzer = ImpactAnalyzer(multi_input_graph)
        report = analyzer.analyze_impact(["x"])

        # x affects 3 out of 4 compute nodes
        assert report.blast_radius == pytest.approx(3 / 4)

    def test_what_if_query(self, multi_input_graph: ComputeGraph) -> None:
        """what_if returns downstream nodes."""
        analyzer = ImpactAnalyzer(multi_input_graph)
        affected = analyzer.what_if("a")

        assert "c" in affected  # c depends on a

    def test_reverse_impact(self, multi_input_graph: ComputeGraph) -> None:
        """reverse_impact finds inputs that trigger a node."""
        analyzer = ImpactAnalyzer(multi_input_graph)
        triggers = analyzer.reverse_impact("c")

        assert "x" in triggers  # c <- a <- x and c <- b <- x
        assert "y" in triggers  # c <- a <- y

    def test_compare_impacts(self, multi_input_graph: ComputeGraph) -> None:
        """compare_impacts shows overlap and differences."""
        analyzer = ImpactAnalyzer(multi_input_graph)
        comparison = analyzer.compare_impacts(["x"], ["z"])

        assert "d" in comparison["only_b"]  # Only z affects d
        assert "a" in comparison["only_a"]  # Only x affects a

    def test_sensitivity_ranking(self, multi_input_graph: ComputeGraph) -> None:
        """Sensitivity ranking orders inputs by impact."""
        analyzer = ImpactAnalyzer(multi_input_graph)
        ranking = analyzer.sensitivity_ranking()

        # x should be most impactful (affects 3 nodes)
        assert ranking[0][0] == "x"
        assert ranking[0][1] == 3

    def test_propagation_layers(self, linear_graph: ComputeGraph) -> None:
        """Propagation layers show BFS distance from input."""
        analyzer = ImpactAnalyzer(linear_graph)
        layers = analyzer.propagation_layers(["x"])

        assert len(layers) >= 4  # 4 levels of propagation
        assert "a" in layers[0]  # a is at distance 1 from x

    def test_find_shared_dependencies(self, multi_input_graph: ComputeGraph) -> None:
        """Finds inputs that affect all specified nodes."""
        analyzer = ImpactAnalyzer(multi_input_graph)
        shared = analyzer.find_shared_dependencies(["a", "b"])

        assert "x" in shared  # Both a and b depend on x

    def test_isolated_effects(self, multi_input_graph: ComputeGraph) -> None:
        """Identifies leaf affected nodes."""
        analyzer = ImpactAnalyzer(multi_input_graph)
        report = analyzer.analyze_impact(["z"])

        # d has no affected dependents
        assert "d" in report.isolated_effects
