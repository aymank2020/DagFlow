"""Example: Optimizer in action — fusion, pruning, and partitioning.

Demonstrates using the optimizer module to analyze and improve
a computation graph's performance:
- Complexity analysis to understand graph structure
- Pattern detection to find optimization opportunities
- Node fusion to reduce scheduling overhead
- Dead node pruning to eliminate waste
- Partitioning to identify parallelism
- Cost modeling to prioritize optimizations

This example builds a realistic data processing graph and shows
how each optimization pass improves it.
"""

from __future__ import annotations

from typing import Any, Dict

from dagflow.core.graph import ComputeGraph
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.analysis.complexity import ComplexityAnalyzer
from dagflow.analysis.patterns import PatternDetector, PatternKind
from dagflow.analysis.impact import ImpactAnalyzer
from dagflow.optimizer.fusion import NodeFuser
from dagflow.optimizer.pruning import DeadNodePruner
from dagflow.optimizer.partitioning import GraphPartitioner
from dagflow.optimizer.cost_model import CostModel


def build_data_processing_graph() -> ComputeGraph:
    """Build a realistic data processing graph with optimization opportunities.

    The graph simulates a data pipeline with:
    - Multiple input sources
    - Linear processing chains (fusible)
    - Dead branches (prunable)
    - Parallel stages (partitionable)
    - Diamond dependencies
    """
    graph = ComputeGraph()

    # Input sources
    graph.add_input("raw_sales", value=1000)
    graph.add_input("raw_returns", value=50)
    graph.add_input("tax_rate", value=0.08)
    graph.add_input("discount_rate", value=0.1)
    graph.add_input("unused_config", value="legacy")  # Dead input

    # Linear chain 1: Sales processing (fusible)
    graph.add_compute(
        "clean_sales",
        lambda d: max(0, d["raw_sales"]),
        ["raw_sales"],
    )
    graph.add_compute(
        "adjusted_sales",
        lambda d: d["clean_sales"] * 1.02,  # Inflation adjustment
        ["clean_sales"],
    )
    graph.add_compute(
        "net_sales",
        lambda d: d["adjusted_sales"] - d["raw_returns"],
        ["adjusted_sales", "raw_returns"],
    )

    # Linear chain 2: Tax computation (fusible)
    graph.add_compute(
        "tax_base",
        lambda d: d["net_sales"] * 0.95,  # Deductions
        ["net_sales"],
    )
    graph.add_compute(
        "tax_amount",
        lambda d: d["tax_base"] * d["tax_rate"],
        ["tax_base", "tax_rate"],
    )

    # Parallel branch: Discount computation
    graph.add_compute(
        "discount_amount",
        lambda d: d["net_sales"] * d["discount_rate"],
        ["net_sales", "discount_rate"],
    )

    # Diamond convergence: Final revenue
    graph.add_compute(
        "final_revenue",
        lambda d: d["net_sales"] - d["tax_amount"] - d["discount_amount"],
        ["net_sales", "tax_amount", "discount_amount"],
    )

    # Dead branch: unused computation
    graph.add_compute(
        "legacy_metric",
        lambda d: len(str(d["unused_config"])) * 42,
        ["unused_config"],
    )
    graph.add_compute(
        "dead_aggregate",
        lambda d: d["legacy_metric"] + 100,
        ["legacy_metric"],
    )

    # Output aggregation
    graph.add_compute(
        "revenue_summary",
        lambda d: {
            "gross": d["net_sales"],
            "tax": d["tax_amount"],
            "discount": d["discount_amount"],
            "net": d["final_revenue"],
        },
        ["net_sales", "tax_amount", "discount_amount", "final_revenue"],
    )

    return graph


def analyze_complexity(graph: ComputeGraph) -> None:
    """Run complexity analysis and print results."""
    analyzer = ComplexityAnalyzer(graph)
    metrics = analyzer.analyze()

    print(f"  Total nodes: {metrics.total_nodes}")
    print(f"  Inputs: {metrics.input_count}, Compute: {metrics.compute_count}")
    print(f"  Depth: {metrics.depth}, Width: {metrics.width}")
    print(f"  Max fan-in: {metrics.max_fan_in}, Max fan-out: {metrics.max_fan_out}")
    print(f"  Density: {metrics.density:.3f}")
    print(f"  Parallelism ratio: {metrics.parallelism_ratio:.2f}")
    print(f"  Is linear: {metrics.is_linear}")


def detect_patterns(graph: ComputeGraph) -> None:
    """Run pattern detection and print findings."""
    detector = PatternDetector(graph)
    patterns = detector.detect_all()

    summary = detector.summarize()
    print(f"  Patterns found: {len(patterns)}")
    for kind, count in summary.items():
        print(f"    {kind.name}: {count}")

    # Show top patterns
    for pattern in patterns[:3]:
        print(f"  [{pattern.kind.name}] {pattern.description} (severity: {pattern.severity:.2f})")


def demonstrate_fusion(graph: ComputeGraph) -> None:
    """Show node fusion optimization."""
    fuser = NodeFuser(graph, min_chain_length=2)
    candidates = fuser.find_candidates()

    print(f"  Fusible chains found: {len(candidates)}")
    for candidate in candidates:
        print(f"    Chain: {' -> '.join(candidate.chain)} (length {candidate.length})")

    # Dry run to see impact
    result = fuser.apply(dry_run=True)
    print(f"  Would fuse {result.fused_count} chains, removing {result.nodes_removed} nodes")


def demonstrate_pruning(graph: ComputeGraph) -> None:
    """Show dead node pruning."""
    # Define outputs (what we actually need)
    outputs = {"revenue_summary", "final_revenue"}
    pruner = DeadNodePruner(graph, outputs=outputs)

    dead = pruner.find_dead_nodes()
    suggestions = pruner.suggest_removals()

    print(f"  Dead nodes found: {len(dead)}")
    for node_id, reason in suggestions.items():
        print(f"    {node_id}: {reason}")

    # Dry run
    result = pruner.prune(dry_run=True)
    print(f"  Would remove {result.removed_count} nodes")
    print(f"  Preserved: {len(result.preserved_nodes)} nodes")


def demonstrate_partitioning(graph: ComputeGraph) -> None:
    """Show graph partitioning for parallelism."""
    partitioner = GraphPartitioner(graph)
    plan = partitioner.partition()

    print(f"  Execution stages: {plan.total_levels}")
    print(f"  Critical path: {plan.critical_path_length}")
    print(f"  Max parallelism: {plan.max_width}")

    for stage in plan.stages:
        nodes = sorted(stage.node_ids)
        print(f"    Stage {stage.level}: [{', '.join(nodes)}] (width={stage.width})")

    speedup = partitioner.estimate_speedup()
    print(f"  Estimated speedup: {speedup:.2f}x")


def demonstrate_cost_model(graph: ComputeGraph) -> None:
    """Show cost estimation and ranking."""
    model = CostModel(graph)
    profile = model.profile_graph()

    print(f"  Total estimated compute time: {profile.total_compute_time:.4f}s")
    print(f"  Bottleneck nodes: {profile.bottleneck_nodes}")

    # Top 5 most expensive nodes
    ranking = model.rank_nodes_by_cost()[:5]
    print("  Top 5 expensive nodes:")
    for node_id, cost in ranking:
        print(f"    {node_id}: cost={cost:.4f}")


def demonstrate_impact(graph: ComputeGraph) -> None:
    """Show impact analysis."""
    analyzer = ImpactAnalyzer(graph)

    # What happens if raw_sales changes?
    report = analyzer.analyze_impact(["raw_sales"])
    print(f"  Changing 'raw_sales' affects {report.affected_count} nodes")
    print(f"  Blast radius: {report.blast_radius:.1%}")
    print(f"  Propagation depth: {report.propagation_depth}")

    # Sensitivity ranking
    ranking = analyzer.sensitivity_ranking()
    print("  Input sensitivity ranking:")
    for input_id, count in ranking:
        print(f"    {input_id}: affects {count} nodes")


def main() -> None:
    """Run the optimization example."""
    graph = build_data_processing_graph()

    print("=" * 60)
    print("DagFlow Optimizer Example: Data Processing Pipeline")
    print("=" * 60)

    print("\n1. COMPLEXITY ANALYSIS")
    print("-" * 40)
    analyze_complexity(graph)

    print("\n2. PATTERN DETECTION")
    print("-" * 40)
    detect_patterns(graph)

    print("\n3. NODE FUSION")
    print("-" * 40)
    demonstrate_fusion(graph)

    print("\n4. DEAD NODE PRUNING")
    print("-" * 40)
    demonstrate_pruning(graph)

    print("\n5. GRAPH PARTITIONING")
    print("-" * 40)
    demonstrate_partitioning(graph)

    print("\n6. COST MODEL")
    print("-" * 40)
    demonstrate_cost_model(graph)

    print("\n7. IMPACT ANALYSIS")
    print("-" * 40)
    demonstrate_impact(graph)

    # Run actual computation
    print("\n8. EXECUTION")
    print("-" * 40)
    cache = MemoCache()
    propagator = EagerPropagator(graph, cache)
    results = propagator.propagate(["raw_sales", "raw_returns", "tax_rate", "discount_rate"])
    print(f"  Computed {len(results)} nodes")
    if "revenue_summary" in results:
        print(f"  Revenue summary: {results['revenue_summary']}")


if __name__ == "__main__":
    main()
