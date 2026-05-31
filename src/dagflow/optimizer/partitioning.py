"""Graph partitioning for parallel execution.

Partitions the DAG into execution stages (levels) where all nodes
within a stage are independent and can execute in parallel. Also
supports weighted partitioning that balances computational cost
across stages.

This module depends on:
- core.graph (ComputeGraph)
- core.node (ComputeNode, InputNode)
- optimizer.cost_model (CostModel, NodeCost)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import ComputeNode, InputNode


@dataclass
class ExecutionStage:
    """A group of nodes that can execute in parallel.

    Attributes:
        level: The stage number (0 = first to execute).
        node_ids: Set of node IDs in this stage.
        estimated_cost: Total estimated cost of this stage.
        max_parallelism: Maximum concurrent executions in this stage.
        dependencies: Stages that must complete before this one.
    """

    level: int
    node_ids: Set[str] = field(default_factory=set)
    estimated_cost: float = 0.0
    max_parallelism: int = 0
    dependencies: Set[int] = field(default_factory=set)

    @property
    def width(self) -> int:
        """Number of nodes in this stage (parallelism potential)."""
        return len(self.node_ids)

    @property
    def is_empty(self) -> bool:
        return len(self.node_ids) == 0


@dataclass
class PartitionPlan:
    """Complete execution plan with stages and metadata.

    Attributes:
        stages: Ordered list of execution stages.
        total_levels: Number of stages in the plan.
        critical_path_length: Length of the longest dependency chain.
        max_width: Maximum parallelism across all stages.
        node_to_stage: Mapping of node_id to its stage level.
    """

    stages: List[ExecutionStage] = field(default_factory=list)
    total_levels: int = 0
    critical_path_length: int = 0
    max_width: int = 0
    node_to_stage: Dict[str, int] = field(default_factory=dict)

    @property
    def total_nodes(self) -> int:
        """Total nodes across all stages."""
        return sum(s.width for s in self.stages)


class GraphPartitioner:
    """Partitions a DAG into parallel execution stages.

    Uses topological level assignment: each node is placed at the
    level equal to 1 + max(levels of its dependencies). Nodes at
    the same level have no mutual dependencies and can run in parallel.

    Usage:
        partitioner = GraphPartitioner(graph)
        plan = partitioner.partition()
        for stage in plan.stages:
            # Execute all nodes in stage concurrently
            parallel_execute(stage.node_ids)
    """

    def __init__(
        self,
        graph: ComputeGraph,
        max_stage_width: Optional[int] = None,
    ) -> None:
        """Initialize the partitioner.

        Args:
            graph: The computation graph to partition.
            max_stage_width: Optional limit on nodes per stage. If exceeded,
                            the stage is split into sub-stages.
        """
        self._graph = graph
        self._max_stage_width = max_stage_width

    def partition(self) -> PartitionPlan:
        """Compute the execution plan by assigning topological levels.

        Returns:
            A PartitionPlan with stages ordered for execution.
        """
        levels = self._assign_levels()
        stages = self._build_stages(levels)

        if self._max_stage_width:
            stages = self._split_wide_stages(stages)

        plan = PartitionPlan(
            stages=stages,
            total_levels=len(stages),
            critical_path_length=self._compute_critical_path(),
            max_width=max((s.width for s in stages), default=0),
            node_to_stage={nid: s.level for s in stages for nid in s.node_ids},
        )
        return plan

    def compute_parallelism_profile(self) -> List[int]:
        """Compute the parallelism at each level.

        Returns:
            List where index i = number of nodes executable at level i.
        """
        levels = self._assign_levels()
        if not levels:
            return []

        max_level = max(levels.values()) if levels else 0
        profile = [0] * (max_level + 1)
        for level in levels.values():
            profile[level] += 1
        return profile

    def find_bottlenecks(self, threshold: int = 1) -> List[str]:
        """Find nodes that are sequential bottlenecks.

        A bottleneck is a node that is the sole occupant of its level,
        meaning all other nodes must wait for it.

        Args:
            threshold: Maximum width to consider a level a bottleneck.

        Returns:
            List of node IDs that are bottlenecks.
        """
        levels = self._assign_levels()
        level_groups: Dict[int, List[str]] = defaultdict(list)
        for nid, level in levels.items():
            level_groups[level].append(nid)

        bottlenecks: List[str] = []
        for level, nodes in level_groups.items():
            if len(nodes) <= threshold:
                bottlenecks.extend(nodes)
        return bottlenecks

    def estimate_speedup(self, sequential_cost: Optional[float] = None) -> float:
        """Estimate parallel speedup over sequential execution.

        Uses Amdahl's law approximation: speedup = total_work / critical_path.

        Args:
            sequential_cost: Total sequential cost. If None, assumes unit cost per node.

        Returns:
            Estimated speedup factor (1.0 = no improvement).
        """
        plan = self.partition()
        if plan.critical_path_length == 0:
            return 1.0

        total_work = sequential_cost or float(plan.total_nodes)
        # Critical path determines minimum time with infinite parallelism
        critical_time = float(plan.critical_path_length)

        return total_work / critical_time if critical_time > 0 else 1.0

    def _assign_levels(self) -> Dict[str, int]:
        """Assign topological levels to all compute nodes.

        Level of a node = 1 + max(levels of dependencies).
        Input nodes are at level -1 (not scheduled).
        """
        levels: Dict[str, int] = {}
        compute_nodes = self._graph.get_compute_nodes()

        # Initialize input nodes at level -1
        for nid in self._graph.get_input_nodes():
            levels[nid] = -1

        # Kahn's algorithm variant for level assignment
        in_degree: Dict[str, int] = {}
        for nid in compute_nodes:
            deps = self._graph.get_dependencies(nid)
            in_degree[nid] = len(deps)

        # Start with nodes whose all deps are inputs (in_degree effectively 0)
        queue: deque = deque()
        for nid in compute_nodes:
            deps = self._graph.get_dependencies(nid)
            all_deps_resolved = all(
                d in levels for d in deps
            )
            if all_deps_resolved:
                dep_levels = [levels.get(d, -1) for d in deps]
                levels[nid] = max(dep_levels, default=-1) + 1
                queue.append(nid)

        # Process remaining nodes
        while queue:
            current = queue.popleft()
            current_level = levels[current]

            for dependent in self._graph.get_dependents(current):
                if dependent in levels:
                    continue

                # Check if all dependencies are resolved
                deps = self._graph.get_dependencies(dependent)
                if all(d in levels for d in deps):
                    dep_levels = [levels[d] for d in deps]
                    levels[dependent] = max(dep_levels, default=-1) + 1
                    queue.append(dependent)

        return {nid: lvl for nid, lvl in levels.items() if lvl >= 0}

    def _build_stages(self, levels: Dict[str, int]) -> List[ExecutionStage]:
        """Group nodes by level into ExecutionStages."""
        if not levels:
            return []

        max_level = max(levels.values())
        stages: List[ExecutionStage] = []

        for lvl in range(max_level + 1):
            nodes_at_level = {nid for nid, l in levels.items() if l == lvl}
            if not nodes_at_level:
                continue

            # Determine stage dependencies
            stage_deps: Set[int] = set()
            for nid in nodes_at_level:
                for dep_id in self._graph.get_dependencies(nid):
                    dep_level = levels.get(dep_id)
                    if dep_level is not None and dep_level < lvl:
                        stage_deps.add(dep_level)

            stage = ExecutionStage(
                level=lvl,
                node_ids=nodes_at_level,
                max_parallelism=len(nodes_at_level),
                dependencies=stage_deps,
            )
            stages.append(stage)

        return stages

    def _split_wide_stages(
        self, stages: List[ExecutionStage]
    ) -> List[ExecutionStage]:
        """Split stages that exceed max_stage_width into sub-stages."""
        if not self._max_stage_width:
            return stages

        result: List[ExecutionStage] = []
        for stage in stages:
            if stage.width <= self._max_stage_width:
                result.append(stage)
            else:
                # Split into chunks
                nodes = sorted(stage.node_ids)
                for i in range(0, len(nodes), self._max_stage_width):
                    chunk = set(nodes[i : i + self._max_stage_width])
                    sub_stage = ExecutionStage(
                        level=stage.level,
                        node_ids=chunk,
                        max_parallelism=len(chunk),
                        dependencies=stage.dependencies,
                    )
                    result.append(sub_stage)

        return result

    def _compute_critical_path(self) -> int:
        """Compute the longest path in the DAG (critical path length)."""
        longest: Dict[str, int] = {}

        # Topological order via BFS
        in_degree: Dict[str, int] = {}
        for nid in self._graph.get_compute_nodes():
            deps = self._graph.get_dependencies(nid)
            in_degree[nid] = len([d for d in deps if not self._graph.get_node(d).is_input])

        queue: deque = deque()
        for nid, deg in in_degree.items():
            if deg == 0:
                longest[nid] = 1
                queue.append(nid)

        while queue:
            current = queue.popleft()
            for dep_id in self._graph.get_dependents(current):
                if dep_id not in in_degree:
                    continue
                in_degree[dep_id] -= 1
                candidate = longest.get(current, 0) + 1
                longest[dep_id] = max(longest.get(dep_id, 0), candidate)
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        return max(longest.values(), default=0)
