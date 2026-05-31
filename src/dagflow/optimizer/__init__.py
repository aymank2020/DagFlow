"""Graph optimization passes for the DAG computation engine.

Provides optimization strategies that transform the graph structure
to improve execution performance without changing semantics:
- Node fusion: merge sequential compute nodes
- Dead node pruning: remove unreachable subgraphs
- Partitioning: split graph into parallel execution stages
- Cost modeling: estimate resource usage for optimization decisions
"""

from dagflow.optimizer.fusion import NodeFuser, FusionResult
from dagflow.optimizer.pruning import DeadNodePruner, PruneResult
from dagflow.optimizer.partitioning import GraphPartitioner, ExecutionStage
from dagflow.optimizer.cost_model import CostModel, NodeCost

__all__ = [
    "NodeFuser",
    "FusionResult",
    "DeadNodePruner",
    "PruneResult",
    "GraphPartitioner",
    "ExecutionStage",
    "CostModel",
    "NodeCost",
]
