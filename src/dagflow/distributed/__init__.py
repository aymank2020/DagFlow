"""Multi-graph coordination for federated DAG computation.

Provides federation of multiple ComputeGraphs with cross-graph
dependencies, synchronization protocols, and conflict resolution
for concurrent updates to shared nodes.
"""

from dagflow.distributed.federation import (
    FederatedGraph,
    GraphEndpoint,
    CrossGraphEdge,
)
from dagflow.distributed.sync import SyncProtocol, SyncState, SyncResult
from dagflow.distributed.conflict import (
    ConflictResolver,
    ConflictPolicy,
    ConflictRecord,
)

__all__ = [
    "FederatedGraph",
    "GraphEndpoint",
    "CrossGraphEdge",
    "SyncProtocol",
    "SyncState",
    "SyncResult",
    "ConflictResolver",
    "ConflictPolicy",
    "ConflictRecord",
]
