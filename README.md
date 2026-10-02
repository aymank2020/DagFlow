# DagFlow

Incremental computation engine with DAG-based memoization and change propagation.

## Overview

DagFlow models computations as a directed acyclic graph (DAG) where:
- **Input nodes** hold external values that can change.
- **Compute nodes** derive values from their dependencies.
- **Memoization** caches results and skips recomputation when inputs haven't changed.
- **Change propagation** efficiently invalidates and recomputes only affected nodes.

## Architecture

```
dagflow/
├── core/         # Graph structure, node types, dependency tracking
├── scheduler/    # Topological ordering, execution planning
├── propagation/  # Change propagation strategies (eager/lazy)
├── memo/         # Memoization, cache management, staleness detection
└── query/        # Demand-driven recomputation interface
```

## Installation

```bash
pip install -e ".[dev]"
```

## Testing

```bash
pytest
```

## Quick start

```python
from dagflow import ComputeGraph, DemandEngine, MemoCache

graph = ComputeGraph()
source = graph.add_input("x", value=2)
graph.add_compute("double", lambda values: values["x"] * 2, ["x"])
engine = DemandEngine(graph, MemoCache())
assert engine.demand("double") == 4
source.set(3)
assert engine.demand("double") == 6
```

Lazy demand checks upstream generations before reusing cached values. Results
including `None` stay cached, and an evicted entry is recomputed on demand.
Unchanged compute nodes are skipped; checking an output still walks its
ancestors to discover direct input changes.

Run `dagflow --help` after installation or `python -m dagflow.cli --help`
from a source checkout. The command classes accept graphs programmatically;
the standalone CLI currently has no graph-file loading option.

## License

MIT
