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

## License

MIT

## Architecture

    dagflow/
      core/         - Graph structure, node types, dependency tracking
      scheduler/    - Topological ordering, execution planning
      propagation/  - Change propagation strategies (eager/lazy)
      memo/         - Memoization, cache management, staleness detection
      query/        - Demand-driven recomputation interface

## License

MIT
