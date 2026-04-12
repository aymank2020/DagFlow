# DagFlow

Incremental computation engine with DAG-based memoization and change propagation.

## Overview

DagFlow models computations as a directed acyclic graph (DAG) where:
- **Input nodes** hold external values that can change.
- **Compute nodes** derive values from their dependencies.
- **Memoization** caches results and skips recomputation when inputs haven't changed.
- **Change propagation** efficiently invalidates and recomputes only affected nodes.

## Installation

```bash
pip install -e ".[dev]"
```

## Testing

```bash
pytest
```
