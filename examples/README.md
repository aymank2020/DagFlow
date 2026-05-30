# DagFlow Examples

These examples demonstrate real-world applications of the DagFlow incremental computation engine.

## Spreadsheet (`spreadsheet.py`)

A simple spreadsheet engine where cells contain either literal values or formulas. Changes to value cells automatically propagate through formula cells.

Key concepts: InputNode as value cells, ComputeNode as formula cells, EagerPropagator for automatic updates.

## Build System (`build_system.py`)

A minimal build system (like Make) that tracks file dependencies. When a source file changes, only affected build artifacts are rebuilt.

Key concepts: Content hashing for change detection, topological scheduling for build order, early-cutoff to avoid unnecessary rebuilds.

## Data Pipeline (`data_pipeline.py`)

An ETL-style data pipeline where raw data flows through transformation, joining, and aggregation stages. When source data changes, only affected downstream stages are re-executed.

Key concepts: Multi-stage processing, join operations, incremental recomputation.

## Running Examples

```python
from examples.spreadsheet import demo_spreadsheet
from examples.build_system import demo_build_system
from examples.data_pipeline import demo_data_pipeline

# Spreadsheet
sheet = demo_spreadsheet()
print(sheet.get("total"))  # Price + tax * quantity

# Build system
build = demo_build_system()
build.build_all()
build.modify_source("util.c", "int util() { return 0; }")
rebuilt = build.build_incremental()  # Only util.o and app rebuilt

# Data pipeline
pipeline = demo_data_pipeline()
pipeline.execute()
print(pipeline.get_result("net_profit"))
```
