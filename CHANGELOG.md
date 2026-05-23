# Changelog

## 0.2.0 (2026-05-23)

### Added
- Demand-driven lazy evaluation engine
- Early-cutoff optimization in eager propagation
- Selective invalidation with generation awareness
- Priority-based tie-breaking in scheduler
- Benchmark script, CI workflow, CONTRIBUTING.md

### Changed
- Refactored ComputeGraph with reverse adjacency tracking
- Consolidated test files

## 0.1.0 (2026-03-05)

### Added
- Initial project structure
- InputNode and ComputeNode types
- ComputeGraph with BFS cycle detection
- TopologicalScheduler
- MemoCache with generation tracking
- Invalidator and EagerPropagator
