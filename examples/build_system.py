"""Build system example — file dependency tracking (like Make).

Demonstrates DagFlow as a minimal build system where source files are
inputs and build artifacts are compute nodes. When a source file changes,
only the affected artifacts are rebuilt.

This example uses:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode)
- memo.cache (MemoCache)
- propagation.eager (EagerPropagator)
- scheduler.topo (TopologicalScheduler)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode, ComputeNode
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.scheduler.topo import TopologicalScheduler


@dataclass
class BuildArtifact:
    """Represents a build output.

    Attributes:
        name: Artifact identifier (e.g., "app.o", "libfoo.so").
        content: Simulated file content (hash or string).
        build_count: How many times this artifact has been built.
    """

    name: str
    content: str = ""
    build_count: int = 0


class BuildSystem:
    """A minimal build system using DagFlow for dependency tracking.

    Source files are InputNodes (their content hash is the value).
    Build rules are ComputeNodes that produce artifacts from sources.

    Usage:
        build = BuildSystem()
        build.add_source("main.c", "int main() {}")
        build.add_source("util.c", "void util() {}")
        build.add_rule("main.o", ["main.c"], compile_c)
        build.add_rule("util.o", ["util.c"], compile_c)
        build.add_rule("app", ["main.o", "util.o"], link)

        build.build_all()
        build.modify_source("util.c", "void util() { return; }")
        rebuilt = build.build_incremental()
        # Only util.o and app are rebuilt
    """

    def __init__(self) -> None:
        self._graph = ComputeGraph()
        self._cache = MemoCache()
        self._propagator = EagerPropagator(self._graph, self._cache)
        self._artifacts: Dict[str, BuildArtifact] = {}
        self._sources: Dict[str, str] = {}  # source_id -> content

    @property
    def graph(self) -> ComputeGraph:
        """The underlying computation graph."""
        return self._graph

    @property
    def cache(self) -> MemoCache:
        """The underlying memo cache."""
        return self._cache

    def add_source(self, name: str, content: str) -> None:
        """Register a source file.

        The node value is the content hash, so identical content
        won't trigger rebuilds.

        Args:
            name: Source file name (used as node ID).
            content: File content.
        """
        content_hash = self._hash_content(content)
        self._graph.add_input(name, value=content_hash)
        self._sources[name] = content

    def add_rule(
        self,
        target: str,
        dependencies: List[str],
        build_func: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> None:
        """Add a build rule.

        Args:
            target: Name of the artifact to produce.
            dependencies: Source files or other artifacts this depends on.
            build_func: Function that produces artifact content from dep values.
                       If None, uses a default that concatenates inputs.
        """
        artifact = BuildArtifact(name=target)
        self._artifacts[target] = artifact

        if build_func is None:
            # Default: hash of all dependency values (simulates compilation)
            def default_build(deps: Dict[str, Any]) -> str:
                combined = "|".join(str(deps[d]) for d in sorted(deps.keys()))
                return self._hash_content(f"build({target}:{combined})")

            build_func = default_build

        # Wrap to track build count
        original_func = build_func

        def tracked_build(deps: Dict[str, Any]) -> str:
            result = original_func(deps)
            artifact.content = result
            artifact.build_count += 1
            return result

        self._graph.add_compute(target, func=tracked_build, dependencies=dependencies)

    def modify_source(self, name: str, new_content: str) -> None:
        """Modify a source file's content.

        Args:
            name: Source file name.
            new_content: New file content.
        """
        node = self._graph.get_node(name)
        if not isinstance(node, InputNode):
            raise ValueError(f"'{name}' is not a source file")

        new_hash = self._hash_content(new_content)
        self._sources[name] = new_content
        node.set(new_hash)

    def build_all(self) -> Dict[str, str]:
        """Build all artifacts from scratch.

        Returns:
            Dict of {artifact_name: content_hash} for all built artifacts.
        """
        # Propagate from all sources
        all_sources = self._graph.get_input_nodes()
        return self._propagator.propagate(all_sources)

    def build_incremental(self) -> Dict[str, str]:
        """Rebuild only what changed since last build.

        Returns:
            Dict of {artifact_name: content_hash} for rebuilt artifacts only.
        """
        # Find which sources have changed (dirty)
        changed = []
        for source_id in self._graph.get_input_nodes():
            node = self._graph.get_node(source_id)
            if isinstance(node, InputNode):
                changed.append(source_id)

        return self._propagator.propagate(changed)

    def get_artifact(self, name: str) -> Optional[BuildArtifact]:
        """Get a build artifact by name."""
        return self._artifacts.get(name)

    def get_build_order(self) -> List[str]:
        """Get the topological build order for all artifacts.

        Returns:
            List of artifact names in build order.
        """
        scheduler = TopologicalScheduler(self._graph)
        return scheduler.full_schedule()

    @property
    def source_count(self) -> int:
        """Number of registered source files."""
        return len(self._sources)

    @property
    def artifact_count(self) -> int:
        """Number of registered build artifacts."""
        return len(self._artifacts)

    @staticmethod
    def _hash_content(content: str) -> str:
        """Compute a short hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def demo_build_system() -> BuildSystem:
    """Create a demo build system simulating a C project.

    Structure:
        main.c -> main.o ─┐
        util.c -> util.o ─┼─> app
        math.c -> math.o ─┘

    Returns:
        A configured BuildSystem instance.
    """
    build = BuildSystem()

    # Source files
    build.add_source("main.c", '#include "util.h"\nint main() { return util(); }')
    build.add_source("util.c", 'int util() { return 42; }')
    build.add_source("math.c", 'double square(double x) { return x*x; }')

    # Compile rules
    build.add_rule("main.o", ["main.c"])
    build.add_rule("util.o", ["util.c"])
    build.add_rule("math.o", ["math.c"])

    # Link rule
    build.add_rule("app", ["main.o", "util.o", "math.o"])

    return build
