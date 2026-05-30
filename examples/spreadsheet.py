"""Spreadsheet example — simple cell-based computation.

Demonstrates DagFlow as a spreadsheet engine where cells can contain
either literal values (InputNodes) or formulas (ComputeNodes) that
reference other cells. Changes propagate automatically.

This example uses:
- core.graph (ComputeGraph)
- core.node (InputNode, ComputeNode)
- memo.cache (MemoCache)
- propagation.eager (EagerPropagator)
- batch.transaction (Transaction)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode, ComputeNode
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.batch.transaction import Transaction


class Spreadsheet:
    """A simple spreadsheet backed by DagFlow's incremental computation.

    Cells are addressed by string names (e.g., "A1", "B2"). A cell can
    hold a literal value or a formula that references other cells.

    Usage:
        sheet = Spreadsheet()
        sheet.set_value("A1", 10)
        sheet.set_value("A2", 20)
        sheet.set_formula("A3", ["A1", "A2"], lambda d: d["A1"] + d["A2"])
        assert sheet.get("A3") == 30

        sheet.set_value("A1", 50)
        assert sheet.get("A3") == 70  # Automatically recomputed
    """

    def __init__(self) -> None:
        self._graph = ComputeGraph()
        self._cache = MemoCache()
        self._propagator = EagerPropagator(self._graph, self._cache)
        self._formula_cells: Dict[str, Callable] = {}

    @property
    def graph(self) -> ComputeGraph:
        """Access the underlying computation graph."""
        return self._graph

    @property
    def cache(self) -> MemoCache:
        """Access the underlying memo cache."""
        return self._cache

    def set_value(self, cell: str, value: Any) -> None:
        """Set a cell to a literal value.

        If the cell doesn't exist, creates it as an InputNode.
        If it already exists as an InputNode, updates its value.

        Args:
            cell: Cell address (e.g., "A1").
            value: The literal value to store.
        """
        try:
            node = self._graph.get_node(cell)
            if isinstance(node, InputNode):
                node.set(value)
                self._propagator.propagate([cell])
            else:
                raise ValueError(
                    f"Cell '{cell}' is a formula cell — use set_formula() to change it"
                )
        except KeyError:
            self._graph.add_input(cell, value=value)

    def set_formula(
        self,
        cell: str,
        references: List[str],
        formula: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """Set a cell to a formula that references other cells.

        Args:
            cell: Cell address for the formula cell.
            references: List of cell addresses this formula reads from.
            formula: Function that receives {cell: value} and returns result.
        """
        if cell in self._formula_cells:
            raise ValueError(f"Cell '{cell}' already has a formula")

        self._graph.add_compute(cell, func=formula, dependencies=references)
        self._formula_cells[cell] = formula

        # Initial computation — use DemandEngine to compute this node
        # immediately, which handles transitive dependencies correctly.
        from dagflow.query.demand import DemandEngine
        engine = DemandEngine(self._graph, self._cache)
        engine.demand(cell)

    def get(self, cell: str) -> Any:
        """Get the current value of a cell.

        Args:
            cell: Cell address.

        Returns:
            The current value (literal or computed).
        """
        node = self._graph.get_node(cell)
        if isinstance(node, InputNode):
            return node.value
        elif isinstance(node, ComputeNode):
            return self._cache.get(cell)
        return None

    def bulk_update(self, changes: Dict[str, Any]) -> None:
        """Update multiple value cells atomically.

        All changes are applied and propagated as a single batch.

        Args:
            changes: Dict of {cell: new_value} for input cells.
        """
        with Transaction(self._graph, self._cache) as txn:
            for cell, value in changes.items():
                txn.set_input(cell, value)

    def get_all_values(self) -> Dict[str, Any]:
        """Get current values of all cells.

        Returns:
            Dict of {cell: value} for every cell in the spreadsheet.
        """
        values: Dict[str, Any] = {}
        for node_id in self._graph.all_node_ids:
            values[node_id] = self.get(node_id)
        return values

    @property
    def cell_count(self) -> int:
        """Total number of cells."""
        return self._graph.node_count


def demo_spreadsheet() -> Spreadsheet:
    """Create a demo spreadsheet with sample data.

    Layout:
        A1=100 (price), A2=0.08 (tax_rate)
        B1 = A1 * A2 (tax_amount)
        B2 = A1 + B1 (total)

    Returns:
        A configured Spreadsheet instance.
    """
    sheet = Spreadsheet()

    # Input cells
    sheet.set_value("price", 100.0)
    sheet.set_value("tax_rate", 0.08)
    sheet.set_value("quantity", 5)

    # Formula cells
    sheet.set_formula(
        "tax_amount",
        ["price", "tax_rate"],
        lambda d: d["price"] * d["tax_rate"],
    )
    sheet.set_formula(
        "subtotal",
        ["price", "quantity"],
        lambda d: d["price"] * d["quantity"],
    )
    sheet.set_formula(
        "total",
        ["subtotal", "tax_amount", "quantity"],
        lambda d: d["subtotal"] + d["tax_amount"] * d["quantity"],
    )

    return sheet
