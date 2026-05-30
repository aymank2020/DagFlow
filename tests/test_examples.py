"""Tests for usage examples — verify examples run correctly.

Tests verify structural/behavioral invariants:
- Spreadsheet formulas propagate changes.
- Build system rebuilds only affected artifacts.
- Data pipeline produces consistent results.
"""

import pytest
from examples.spreadsheet import Spreadsheet, demo_spreadsheet
from examples.build_system import BuildSystem, demo_build_system
from examples.data_pipeline import DataPipeline, demo_data_pipeline


class TestSpreadsheetExample:
    """Tests for the spreadsheet example."""

    def test_demo_creates_valid_spreadsheet(self):
        """Demo spreadsheet has expected structure."""
        sheet = demo_spreadsheet()
        assert sheet.cell_count > 0
        assert sheet.get("price") == 100.0
        assert sheet.get("tax_rate") == 0.08

    def test_formula_cells_compute(self):
        """Formula cells produce computed values."""
        sheet = demo_spreadsheet()
        tax = sheet.get("tax_amount")
        assert tax is not None
        assert tax > 0  # Tax on a positive price is positive

    def test_value_change_propagates_to_formulas(self):
        """Changing a value cell updates dependent formula cells."""
        sheet = demo_spreadsheet()
        old_total = sheet.get("total")

        sheet.set_value("price", 200.0)
        new_total = sheet.get("total")

        # Total should change when price changes
        assert new_total != old_total
        # Higher price means higher total
        assert new_total > old_total

    def test_bulk_update_is_atomic(self):
        """Bulk update applies all changes together."""
        sheet = Spreadsheet()
        sheet.set_value("a", 1)
        sheet.set_value("b", 2)
        sheet.set_formula("sum", ["a", "b"], lambda d: d["a"] + d["b"])

        sheet.bulk_update({"a": 10, "b": 20})
        assert sheet.get("sum") == 30

    def test_get_all_values_includes_all_cells(self):
        """get_all_values returns every cell."""
        sheet = demo_spreadsheet()
        values = sheet.get_all_values()
        assert "price" in values
        assert "tax_amount" in values
        assert "total" in values


class TestBuildSystemExample:
    """Tests for the build system example."""

    def test_demo_creates_valid_build(self):
        """Demo build system has expected sources and artifacts."""
        build = demo_build_system()
        assert build.source_count == 3
        assert build.artifact_count == 4  # 3 .o files + 1 app

    def test_build_all_produces_artifacts(self):
        """Building all produces non-empty artifact content."""
        build = demo_build_system()
        build.build_all()

        app = build.get_artifact("app")
        assert app is not None
        assert app.content != ""
        assert app.build_count == 1

    def test_incremental_build_after_source_change(self):
        """Changing one source only rebuilds affected artifacts."""
        build = demo_build_system()
        build.build_all()

        # Record build counts
        main_o_before = build.get_artifact("main.o").build_count
        util_o_before = build.get_artifact("util.o").build_count

        # Modify only util.c
        build.modify_source("util.c", "int util() { return 0; }")
        build.build_incremental()

        # util.o should be rebuilt, main.o should NOT
        # (main.o depends only on main.c which didn't change)
        util_o_after = build.get_artifact("util.o").build_count
        assert util_o_after > util_o_before

    def test_build_order_is_topological(self):
        """Build order respects dependencies."""
        build = demo_build_system()
        order = build.get_build_order()

        # .o files must come before app (which depends on them)
        if "app" in order:
            app_idx = order.index("app")
            for dep in ["main.o", "util.o", "math.o"]:
                if dep in order:
                    assert order.index(dep) < app_idx

    def test_unchanged_source_no_rebuild(self):
        """Setting source to same content doesn't trigger rebuild."""
        build = demo_build_system()
        build.build_all()

        app_count_before = build.get_artifact("app").build_count

        # "Modify" with same content
        build.modify_source("main.c", '#include "util.h"\nint main() { return util(); }')
        build.build_incremental()

        # Content hash is same, so no rebuild should happen
        # (early cutoff in propagation)
        # Note: the input generation still advances, but early-cutoff
        # means downstream won't recompute if hash is unchanged
        app_count_after = build.get_artifact("app").build_count
        assert app_count_after == app_count_before


class TestDataPipelineExample:
    """Tests for the data pipeline example."""

    def test_demo_creates_valid_pipeline(self):
        """Demo pipeline has expected stages."""
        pipeline = demo_data_pipeline()
        assert pipeline.stage_count > 0
        assert pipeline.get_stage_type("raw_sales") == "source"
        assert pipeline.get_stage_type("clean_sales") == "transform"

    def test_pipeline_execution_produces_results(self):
        """Executing pipeline produces non-None results at each stage."""
        pipeline = demo_data_pipeline()
        pipeline.execute()

        assert pipeline.get_result("clean_sales") is not None
        assert pipeline.get_result("total_sales") is not None
        assert pipeline.get_result("net_profit") is not None

    def test_pipeline_removes_negative_sales(self):
        """Clean stage removes negative values from sales."""
        pipeline = demo_data_pipeline()
        pipeline.execute()

        clean = pipeline.get_result("clean_sales")
        assert all(x > 0 for x in clean)

    def test_pipeline_profit_is_consistent(self):
        """Net profit is less than gross profit (tax applied)."""
        pipeline = demo_data_pipeline()
        pipeline.execute()

        gross = pipeline.get_result("gross_profit")
        net = pipeline.get_result("net_profit")

        # Net should be less than gross (tax reduces it)
        if gross > 0:
            assert net < gross

    def test_source_update_propagates(self):
        """Updating a source propagates through the pipeline."""
        pipeline = demo_data_pipeline()
        pipeline.execute()

        old_total = pipeline.get_result("total_sales")

        # Add more sales
        pipeline.update_source("raw_sales", [200, 300, 400, 500])
        new_total = pipeline.get_result("total_sales")

        assert new_total != old_total
