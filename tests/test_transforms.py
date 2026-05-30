"""Tests for transforms — map/reduce, window, aggregate, conditional.

Tests verify behavioral invariants:
- Map transforms apply correctly to elements and collections.
- Filter preserves only matching elements.
- Reduce combines elements correctly.
- Window maintains correct buffer size.
- Aggregates produce correct results for multiple inputs.
- Conditionals select the correct branch.
"""

import pytest
from dagflow.core.graph import ComputeGraph
from dagflow.core.node import InputNode
from dagflow.memo.cache import MemoCache
from dagflow.propagation.eager import EagerPropagator
from dagflow.transforms.map_reduce import MapNode, FilterNode, ReduceNode
from dagflow.transforms.window import WindowNode, WindowBuffer
from dagflow.transforms.aggregate import AggregateNode, AggregateOp
from dagflow.transforms.conditional import ConditionalNode, SwitchNode


class TestMapNode:
    """Tests for map transforms."""

    def test_map_element_wise_on_list(self):
        """Map applies transform to each element of a list."""
        g = ComputeGraph()
        g.add_input("nums", value=[1, 2, 3, 4])
        cache = MemoCache()

        mapper = MapNode(g)
        mapper.create("doubled", source="nums", transform=lambda x: x * 2)

        EagerPropagator(g, cache).propagate(["nums"])
        assert cache.get("doubled") == [2, 4, 6, 8]

    def test_map_whole_value(self):
        """Map with element_wise=False applies to entire value."""
        g = ComputeGraph()
        g.add_input("text", value="hello")
        cache = MemoCache()

        mapper = MapNode(g)
        mapper.create("upper", source="text", transform=str.upper, element_wise=False)

        EagerPropagator(g, cache).propagate(["text"])
        assert cache.get("upper") == "HELLO"

    def test_map_chain(self):
        """Chained maps compose correctly."""
        g = ComputeGraph()
        g.add_input("x", value=2)
        cache = MemoCache()

        mapper = MapNode(g)
        mapper.create_chain(
            ["step1", "step2"],
            source="x",
            transforms=[lambda v: v + 10, lambda v: v * 3],
        )

        EagerPropagator(g, cache).propagate(["x"])
        # step1 = 2 + 10 = 12, step2 = 12 * 3 = 36
        assert cache.get("step2") == 36

    def test_map_propagates_on_input_change(self):
        """Map recomputes when source changes."""
        g = ComputeGraph()
        inp = g.add_input("x", value=5)
        cache = MemoCache()

        mapper = MapNode(g)
        mapper.create("neg", source="x", transform=lambda v: -v, element_wise=False)

        prop = EagerPropagator(g, cache)
        prop.propagate(["x"])
        assert cache.get("neg") == -5

        inp.set(10)
        prop.propagate(["x"])
        assert cache.get("neg") == -10


class TestFilterNode:
    """Tests for filter transforms."""

    def test_filter_keeps_matching_elements(self):
        """Filter retains only elements satisfying the predicate."""
        g = ComputeGraph()
        g.add_input("nums", value=[1, 2, 3, 4, 5, 6])
        cache = MemoCache()

        filterer = FilterNode(g)
        filterer.create("evens", source="nums", predicate=lambda x: x % 2 == 0)

        EagerPropagator(g, cache).propagate(["nums"])
        result = cache.get("evens")
        assert all(x % 2 == 0 for x in result)
        assert len(result) == 3

    def test_filter_empty_result(self):
        """Filter can produce empty collection."""
        g = ComputeGraph()
        g.add_input("nums", value=[1, 3, 5])
        cache = MemoCache()

        filterer = FilterNode(g)
        filterer.create("evens", source="nums", predicate=lambda x: x % 2 == 0)

        EagerPropagator(g, cache).propagate(["nums"])
        assert cache.get("evens") == []


class TestReduceNode:
    """Tests for reduce transforms."""

    def test_reduce_sums_list(self):
        """Reduce with addition produces sum."""
        g = ComputeGraph()
        g.add_input("nums", value=[10, 20, 30])
        cache = MemoCache()

        reducer = ReduceNode(g)
        reducer.create("total", source="nums", reducer=lambda a, b: a + b, initial=0)

        EagerPropagator(g, cache).propagate(["nums"])
        assert cache.get("total") == 60

    def test_reduce_multi_source(self):
        """Multi-source reduce combines values from multiple nodes."""
        g = ComputeGraph()
        g.add_input("a", value=10)
        g.add_input("b", value=20)
        g.add_input("c", value=30)
        cache = MemoCache()

        reducer = ReduceNode(g)
        reducer.create_multi_source(
            "sum_all",
            sources=["a", "b", "c"],
            reducer=lambda acc, val: acc + val,
            initial=0,
        )

        EagerPropagator(g, cache).propagate(["a", "b", "c"])
        assert cache.get("sum_all") == 60


class TestWindowBuffer:
    """Tests for the WindowBuffer data structure."""

    def test_buffer_respects_max_size(self):
        """Buffer never exceeds max_size."""
        buf = WindowBuffer(3)
        for i in range(10):
            buf.push(i)
        assert buf.current_size == 3

    def test_buffer_evicts_oldest(self):
        """Oldest values are evicted when buffer is full."""
        buf = WindowBuffer(3)
        buf.push(1)
        buf.push(2)
        buf.push(3)
        evicted = buf.push(4)

        assert evicted == 1
        assert buf.values == [2, 3, 4]

    def test_buffer_invalid_size_raises(self):
        """Window size < 1 raises ValueError."""
        with pytest.raises(ValueError):
            WindowBuffer(0)


class TestWindowNode:
    """Tests for sliding window computations."""

    def test_moving_average(self):
        """Moving average computes correctly over window."""
        g = ComputeGraph()
        inp = g.add_input("temp", value=10)
        cache = MemoCache()

        window = WindowNode(g, window_size=3)
        window.create_moving_average("avg_temp", source="temp")

        prop = EagerPropagator(g, cache)
        prop.propagate(["temp"])  # Window: [10], avg = 10

        inp.set(20)
        prop.propagate(["temp"])  # Window: [10, 20], avg = 15

        inp.set(30)
        prop.propagate(["temp"])  # Window: [10, 20, 30], avg = 20

        assert cache.get("avg_temp") == 20.0

    def test_custom_window_function(self):
        """Custom window function receives buffer contents."""
        g = ComputeGraph()
        inp = g.add_input("x", value=1)
        cache = MemoCache()

        window = WindowNode(g, window_size=5)
        window.create_custom(
            "range",
            source="x",
            compute=lambda vals: max(vals) - min(vals),
        )

        prop = EagerPropagator(g, cache)
        prop.propagate(["x"])

        inp.set(5)
        prop.propagate(["x"])

        inp.set(3)
        prop.propagate(["x"])

        # Window: [1, 5, 3], range = 5 - 1 = 4
        assert cache.get("range") == 4


class TestAggregateNode:
    """Tests for multi-input aggregation."""

    def test_sum_aggregation(self):
        """SUM aggregation adds all source values."""
        g = ComputeGraph()
        g.add_input("a", value=10)
        g.add_input("b", value=20)
        g.add_input("c", value=30)
        cache = MemoCache()

        agg = AggregateNode(g)
        agg.create("total", sources=["a", "b", "c"], op=AggregateOp.SUM)

        EagerPropagator(g, cache).propagate(["a", "b", "c"])
        assert cache.get("total") == 60

    def test_average_aggregation(self):
        """AVERAGE aggregation computes mean of source values."""
        g = ComputeGraph()
        g.add_input("s1", value=10)
        g.add_input("s2", value=20)
        cache = MemoCache()

        agg = AggregateNode(g)
        agg.create("avg", sources=["s1", "s2"], op=AggregateOp.AVERAGE)

        EagerPropagator(g, cache).propagate(["s1", "s2"])
        assert cache.get("avg") == 15.0

    def test_min_max_aggregation(self):
        """MIN and MAX find extremes."""
        g = ComputeGraph()
        g.add_input("a", value=5)
        g.add_input("b", value=15)
        g.add_input("c", value=10)
        cache = MemoCache()

        agg = AggregateNode(g)
        agg.create("lo", sources=["a", "b", "c"], op=AggregateOp.MIN)
        agg.create("hi", sources=["a", "b", "c"], op=AggregateOp.MAX)

        EagerPropagator(g, cache).propagate(["a", "b", "c"])
        assert cache.get("lo") == 5
        assert cache.get("hi") == 15

    def test_weighted_average(self):
        """Weighted average applies weights correctly."""
        g = ComputeGraph()
        g.add_input("exam", value=90)
        g.add_input("homework", value=80)
        cache = MemoCache()

        agg = AggregateNode(g)
        agg.create_weighted(
            "grade",
            sources=["exam", "homework"],
            weights=[0.7, 0.3],
        )

        EagerPropagator(g, cache).propagate(["exam", "homework"])
        # (90*0.7 + 80*0.3) / (0.7+0.3) = 63+24 = 87
        assert cache.get("grade") == 87.0


class TestConditionalNode:
    """Tests for conditional computation."""

    def test_if_then_else_selects_branch(self):
        """Conditional selects then-branch when condition is truthy."""
        g = ComputeGraph()
        g.add_input("flag", value=True)
        g.add_input("yes_val", value="selected")
        g.add_input("no_val", value="rejected")
        cache = MemoCache()

        cond = ConditionalNode(g)
        cond.create("result", condition="flag", then_source="yes_val", else_source="no_val")

        EagerPropagator(g, cache).propagate(["flag", "yes_val", "no_val"])
        assert cache.get("result") == "selected"

    def test_if_then_else_switches_on_change(self):
        """Conditional switches branch when condition changes."""
        g = ComputeGraph()
        flag = g.add_input("flag", value=True)
        g.add_input("a", value=10)
        g.add_input("b", value=20)
        cache = MemoCache()

        cond = ConditionalNode(g)
        cond.create("out", condition="flag", then_source="a", else_source="b")

        prop = EagerPropagator(g, cache)
        prop.propagate(["flag", "a", "b"])
        assert cache.get("out") == 10

        flag.set(False)
        prop.propagate(["flag"])
        assert cache.get("out") == 20

    def test_threshold_conditional(self):
        """Threshold node produces correct output above/below threshold."""
        g = ComputeGraph()
        g.add_input("temp", value=105)
        cache = MemoCache()

        cond = ConditionalNode(g)
        cond.create_threshold(
            "alarm", source="temp", threshold=100,
            above_value="OVERHEAT", below_value="OK",
        )

        EagerPropagator(g, cache).propagate(["temp"])
        assert cache.get("alarm") == "OVERHEAT"


class TestSwitchNode:
    """Tests for multi-way switch."""

    def test_switch_selects_correct_case(self):
        """Switch selects the source matching the selector value."""
        g = ComputeGraph()
        g.add_input("mode", value="fast")
        g.add_input("fast_val", value=100)
        g.add_input("slow_val", value=10)
        cache = MemoCache()

        switch = SwitchNode(g)
        switch.create(
            "speed",
            selector="mode",
            cases={"fast": "fast_val", "slow": "slow_val"},
        )

        EagerPropagator(g, cache).propagate(["mode", "fast_val", "slow_val"])
        assert cache.get("speed") == 100

    def test_switch_uses_default_when_no_match(self):
        """Switch falls back to default when no case matches."""
        g = ComputeGraph()
        g.add_input("mode", value="unknown")
        g.add_input("a", value=1)
        g.add_input("fallback", value=-1)
        cache = MemoCache()

        switch = SwitchNode(g)
        switch.create(
            "out",
            selector="mode",
            cases={"known": "a"},
            default="fallback",
        )

        EagerPropagator(g, cache).propagate(["mode", "a", "fallback"])
        assert cache.get("out") == -1
