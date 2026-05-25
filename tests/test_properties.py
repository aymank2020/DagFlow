"""Property-based tests for graph invariants."""
from hypothesis import given, strategies as st
from dagflow.core.graph import ComputeGraph

@given(st.lists(st.text(min_size=1, max_size=5, alphabet="abcdefgh"), min_size=2, max_size=8, unique=True))
def test_input_count_matches(names):
    g = ComputeGraph()
    for name in names:
        g.add_input(name)
    assert g.node_count == len(names)
    assert len(g.get_input_nodes()) == len(names)
