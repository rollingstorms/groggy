"""Typed, stable joins share the native engine and retain frozen source contexts."""
import pytest
from groggy import BaseTable, Graph


def table(**columns):
    return BaseTable.from_dict(columns)


def test_composite_join_uses_every_component_and_dictionary_key_pairs():
    left = table(x=[1, 1, 2], y=["a", "b", "b"], value=[10, 11, 12])
    right = table(u=[1, 2], v=["b", "a"], extra=[20, 21])
    result = left.join(right, on={"left": ["x", "y"], "right": ["u", "v"]})
    assert result["value"] == 11
    assert result["extra"] == 20
    same = table(x=[1, 2], y=["b", "a"])
    assert left.join(same, on=["x", "y"])["value"] == 11


def test_outer_order_coalesced_keys_and_duplicate_matches_are_stable():
    left = table(k=[2, 1, 2, None], a=[0, 1, 2, 3])
    right = table(k=[2, 3, 2, None], b=[10, 11, 12, 13])
    for _ in range(5):
        result = left.join(right, on="k", how="outer")
        assert result["a"].to_list() == [0, 0, 1, 2, 2, 3, None]
        assert result["b"].to_list() == [10, 12, None, 10, 12, 13, 11]
        assert result["k"].to_list() == [2, 2, 1, 2, 2, None, 3]


@pytest.mark.parametrize("how", ["inner", "left", "outer", "cross"])
def test_join_column_name_collisions_reject_without_losing_input_data(how):
    left = table(k=[1], v=[2], left_v=[3], v_right=[4])
    right = table(k=[1], v=[5])
    with pytest.raises(RuntimeError, match="collides"):
        left.join(right, on="k", how=how)
    assert left["v"] == 2


def test_empty_duplicate_and_missing_key_lists_fail_explicitly():
    left = table(x=[1], y=[2])
    for keys in [[], ["x", "x"], ["missing"]]:
        with pytest.raises(RuntimeError, match="Join failed"):
            left.join(left, on=keys)
    with pytest.raises(ValueError, match="Invalid join type"):
        left.join(left, on="x", how="unknown")


def test_composite_join_retains_original_sources_and_rejects_automatic_writeback():
    graph = Graph()
    graph.add_nodes([{"x": 1, "y": "a"}, {"x": 1, "y": "b"}])
    source = graph.read_view().nodes_table().base_table()
    result = source.join(source, on=["x", "y"])
    assert len(result.source_views()) == 1
    assert result.source_entity_ids is None
    graph.set_node_attr(0, "x", 99)
    assert result.source_views()[0].node_attr(0, "x") == 1
    with pytest.raises(ValueError, match="explicit identity"):
        result.update_source_graph(graph, ["x"])
