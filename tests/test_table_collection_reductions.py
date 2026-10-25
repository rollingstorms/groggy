"""Both table collection adapters delegate to checked native reductions."""
import pytest
from groggy import BaseTable, Graph


def column(table, name):
    value = table[name]
    return value.to_list() if hasattr(value, "to_list") else [value]


def fixture():
    return BaseTable.from_dict({"kind": ["b", "a", "b"], "value": [1, 4, 2], "flag": [False, True, True], "text": ["x", "y", "z"]})


@pytest.mark.parametrize("method", ["group_by", "groupby"])
def test_both_group_wrappers_share_numeric_counts_and_keys(method):
    groups = getattr(fixture(), method)("kind")
    assert column(groups.sum(), "kind") == ["b", "a"]
    assert column(groups.sum(), "value") == [3, 4]
    assert column(groups.mean(), "value") == [1.5, 4.0]
    assert column(groups.min(), "value") == [1, 4]
    assert column(groups.max(), "value") == [2, 4]
    assert column(groups.count(), "text") == [2, 1]
    assert column(groups.all(), "flag") == [False, True]
    assert column(groups.agg({"text": "nunique"}), "text") == [2, 1]


def test_native_ungrouped_collection_uses_all_observations_for_means():
    core = type(fixture().groupby("kind"))
    groups = core([BaseTable.from_dict({"x": [1]}), BaseTable.from_dict({"x": [3, 5, 7]})])
    assert column(groups.sum(), "x") == [16]
    assert column(groups.mean(), "x") == [4.0]
    assert column(groups.count(), "x") == [4]


def test_legacy_constructor_preserves_per_table_semantics_and_projection_is_not_a_group_key():
    legacy = type(fixture().group_by("kind"))
    groups = legacy([BaseTable.from_dict({"x": [1, 3]}), BaseTable.from_dict({"x": [5]})], columns=["x"])
    assert column(groups.mean(), "x") == [2.0, 5.0]
    assert column(groups.sum(), "x") == [4, 5]


@pytest.mark.parametrize("method", ["group_by", "groupby"])
def test_explicit_bad_specs_collisions_and_false_before_invalid_boolean_reject(method):
    groups = getattr(fixture(), method)("kind")
    for specs in [{"missing": "sum"}, {"kind": "count"}, {"value": "bad"}]:
        with pytest.raises(ValueError):
            groups.agg(specs)
    bad = BaseTable.from_dict({"kind": [1, 1], "flag": [False, "bad"]})
    with pytest.raises(ValueError, match="boolean"):
        getattr(bad, method)("kind").all()


def test_core_flatten_preserves_later_columns_and_pads_missing_values():
    core = type(fixture().groupby("kind"))
    groups = core([BaseTable.from_dict({"a": [1]}), BaseTable.from_dict({"b": [2, 3]})])
    flattened = groups.concat()
    assert column(flattened, "a") == [1, None, None]
    assert column(flattened, "b") == [None, 2, 3]
    assert column(groups.agg({"b": "count"}), "b") == [2]


@pytest.mark.parametrize("method", ["group_by", "groupby"])
def test_null_reductions_counts_and_logical_unique_values(method):
    table = BaseTable.from_dict({"kind": [1, 1, 2], "x": [None, None, 3], "text": ["a", "a", None]})
    groups = getattr(table, method)("kind")
    assert column(groups.agg({"x": "sum"}), "x") == [None, 3]
    assert column(groups.agg({"x": "count"}), "x") == [0, 1]
    assert column(groups.agg({"text": "nunique"}), "text") == [1, 0]


@pytest.mark.parametrize("method", ["group_by", "groupby"])
def test_reductions_retain_frozen_sources_and_remain_read_only(method):
    graph = Graph()
    graph.add_nodes([{"kind": 1, "x": 2}, {"kind": 1, "x": 3}])
    table = graph.read_view().nodes_table().base_table()
    result = getattr(table, method)("kind").agg({"x": "sum"})
    assert column(result, "x") == [5]
    assert len(result.source_views()) == 1
    assert result.source_entity_ids is None
    graph.set_node_attr(0, "x", 99)
    assert result.source_views()[0].node_attr(0, "x") == 2
    with pytest.raises(ValueError, match="explicit identity"):
        result.update_source_graph(graph, ["x"])


def test_specialized_edge_groups_keep_group_keys_in_native_reductions():
    graph = Graph()
    graph.add_nodes([{}, {}])
    graph.add_edges([(0, 1, {"kind": "a", "weight": 2}), (0, 1, {"kind": "a", "weight": 3})])
    groups = graph.read_view().edges_table().group_by("kind")
    result = groups.agg({"weight": "sum"})
    assert column(result, "kind") == ["a"]
    assert column(result, "weight") == [5]
    assert len(result.source_views()) == 1
