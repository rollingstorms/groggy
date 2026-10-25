"""Grouping, deduplication and table combinations use stable typed row keys."""
import pytest
from groggy import BaseTable, Graph


def column(table, name):
    value = table[name]
    return value.to_list() if hasattr(value, "to_list") else [value]


def test_groups_and_dedup_keep_first_occurrences_and_null_groups():
    table = BaseTable.from_dict({"k": [2, 1, 2, None, None], "value": [10, 11, 12, 13, 14]})
    for groups in [table.group_by("k"), table.groupby("k")]:
        assert len(groups) == 3
        assert column(groups[0], "value") == [10, 12]
        assert column(groups[1], "value") == [11]
        assert column(groups[2], "value") == [13, 14]
    assert column(table.drop_duplicates(["k"]), "value") == [10, 11, 13]


def test_signed_zero_and_nan_identity_are_shared_with_set_operations():
    table = BaseTable.from_dict({"k": [-0.0, 0.0, float("nan"), float("nan")]})
    assert len(table.drop_duplicates()) == 2
    assert len(table.union(table)) == 2
    assert len(table.intersect(table)) == 2
    assert len(table.groupby("k")) == 2


def test_delimited_text_does_not_merge_distinct_composite_rows():
    table = BaseTable.from_dict({"x": ["a|t:b", "a"], "y": ["c", "b|t:c"]})
    assert len(table.drop_duplicates()) == 2
    assert len(table.union(table)) == 2
    assert len(table.drop_duplicates([])) == 1


def test_stable_sets_and_stack_keep_explicit_semantics():
    left = BaseTable.from_dict({"k": [2, 1, 2, None]})
    right = BaseTable.from_dict({"k": [1, 3, None]})
    assert column(left.union(right), "k") == [2, 1, None, 3]
    assert column(left.intersect(right), "k") == [1, None]
    assert column(left.stack(right), "k") == [2, 1, 2, None, 1, 3, None]
    with pytest.raises(RuntimeError, match="identical ordered"):
        left.stack(BaseTable.from_dict({"other": [1]}))


def test_concat_pads_in_input_column_order_and_rejects_name_collisions():
    left = BaseTable.from_dict({"b": [1], "a": [2]}).select(["b", "a"])
    right = BaseTable.from_dict({"a": [3, 4], "c": [5, 6]}).select(["a", "c"])
    result = left.concat(right)
    assert column(result, "b") == [1, None]
    assert column(result, "a_y") == [3, 4]
    assert column(result, "c") == [5, 6]
    with pytest.raises(RuntimeError, match="collides"):
        left.concat(BaseTable.from_dict({"a_y": [9], "a": [10]}))


@pytest.mark.parametrize("method", ["group_by", "groupby", "drop_duplicates"])
def test_repeated_and_missing_group_keys_reject(method):
    table = BaseTable.from_dict({"k": [1, 2]})
    for keys in [["k", "k"], ["missing"]]:
        with pytest.raises((ValueError, RuntimeError)):
            getattr(table, method)(keys)


def test_group_aggregation_validates_empty_inputs_and_preserves_key_order():
    table = BaseTable.from_dict({"k": [2, 1, 2], "value": [10, 11, 12]})
    grouped = table.group_by_agg(["k"], {"value": "sum"})
    assert column(grouped, "k") == [2, 1]
    assert column(grouped, "value") == [22, 11]
    for specs in [{"k": "sum"}, {"value": "bad"}, {"missing": "sum"}]:
        with pytest.raises(RuntimeError):
            table.head(0).group_by_agg(["k"], specs)


def test_group_selection_writeback_and_combined_read_only_sources():
    graph = Graph()
    graph.add_nodes([{"k": 2}, {"k": 1}, {"k": 2}])
    table = graph.read_view().nodes_table().base_table()
    group = table.groupby("k")[0]
    assert group.source_entity_ids == [0, 2]
    assert table.drop_duplicates(["k"]).source_entity_ids == [0, 1]
    group["k"] = [9, 9]
    group.update_source_graph(graph, ["k"])
    assert graph.get_node_attr(2, "k") == 9
    for combined in [table.union(table), table.intersect(table), table.stack(table), table.concat(table)]:
        assert len(combined.source_views()) == 1
        assert combined.source_entity_ids is None
        assert combined.source_views()[0].node_attr(2, "k") == 2
        with pytest.raises(ValueError, match="explicit identity"):
            combined.update_source_graph(graph, ["k"])
