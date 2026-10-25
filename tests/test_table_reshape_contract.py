"""Typed entity filters and checked table reshape boundaries."""
import pytest
from groggy import BaseTable, EdgesTable, Graph, NodesTable


def column(table, name):
    value = table[name]
    return value.to_list() if hasattr(value, "to_list") else [value]


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_attribute_filter_matches_literal_values_without_query_interpolation(kind):
    data = {"odd column": ['a" == null or x == "b', None, [1, 2], 1.0000000000000002, 7]}
    if kind == "nodes":
        data["node_id"] = list(range(5))
        table = NodesTable.from_base_table(BaseTable.from_dict(data))
        id_column = "node_id"
    else:
        data.update(edge_id=list(range(5)), source=[0] * 5, target=[0] * 5)
        table = EdgesTable.from_base_table(BaseTable.from_dict(data))
        id_column = "edge_id"
    for i, value in enumerate(data["odd column"]):
        result = table.filter_by_attr("odd column", value).base_table()
        assert column(result, id_column) == [i]
    with pytest.raises(RuntimeError, match="not found"):
        table.filter_by_attr("missing", None)


@pytest.mark.parametrize("rows", [0, 1])
def test_melt_validates_keys_and_output_names_on_empty_and_nonempty_tables(rows):
    table = BaseTable.from_dict({"id": [1] * rows, "x": [2] * rows})
    for ids, values, variable, value in [
        (["id", "id"], ["x"], "variable", "value"),
        (["id"], ["x", "x"], "variable", "value"),
        (["id"], ["id"], "variable", "value"),
        (["id"], ["x"], "id", "value"),
        (["id"], ["x"], "same", "same"),
    ]:
        with pytest.raises(ValueError):
            table.melt(values, variable, value, id_vars=ids)


def test_pivot_group_order_is_first_occurrence_and_duplicate_keys_reject():
    table = BaseTable.from_dict({"g": [20, 10, 20], "k": ["x"] * 3, "v": [2, 4, 8]})
    for _ in range(8):
        pivot = table.pivot_table(["g"], "k", "v", "sum")
        assert column(pivot, "g") == [20, 10]
        assert column(pivot, "k_x") == [10.0, 4.0]
    with pytest.raises(ValueError, match="repeated"):
        table.pivot_table(["g", "g"], "k", "v", "sum")


def test_typed_filter_preserves_source_binding_and_immutable_values():
    graph = Graph()
    graph.add_nodes([{"x": 1}, {"x": 2}])
    nodes = graph.read_view().table().nodes.filter_by_attr("x", 2).base_table()
    assert nodes.source_entity_ids == [1]
    graph.set_node_attr(1, "x", 99)
    assert column(nodes, "x") == [2]
    assert nodes.source_views()[0].node_attr(1, "x") == 2


def test_melt_identifier_keyword_preserves_existing_positional_api():
    table = BaseTable.from_dict({"id": [8], "x": [2], "y": [3]})
    plain = table.melt(["x", "y"], "metric", "score")
    assert plain.columns == ["metric", "score"]
    assert column(plain, "score") == [2, 3]
    identified = table.melt(["x", "y"], "metric", "score", id_vars=["id"])
    assert identified.columns == ["id", "metric", "score"]
    assert column(identified, "id") == [8, 8]
