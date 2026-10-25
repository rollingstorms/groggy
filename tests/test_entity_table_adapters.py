"""Checked ID-mapped attribute columns and exact entity ID readers."""
import pytest
from groggy import BaseTable, EdgesTable, Graph, NodesTable


def column(table, name):
    value = table[name]
    return value.to_list() if hasattr(value, "to_list") else [value]


def entities(kind):
    if kind == "nodes":
        return NodesTable.from_base_table(BaseTable.from_dict({"node_id": [7, 3]})), "node_id", [7, 3]
    return EdgesTable.from_base_table(BaseTable.from_dict({"edge_id": [9, 5], "source": [7, 3], "target": [3, 7]})), "edge_id", [9, 5]


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_attribute_attachment_uses_ids_preserves_order_and_fills_missing_rows(kind):
    table, id_column, ids = entities(kind)
    changed = table.with_attributes("x", {ids[1]: 1.0000000000000002})
    assert column(changed.base_table(), id_column) == ids
    assert column(changed.base_table(), "x") == [None, 1.0000000000000002]
    assert "x" not in table.base_table().columns
    records = changed.with_attributes("x", [{"id": ids[0], "value": [1, 2]}, {"id": ids[1], "value": None}])
    assert column(records.base_table(), "x") == [[1, 2], None]
    assert column(changed.with_attributes("x", {}).base_table(), "x") == [None, None]


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_unknown_and_duplicate_attribute_ids_fail_without_changing_input(kind):
    table, _, ids = entities(kind)
    with pytest.raises(ValueError, match="Unknown"):
        table.with_attributes("x", {999: 1})
    with pytest.raises(ValueError, match="Duplicate"):
        table.with_attributes("x", [{"id": ids[0], "value": 1}, {"id": ids[0], "value": 2}])
    assert "x" not in table.base_table().columns


@pytest.mark.parametrize("attributes", [[{"id": 7}], [{"value": 1}], {True: 1}, {7.5: 1}, "invalid"])
def test_attribute_input_marshaling_errors_are_explicit(attributes):
    table, _, _ = entities("nodes")
    with pytest.raises((TypeError, ValueError)):
        table.with_attributes("x", attributes)


def test_edge_ids_and_endpoints_are_int64_arrays_without_float_rounding():
    wide = 2**53 + 1
    table = EdgesTable.from_base_table(BaseTable.from_dict({"edge_id": [wide], "source": [wide + 2], "target": [wide + 4]}))
    assert table.edge_ids().to_list() == [wide]
    assert table.sources().to_list() == [wide + 2]
    assert table.targets().to_list() == [wide + 4]
    assert table.as_tuples() == [(wide, wide + 2, wide + 4)]
    assert table.with_attributes("x", {wide: 3}).edge_ids().to_list() == [wide]


def test_malformed_structural_values_fail_instead_of_shortening_id_arrays():
    table = EdgesTable.from_base_table(BaseTable.from_dict({"edge_id": [9, 5], "source": [None, 3], "target": [3, 7]}))
    with pytest.raises(ValueError, match="source row 0"):
        table.sources()
    with pytest.raises(ValueError, match="source row 0"):
        table.as_tuples()
    with pytest.raises(RuntimeError):
        table.filter_by_sources([3])


def test_attribute_edits_keep_frozen_source_binding_and_publish_only_explicitly():
    graph = Graph()
    graph.add_nodes([{"x": 1}, {"x": 2}])
    graph.add_edge(0, 1)
    table = graph.read_view().table()
    nodes = table.nodes.with_attributes("x", {1: 9}).base_table()
    edges = table.edges.with_attributes("w", {0: 3}).base_table()
    assert nodes.source_entity_ids == [0, 1]
    assert edges.source_entity_ids == [0]
    assert graph.get_node_attr(1, "x") == 2
    nodes.update_source_graph(graph, ["x"])
    assert graph.get_node_attr(1, "x") == 9
    assert table.nodes.base_table()["x"].to_list() == [1, 2]


def test_mixed_id_mapped_property_columns_publish_one_delta_with_explicit_time():
    from groggy import GraphTable
    graph = Graph()
    graph.add_nodes([{"x": 1}, {"x": 2}])
    graph.add_edge(0, 1, w=1)
    captured = graph.read_view().table()
    nodes = captured.nodes.with_attributes("x", {0: 10, 1: 20})
    edges = captured.edges.with_attributes("w", {0: 4})
    before = graph.current_state_id
    result = GraphTable(nodes, edges).update_source_graph(
        graph, node_columns=["x"], edge_columns=["w"], effective_at=123
    )
    assert result["state_id"] == before + 1
    assert graph.get_node_attr(0, "x") == 10
    assert graph.get_edge_attr(0, "w") == 4
    assert captured.nodes.base_table()["x"].to_list() == [1, 2]
    metadata = graph.publication_metadata(before + 1)
    assert len(metadata["change_times"]) == 3
    assert all(change["effective_at"] == 123 for change in metadata["change_times"])
