"""Explicit BaseTable adapters validate identity without creating a graph."""
import pytest
from groggy import BaseTable, Graph


def test_custom_node_key_keeps_row_order_types_and_the_original_schema():
    table = BaseTable.from_dict({"uid": [70, 12], "value": [1.23456789012345, None]})
    nodes = table.to_nodes("uid")
    assert nodes.node_ids().to_list() == [70, 12]
    assert nodes.base_table()["value"].to_list() == [1.23456789012345, None]
    assert nodes.base_table().columns == ["node_id", "value"]
    assert table.columns == ["uid", "value"]


def test_edge_aliases_convert_with_parallel_edges_and_self_loops():
    table = BaseTable.from_dict({
        "edge_ids": [91, 92, 93], "src": [70, 70, 12], "dst": [12, 12, 12],
        "weight": [3.75, None, 2],
    })
    edges = table.to_edges()
    assert edges.as_tuples() == [(91, 70, 12), (92, 70, 12), (93, 12, 12)]
    assert edges.base_table()["weight"].to_list() == [3.75, None, 2]
    assert "src" in table.columns


@pytest.mark.parametrize("values", [[None], [-1], [1.5], [True], ["1"], [7, 7]])
def test_checked_node_conversion_rejects_malformed_or_duplicate_ids(values):
    table = BaseTable.from_dict({"uid": values})
    with pytest.raises(ValueError):
        table.to_nodes("uid")
    assert table.columns == ["uid"]


def test_unknown_node_key_collision_and_invalid_edges_reject():
    table = BaseTable.from_dict({"node_id": [1], "uid": [2]})
    with pytest.raises(ValueError, match="not found"):
        table.to_nodes("missing")
    with pytest.raises(ValueError, match="Duplicate column"):
        table.to_nodes("uid")
    for data in [
        {"edge_id": [7, 7], "source": [70, 70], "target": [12, 12]},
        {"edge_id": [None], "source": [70], "target": [12]},
        {"edge_id": [7], "source": [-1], "target": [12]},
    ]:
        with pytest.raises(ValueError):
            BaseTable.from_dict(data).to_edges()


def test_source_identity_survives_exact_conversion_and_mismatches_become_read_only():
    graph = Graph()
    graph.add_nodes([{"x": 1}, {"x": 2}])
    graph.add_edge(0, 1)
    original = graph.nodes.table().base_table()
    nodes = original.rename({"node_id": "uid"}).to_nodes("uid")
    assert nodes.base_table().source_entity_ids == [0, 1]
    assert nodes.base_table().source_entity_kind == "node"
    changed = nodes.base_table()
    changed["x"] = [8, 9]
    changed.update_source_graph(graph, ["x"], effective_at=123)
    assert graph.get_node_attr(0, "x") == 8
    assert original["x"].to_list() == [1, 2]

    edge_rows = graph.edges.table().base_table()
    assert edge_rows.to_edges().base_table().source_entity_kind == "edge"
    endpoints = edge_rows.select(["source"]).to_nodes("source").base_table()
    assert endpoints.source_entity_ids is None
    assert endpoints.source_entity_kind is None
    assert len(endpoints.source_views()) == 1
    edge_rows["target"] = [0]
    changed_topology = edge_rows.to_edges().base_table()
    assert changed_topology.source_entity_ids is None
    assert len(changed_topology.source_views()) == 1


def test_empty_explicit_conversions_keep_declared_schema_without_rows():
    assert BaseTable.from_dict({"node_id": []}).to_nodes().nrows() == 0
    assert BaseTable.from_dict({"edge_id": [], "source": [], "target": []}).to_edges().nrows() == 0
