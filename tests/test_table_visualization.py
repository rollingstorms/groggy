"""Table visualization validates identities before starting a server."""
import pytest
from groggy import BaseTable, EdgesTable, GraphTable, NodesTable


def test_sparse_tables_create_viz_accessors_without_materializing_a_graph():
    nodes = NodesTable.from_base_table(BaseTable.from_dict({
        "node_id": [70, 12], "name": ["seventy", "twelve"],
        "value": [1.23456789012345, None],
    }))
    edges = EdgesTable.from_base_table(BaseTable.from_dict({
        "edge_id": [900, 200, 300], "source": [70, 70, 12],
        "target": [12, 12, 12], "weight": [2.0, 3.75, None],
    }))
    graph_table = GraphTable(nodes, edges)
    assert repr(nodes.viz) == "VizAccessor(NodesTable)"
    assert repr(edges.viz) == "VizAccessor(EdgesTable)"
    assert repr(graph_table.viz) == "VizAccessor(GraphTable)"
    assert nodes.node_ids().to_list() == [70, 12]
    assert edges.as_tuples() == [(900, 70, 12), (200, 70, 12), (300, 12, 12)]


@pytest.mark.parametrize("bad", [None, -1, 1.5, True, "1"])
def test_viz_rejects_invalid_ids_before_server_creation(bad):
    table = NodesTable.from_base_table(BaseTable.from_dict({"node_id": [bad]}))
    with pytest.raises(RuntimeError, match="nonnegative integer ID"):
        _ = table.viz
    edges = EdgesTable.from_base_table(BaseTable.from_dict({
        "edge_id": [0], "source": [bad], "target": [1],
    }))
    with pytest.raises(RuntimeError, match="nonnegative integer ID"):
        _ = edges.viz


def test_viz_rejects_duplicates_and_dangling_endpoints_without_dropping_rows():
    with pytest.raises(RuntimeError, match="Duplicate"):
        _ = NodesTable([12, 12]).viz
    with pytest.raises(RuntimeError, match="Duplicate"):
        _ = EdgesTable([(5, 12, 70), (5, 70, 12)]).viz
    table = GraphTable(NodesTable([70]), EdgesTable([(9, 70, 12)]))
    with pytest.raises(RuntimeError, match="endpoint outside"):
        _ = table.viz
    assert table.nodes.node_ids().to_list() == [70]
    assert table.edges.as_tuples() == [(9, 70, 12)]


def test_empty_tables_have_real_viz_accessors():
    assert repr(NodesTable([]).viz) == "VizAccessor(NodesTable)"
    assert repr(EdgesTable([]).viz) == "VizAccessor(EdgesTable)"
    assert repr(GraphTable(NodesTable([]), EdgesTable([])).viz) == "VizAccessor(GraphTable)"
