"""Projection schemas cannot repeat physical column names."""
import pytest
from groggy import BaseTable, EdgesTable, Graph, NodesTable


@pytest.mark.parametrize("rows", [0, 2])
def test_base_selection_rejects_duplicate_names_without_corrupting_schema(rows):
    table = BaseTable.from_dict({"x": [1] * rows, "y": [2] * rows})
    with pytest.raises(RuntimeError, match="repeated"):
        table.select(["x", "x"])
    with pytest.raises(RuntimeError, match="not found"):
        table.select(["ghost"])
    assert table.columns == ["x", "y"]
    assert table.shape == (rows, 2)
    assert table.select(["y", "x"]).columns == ["y", "x"]


@pytest.mark.parametrize("kind", ["node", "edge"])
def test_specialized_selection_does_not_silently_remove_duplicate_structural_requests(kind):
    table = NodesTable([0, 1]) if kind == "node" else EdgesTable([(0, 0, 1)])
    name = "node_id" if kind == "node" else "source"
    with pytest.raises(RuntimeError, match="repeated"):
        table.select([name, name])


def test_zero_column_projection_keeps_rows_and_source_identity():
    graph = Graph()
    graph.add_nodes([{"x": 1}, {"x": 2}])
    table = graph.nodes.table().base_table()
    empty = table.select([])
    assert empty.shape == (2, 0)
    assert empty.source_entity_ids == [0, 1]
    with pytest.raises(RuntimeError):
        table.select(["x", "x"])
    assert table.source_entity_ids == [0, 1]
    assert graph.get_node_attr(1, "x") == 2


def test_general_sparse_constructor_keeps_documented_null_padding():
    table = BaseTable.from_dict({"x": [1, 2], "y": [3]})
    assert table["y"].to_list() == [3, None]
    assert table.shape == (2, 2)
