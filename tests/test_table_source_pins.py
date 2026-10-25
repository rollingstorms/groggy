"""Frozen graph table exports preserve row identity independently of editable cells."""
import pytest
from groggy import BaseTable, DurableGraph, EdgesTable, Graph, GraphBatch, GraphTable, NodesTable


def fixture():
    graph = Graph()
    graph.add_nodes([{"score": 0}, {"score": 1}, {"score": 2}, {"score": 3}])
    graph.add_edges([(0, 1, {"weight": .5}), (0, 1, {"weight": 1.5}), (1, 1, {"weight": 2.5}), (2, 3, {"weight": 3.5})])
    return graph


def test_filtered_sorted_projection_keeps_write_identity_without_id_column():
    graph = fixture()
    old = graph.read_view()
    table = old.nodes_table().base_table().sort_by("score", ascending=False)
    table = table.filter("score >= 2").select(["score"])
    assert table.source_entity_ids == [3, 2]
    assert table.source_entity_kind == "node"
    head = graph.current_state_id
    table["score"] = [None, 2.0000000000000004]
    result = table.update_source_graph(graph, ["score"], effective_at=10)
    assert result["state_id"] == head + 1
    assert result["view"].node_ids == [2, 3]
    assert graph.get_node_attr(2, "score") == 2.0000000000000004
    assert old.node_attr(2, "score") == 2
    assert table.source_state_id == head
    with pytest.raises(ValueError, match="Stale"):
        table.update_source_graph(graph, ["score"])


def test_selected_parallel_edge_rows_update_exact_instances():
    graph = fixture()
    table = graph.read_view().select_edges([1, 2]).edges_table().base_table()
    table = table.sort_by("weight", ascending=False).select(["weight"])
    assert table.source_entity_ids == [2, 1]
    table["weight"] = [9.5, 7.5]
    result = table.update_source_graph(graph, ["weight"])
    assert result["view"].edge_ids == [1, 2]
    assert graph.get_edge_attr(0, "weight") == .5
    assert graph.get_edge_attr(1, "weight") == 7.5
    assert graph.get_edge_attr(2, "weight") == 9.5


def test_graph_table_mixed_columns_publish_once_and_preserve_shared_capture():
    graph = fixture()
    exported = graph.read_view().table()
    nodes = exported.nodes.base_table()
    edges = exported.edges.base_table()
    nodes["score"] = [10, 11, 12, 13]
    edges["weight"] = [10.5, 11.5, 12.5, 13.5]
    bundle = GraphTable(NodesTable.from_base_table(nodes), EdgesTable.from_base_table(edges))
    before = graph.current_state_id
    result = bundle.update_source_graph(graph, node_columns=["score"], edge_columns=["weight"], effective_at=-5)
    assert result["state_id"] == before + 1
    assert graph.get_node_attr(3, "score") == 13
    assert graph.get_edge_attr(1, "weight") == 11.5
    assert len(graph.publication_metadata(before + 1)["change_times"]) == 8
    assert exported.nodes["score"].to_list() == [0, 1, 2, 3]


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer", "cross"])
def test_join_keeps_frozen_sources_without_guessing_write_identity(how):
    graph = fixture()
    other = fixture()
    left = graph.read_view().nodes_table().base_table()
    right = other.read_view().nodes_table().base_table()
    joined = left.join(right, on="node_id", how=how)
    assert len(joined.source_views()) == 2
    assert joined.source_entity_ids is None
    assert joined.source_state_id is None
    graph.set_node_attr(0, "score", 99)
    assert joined.source_views()[0].node_attr(0, "score") == 0
    with pytest.raises(ValueError, match="explicit identity"):
        joined.update_source_graph(graph, ["score"])
    assert len(joined.head(0).source_views()) == 2


@pytest.mark.parametrize("kind", ["node", "edge"])
def test_structural_edits_and_foreign_graphs_cannot_retarget_bound_rows(kind):
    graph = fixture()
    view = graph.read_view()
    table = (view.nodes_table() if kind == "node" else view.edges_table()).base_table()
    column = "score" if kind == "node" else "weight"
    foreign = fixture()
    with pytest.raises(ValueError, match="different graph"):
        table.update_source_graph(foreign, [column])
    table["node_id" if kind == "node" else "edge_id"] = [3, 2, 1, 0]
    with pytest.raises(ValueError, match="retained row identity"):
        table.update_source_graph(graph, [column])
    assert graph.current_state_id == view.write_state_id


def test_empty_slices_and_unbound_tables_do_not_invent_graph_identity():
    graph = fixture()
    table = graph.read_view().nodes_table().base_table()
    empty = table.head(0)
    assert empty.source_entity_ids == []
    assert empty.source_entity_kind == "node"
    assert empty["score"].to_list() == []
    assert empty.source_state_id == graph.current_state_id
    plain = BaseTable.from_dict({"node_id": [0], "score": [99]})
    assert plain.source_entity_ids is None
    with pytest.raises(ValueError, match="no pinned graph source"):
        plain.update_source_graph(graph, ["score"])


def test_mixed_independent_captures_reject_even_with_equal_numeric_state():
    graph = fixture()
    bundle = GraphTable(graph.read_view().nodes_table(), graph.read_view().edges_table())
    with pytest.raises(ValueError, match="exact source capture"):
        bundle.update_source_graph(graph, node_columns=["score"], edge_columns=[])


def test_durable_table_writeback_syncs_and_recovers(tmp_path):
    path = tmp_path / "tables.ggj"
    with DurableGraph.create(path) as graph:
        batch = GraphBatch()
        batch.add_node(score=1)
        graph.apply_batch(batch, 0, "seed", "test")
        table = graph.read_view().nodes_table().base_table()
        table["score"] = [9]
        result = table.update_source_graph(graph, ["score"], effective_at=10)
        assert result["view"].node_attr(0, "score") == 9
        before = path.read_bytes()
        with pytest.raises(ValueError, match="Stale"):
            table.update_source_graph(graph, ["score"])
        assert path.read_bytes() == before
    with DurableGraph.open(path) as graph:
        assert graph.get_node_attr(0, "score") == 9
        with pytest.raises(ValueError, match="different graph"):
            table.update_source_graph(graph, ["score"])
