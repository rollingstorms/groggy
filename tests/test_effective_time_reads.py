"""Effective-time graph reads use accepted ancestry and preserve immutable pins."""
import pytest
from groggy import Graph, GraphBatch, DurableGraph


def create(graph):
    batch = GraphBatch(effective_at=10)
    left = batch.add_node(value=1, payload=b"\x00\xff")
    right = batch.add_node()
    batch.add_edge(left, right, weight=2.5)
    return graph.apply_batch(batch, graph.current_state_id, "initial", "test")["state_id"]


def update(graph, at, value):
    batch = GraphBatch(effective_at=at)
    batch.set_node_attr(0, "value", value)
    return graph.apply_batch(batch, graph.current_state_id, "update", "test")["state_id"]


def test_corrections_pin_state_time_and_property_values_without_publishing():
    graph = Graph()
    create(graph)
    update(graph, 30, 3)
    update(graph, 20, 2)
    head = graph.current_state_id
    view = graph.read_view_at(25)
    assert view.node_attr(0, "value") == 2
    assert view.node_attr(0, "payload") == b"\x00\xff"
    assert view.filter_nodes("value == 2").node_ids == [0]
    assert view.bfs_distances(0) == [(0, 0), (1, 1)]
    assert view.read_time == {"branch": "main", "published_head": head, "effective_at": 25, "recorded_by": None}
    assert view.select_nodes([0]).read_time == view.read_time
    assert view.write_state_id is None
    assert graph.read_view_at(35).node_attr(0, "value") == 3
    assert graph.read_view_at(9).node_ids == []
    assert graph.current_state_id == head
    update(graph, 20, None)
    assert graph.read_view_at(25).node_attr(0, "value") is None
    assert view.node_attr(0, "value") == 2
    assert graph.read_view().read_time is None


def test_recorded_cutoff_is_explicit_and_history_survives_durable_reopen(tmp_path):
    import time
    path = tmp_path / "effective.ggj"
    with DurableGraph.create(path, directed=True) as graph:
        first = create(graph)
        recorded = graph.publication_metadata(first)["recorded_at"]
        time.sleep(0.002)
        update(graph, 20, 2)
        view = graph.read_view_at(25)
        assert view.is_directed
        assert graph.read_view_at(25, recorded_by=recorded).node_attr(0, "value") == 1
    with DurableGraph.open(path) as graph:
        assert graph.read_view_at(25).node_attr(0, "value") == 2
        old = graph.read_view_at(25, recorded_by=recorded)
        assert old.node_attr(0, "value") == 1
        assert old.read_time["recorded_by"] == recorded
        assert old.table().nodes.base_table()["payload"].to_list()[0] == b"\x00\xff"
    assert view.node_attr(0, "value") == 2


@pytest.mark.parametrize("clock", [True, 1.5, "yesterday", 2**80])
def test_time_read_rejects_invalid_clocks_without_mutation(clock):
    graph = Graph()
    create(graph)
    head = graph.current_state_id
    with pytest.raises((TypeError, OverflowError)):
        graph.read_view_at(clock)
    with pytest.raises((TypeError, OverflowError)):
        graph.read_view_at(25, recorded_by=clock)
    assert graph.current_state_id == head


def test_temporal_table_cannot_implicitly_write_back_to_current_head():
    graph = Graph()
    create(graph)
    view = graph.read_view_at(15)
    table = view.nodes_table().base_table()
    table["value"] = [7, 8]
    with pytest.raises(ValueError):
        table.update_source_graph(graph, ["value"])
    assert graph.read_view().node_attr(0, "value") == 1


def test_join_and_table_captures_retain_temporal_context_after_owner_changes():
    graph = Graph()
    create(graph)
    update(graph, 20, 2)
    view = graph.read_view_at(15)
    table = view.nodes_table().base_table()
    joined = table.join(table, on="node_id")
    assert len(joined.source_views()) == 1
    assert joined.source_views()[0].read_time == view.read_time
    assert joined.source_views()[0].node_attr(0, "value") == 1
    update(graph, 10, 9)
    del graph
    assert table.source_views()[0].read_time == view.read_time
    assert joined.source_views()[0].node_attr(0, "value") == 1
