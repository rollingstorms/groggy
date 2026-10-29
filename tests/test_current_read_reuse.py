"""Warm immutable captures must still observe every accepted owner change."""
import pytest

from groggy import DurableGraph, Graph, GraphBatch


def test_warm_captures_refresh_bulk_properties_unsets_endpoints_and_membership():
    graph = Graph(directed=True)
    graph.add_nodes(3)
    graph.add_edge(0, 1, weight=1)
    first = graph.query()
    second = graph.read_view()
    batch = GraphBatch()
    batch.set_node_attr(0, "value", None)
    batch.set_edge_attr(0, "weight", 2)
    graph.writer().apply_batch(batch)
    changed = graph.query()
    assert changed.node_attr(0, "value") is None
    assert changed.query_nodes("value == null") == [0]
    assert first.query_nodes("value == null") == []
    assert second.edge_attr(0, "weight") == 1
    assert changed.edge_attr(0, "weight") == 2
    graph.writer().unset_node_attr(0, "value")
    assert graph.query().query_nodes("value == null") == []
    graph.set_edge_endpoints(0, 2, 1)
    rewired = graph.query()
    assert rewired.edge_endpoints(0) == (2, 1)
    assert changed.edge_endpoints(0) == (0, 1)
    graph.remove_node(2)
    assert graph.query().edge_ids == []
    assert rewired.edge_ids == [0]


def test_warm_same_head_branch_capture_has_fresh_writeback_coordinates():
    graph = Graph()
    graph.add_node(value=1)
    first = graph.query()
    graph.create_branch("other")
    graph.checkout_branch("other")
    second = graph.query()
    batch = GraphBatch()
    batch.set_node_attr(0, "value", 2)
    with pytest.raises(ValueError, match="branch"):
        graph.apply_view_batch(first, batch, "wrong source branch", "test")
    graph.apply_view_batch(second, batch, "fresh source branch", "test")
    assert first.node_attr(0, "value") == second.node_attr(0, "value") == 1
    assert graph.query().node_attr(0, "value") == 2
    graph.checkout_branch("main")
    assert graph.query().node_attr(0, "value") == 1


@pytest.mark.parametrize("maintenance", ["compact_pool", "purge_unreferenced"])
def test_warm_capture_survives_explicit_physical_maintenance(maintenance):
    graph = Graph()
    graph.add_node(value=1)
    graph.set_node_attr(0, "value", 2)
    first = graph.query()
    assert graph.read_view().node_attr(0, "value") == 2
    head = graph.current_state_id
    getattr(graph, maintenance)()
    assert graph.current_state_id == head
    assert first.node_attr(0, "value") == graph.query().node_attr(0, "value") == 2
    graph.set_node_attr(0, "value", 3)
    assert first.node_attr(0, "value") == 2
    assert graph.query().node_attr(0, "value") == 3


def test_warm_durable_captures_refresh_after_write_and_are_independent_of_reopen(tmp_path):
    path = tmp_path / "capture.ggj"
    with DurableGraph.create(path) as graph:
        batch = GraphBatch()
        batch.add_node(value=1)
        graph.writer().apply_batch(batch)
        first = graph.query()
        second = graph.query()
        before = path.read_bytes()
        assert second.node_attr(0, "value") == 1
        assert path.read_bytes() == before
        graph.writer().set_node_attr(0, "value", 2)
        changed = graph.query()
    with DurableGraph.open(path) as graph:
        assert graph.query().node_attr(0, "value") == 2
    assert first.node_attr(0, "value") == second.node_attr(0, "value") == 1
    assert changed.node_attr(0, "value") == 2
