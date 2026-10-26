"""Timed topology failures must precede pool allocation and durable publication."""
import pytest
from groggy import Graph, GraphBatch, DurableGraph


def initial(graph, at=10):
    batch = GraphBatch(effective_at=at)
    a = batch.add_node()
    b = batch.add_node()
    batch.add_edge(a, b)
    graph.apply_batch(batch, graph.current_state_id, "initial", "test")


def test_invalid_explicit_cascade_time_rejects_atomically_then_valid_retry_succeeds():
    graph = Graph()
    initial(graph)
    head = graph.current_state_id
    batch = GraphBatch(effective_at=20)
    batch.remove_node(0)
    batch.remove_edge(0, effective_at=30)
    with pytest.raises(ValueError, match="endpoint timeline"):
        graph.apply_batch(batch, head, "invalid", "test")
    assert graph.current_state_id == head
    assert graph.read_view().node_ids == [0, 1]
    assert graph.read_view_at(40).edge_ids == [0]
    batch = GraphBatch(effective_at=20)
    batch.remove_node(0)
    graph.apply_batch(batch, head, "valid", "test")
    assert graph.read_view_at(20).node_ids == [1]
    assert graph.read_view_at(20).edge_ids == []


def test_invalid_new_entities_leave_ids_available_for_retry():
    graph = Graph()
    batch = GraphBatch(effective_at=20)
    a = batch.add_node()
    b = batch.add_node()
    batch.set_node_external_id(a, "future")
    batch.add_edge(a, b, effective_at=19)
    with pytest.raises(ValueError, match="endpoint timeline"):
        graph.apply_batch(batch, 0, "invalid", "test")
    assert graph.allocated_entity_for_external_id("future") is None
    assert graph.current_state_id == 0
    assert graph.read_view().node_ids == []
    initial(graph)
    assert graph.read_view().node_ids == [0, 1]
    assert graph.read_view().edge_ids == [0]


def test_durable_failure_has_no_frame_and_valid_future_timeline_reopens(tmp_path):
    path = tmp_path / "timeline.ggj"
    with DurableGraph.create(path) as graph:
        initial(graph, at=2**63 - 1)
        head = graph.current_state_id
        before = path.read_bytes()
        batch = GraphBatch(effective_at=2**63 - 2)
        batch.add_edge(0, 1)
        with pytest.raises(ValueError, match="endpoint timeline"):
            graph.apply_batch(batch, head, "invalid", "test")
        assert path.read_bytes() == before
        assert graph.current_state_id == head
    with DurableGraph.open(path) as graph:
        assert graph.read_view_at(2**63 - 1).edge_ids == [0]
        assert graph.read_view_at(2**63 - 2).node_ids == []
