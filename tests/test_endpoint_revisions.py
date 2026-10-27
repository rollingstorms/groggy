"""Stable edge endpoint revisions across the checked Rust boundary."""
import pytest

from groggy import DurableGraph, Graph, GraphBatch


def source():
    graph = Graph()
    batch = GraphBatch(effective_at=10)
    a, b, c = [batch.add_node() for _ in range(3)]
    batch.add_edge(a, b, weight=7)
    graph.apply_batch(batch, 0, "initial", "test")
    return graph


def test_endpoint_revision_bulk_times_and_old_views_preserve_edge_properties(tmp_path):
    graph = source()
    first, pin = graph.current_state_id, graph.read_view()
    graph.writer().set_edge_external_id(0, "fact", effective_at=10)
    changed = graph.writer().set_edge_endpoints(0, 2, 1, effective_at=20)
    assert graph.edge_endpoints(0) == (2, 1)
    assert pin.edge_endpoints(0) == (0, 1)
    assert graph.read_view_at(19).edge_endpoints(0) == (0, 1)
    assert graph.read_view_at(20).edge_endpoints(0) == (2, 1)
    assert graph.get_edge_attr(0, "weight") == 7
    assert graph.edge_by_external_id("fact") == 0
    delta = graph.diff_commits(first, changed)
    assert delta["edge_endpoint_changes"] == [{"edge_id": 0, "old": (0, 1), "new": (2, 1)}]
    assert delta["edges_added"] == delta["edges_removed"] == []
    graph.set_edge_endpoints_bulk([(0, 1, 2)], effective_at=20)
    assert graph.read_view_at(20).edge_endpoints(0) == (1, 2)
    path = tmp_path / "archive"
    graph.save_archive(path)
    copy = Graph.load_archive(path)
    assert copy.edge_endpoints(0) == (1, 2)
    copy.reverse_edges([0], effective_at=30)
    assert copy.edge_endpoints(0) == (2, 1)
    copy.check_integrity()


def test_invalid_endpoint_revision_batches_have_no_effect_and_mixed_writes_are_atomic():
    graph = source()
    state = graph.current_state_id
    for rows, at in [([(0, 2, 1), (0, 1, 2)], 20), ([(0, 99, 1)], 20), ([(0, 2, 1)], 9)]:
        with pytest.raises(ValueError):
            graph.writer().set_edge_endpoints_bulk(rows, effective_at=at)
        assert graph.current_state_id == state
        assert graph.edge_endpoints(0) == (0, 1)
    batch = GraphBatch(effective_at=20)
    node = batch.add_node()
    batch.set_edge_endpoints(0, node, 1, effective_at=25)
    batch.set_edge_attr(0, "weight", 8)
    graph.apply_batch(batch, state, "mixed", "test")
    assert graph.edge_endpoints(0) == (3, 1)
    assert graph.get_edge_attr(0, "weight") == 8
    assert graph.read_view_at(24).edge_endpoints(0) == (0, 1)
    assert graph.read_view_at(25).edge_endpoints(0) == (3, 1)
    graph.check_integrity()


def test_durable_endpoint_revision_reopen_and_explicit_maintenance(tmp_path):
    graph = source()
    path = tmp_path / "graph"
    with DurableGraph.create_from_graph(path, graph) as store:
        state = store.writer().set_edge_endpoints(0, 2, 1, effective_at=20)
        store.checkpoint_storage(expected_state=state)
        store.compact_storage(expected_state=state, reclaim_unreferenced=True)
        store.purge_unreferenced(expected_state=state)
        store.verify_storage()
    with DurableGraph.open(path) as store:
        assert store.read_view_at(19).edge_endpoints(0) == (0, 1)
        assert store.read_view_at(20).edge_endpoints(0) == (2, 1)
        store.check_integrity()


def test_endpoint_conveniences_reject_boolean_ids_and_times_before_writing():
    graph = source()
    state = graph.current_state_id
    operations = [
        lambda: graph.set_edge_endpoints(True, 2, 1),
        lambda: graph.set_edge_endpoints(0, False, 1),
        lambda: graph.set_edge_endpoints(0, 2, 1, effective_at=True),
        lambda: graph.set_edge_endpoints_bulk([(0, 2, False)]),
        lambda: graph.reverse_edges([False]),
    ]
    for operation in operations:
        with pytest.raises(TypeError):
            operation()
        assert graph.current_state_id == state
        assert graph.edge_endpoints(0) == (0, 1)
