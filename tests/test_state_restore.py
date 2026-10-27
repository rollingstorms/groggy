"""Explicit retained-state restoration changes selection without implicit history pruning."""
import pytest

from groggy import DurableGraph, Graph


def fixture():
    graph = Graph()
    graph.add_nodes(2)
    graph.add_edge(0, 1, weight=7)
    first = graph.current_state_id
    graph.set_edge_endpoints(0, 1, 0)
    graph.set_node_attr(0, "value", "future")
    return graph, first, graph.current_state_id


def test_restore_state_recovers_selected_contents_and_keeps_future_and_immutable_views():
    graph, first, future = fixture()
    pin = graph.read_view()
    versions = len(graph.commit_history())
    assert graph.restore_state(first, mode="switch", expected_state=future) == first
    assert graph.edge_endpoints(0) == (0, 1)
    assert pin.edge_endpoints(0) == (1, 0)
    assert len(graph.commit_history()) == versions
    assert graph.snapshot_at_commit(future).node_attr(0, "value").value == "future"
    writer = graph.writer()
    assert writer.restore_state(future, mode="switch") == future
    assert writer.state_id == future
    assert graph.edge_endpoints(0) == (1, 0)
    graph.check_integrity()


def test_restore_rejects_stale_unknown_root_boolean_and_dirty_manual_state():
    graph, first, future = fixture()
    for target, expected in [(first, first), (999, future), (0, future)]:
        with pytest.raises(ValueError):
            graph.restore_state(target, mode="switch", expected_state=expected)
        assert graph.current_state_id == future
    with pytest.raises(TypeError):
        graph.restore_state(True, mode="switch")
    manual = Graph(manual=True)
    manual.add_node()
    target = manual.commit("initial", "test")
    manual.set_node_attr(0, "value", 1)
    with pytest.raises(ValueError, match="uncommitted"):
        manual.restore_state(target, mode="switch")
    assert manual.has_uncommitted_changes()
    manual.reset_hard()
    assert manual.restore_state(target, mode="switch") == target


def test_durable_restore_reopen_backup_and_noop_do_not_add_versions_or_records(tmp_path):
    graph, first, future = fixture()
    path, backup = tmp_path / "graph", tmp_path / "backup"
    with DurableGraph.create_from_graph(path, graph) as store:
        assert store.writer().restore_state(first, mode="switch") == first
        before = path.read_bytes()
        assert store.restore_state(first, mode="switch") == first
        assert path.read_bytes() == before
        store.verify_storage()
        store.backup(backup)
    for candidate in (path, backup):
        with DurableGraph.open(candidate) as store:
            assert store.current_state_id == first
            assert store.snapshot_at_commit(future).node_attr(0, "value").value == "future"
            store.writer().restore_state(future, mode="switch")
            store.verify_storage()
