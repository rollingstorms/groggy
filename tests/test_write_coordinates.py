"""A writer cannot follow a same-head branch switch without explicit refresh."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from groggy import DurableGraph, Graph, GraphBatch


def source():
    g = Graph()
    g.add_node(value=0)
    root = g.current_state_id
    g.create_branch("right")
    g.checkout_branch("right")
    g.set_node_attr(0, "value", 2)
    right = g.current_state_id
    g.checkout_branch("main")
    g.create_branch("same")
    return g, root, right


@pytest.mark.parametrize("durable", [False, True])
def test_same_head_branch_switch_rejects_every_writer_entry_without_bytes_or_versions(tmp_path, durable):
    g, root, right = source()
    path = tmp_path / "graph"
    graph = DurableGraph.create_from_graph(path, g) if durable else g
    try:
        writer = graph.writer()
        if durable:
            graph.checkout_branch("same", graph.current_state_id)
        else:
            graph.checkout_branch("same")
        history = len(graph.commit_history())
        checkpoints = graph.checkpoints()
        before = path.read_bytes() if durable else None
        for operation in (
            lambda: writer.add_node(), lambda: writer.checkpoint(),
            lambda: writer.restore_state(root, mode="switch"),
            lambda: writer.restore_state(root, mode="tracked"),
            lambda: writer.merge(right),
            lambda: writer.upsert_nodes([{"external_id": "new"}]),
            lambda: writer.upsert_edges([{"external_id": "new-edge", "source": 0, "target": 0}]),
        ):
            with pytest.raises(ValueError, match="coordinate"):
                operation()
            assert writer.state_id == root
            assert writer.branch == "main"
            assert graph.current_state_id == root
            assert len(graph.commit_history()) == history
            assert graph.checkpoints() == checkpoints
            if durable:
                assert path.read_bytes() == before
        assert writer.refresh() == root
        assert writer.branch == "same"
        assert writer.add_node() == 1
        assert writer.state_id == graph.current_state_id
        if durable:
            graph.checkout_branch("main", graph.current_state_id)
        else:
            graph.checkout_branch("main")
        assert graph.current_state_id == root
        assert graph.read_view().node_count() == 1
        if durable:
            graph.verify_storage()
    finally:
        if durable:
            graph.close()


@pytest.mark.parametrize("durable", [False, True])
def test_foreign_coordinate_rejects_even_with_matching_branch_and_head(tmp_path, durable):
    g, root, _ = source()
    foreign = Graph()
    foreign.add_node()
    token = foreign.write_coordinate()
    assert token.state_id == root
    path = tmp_path / "graph"
    graph = DurableGraph.create_from_graph(path, g) if durable else g
    try:
        batch = GraphBatch()
        batch.add_node()
        before = path.read_bytes() if durable else None
        with pytest.raises(ValueError, match="foreign"):
            graph.apply_batch(batch, root, "bad", "test", expected_coordinate=token)
        assert graph.current_state_id == root
        if durable:
            assert path.read_bytes() == before
    finally:
        if durable:
            graph.close()


def test_coordinates_are_immutable_thread_safe_identity_markers():
    g = Graph()
    token = g.write_coordinate()
    with pytest.raises(AttributeError):
        token.state_id = 7
    with pytest.raises(AttributeError):
        token.branch = "other"
    with ThreadPoolExecutor(max_workers=1) as workers:
        assert workers.submit(lambda: (token.state_id, token.branch)).result() == (0, "main")
    with pytest.raises(TypeError):
        token._advance(True)


def test_merge_callback_refresh_cannot_change_the_inflight_writer_coordinate():
    g, _, right = source()
    g.set_node_attr(0, "value", 1)
    left = g.current_state_id
    g.create_branch("callback")
    writer = g.writer()

    def resolve(conflict):
        g.checkout_branch("callback")
        writer.refresh()
        assert writer.branch == "callback"
        g.checkout_branch("main")
        return "right"

    state = writer.merge(right, resolver=resolve)
    assert state > left and writer.state_id == state
    assert writer.branch == "main"
    writer.set_node_attr(0, "after", True)
    assert g.get_node_attr(0, "after") is True


def test_physical_maintenance_preserves_writer_owner_and_reopen_requires_new_coordinate(tmp_path):
    source = Graph()
    source.add_node()
    path = tmp_path / "graph"
    with DurableGraph.create_from_graph(path, source) as graph:
        writer = graph.writer()
        old_coordinate = graph.write_coordinate()
        graph.checkpoint_storage()
        graph.compact_storage(reclaim_unreferenced=True)
        writer.set_node_attr(0, "value", 3)
        assert writer.branch == "main"
        state = graph.current_state_id
    with DurableGraph.open(path) as graph:
        batch = GraphBatch()
        batch.add_node()
        before = path.read_bytes()
        with pytest.raises(ValueError, match="foreign"):
            graph.apply_batch(batch, state, "bad", "test", expected_coordinate=old_coordinate)
        assert path.read_bytes() == before
        assert graph.writer().add_node() == 1
        graph.verify_storage()
