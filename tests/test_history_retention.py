"""Explicit history pruning preserves shared versions, immutable sources and raw reservations."""
import pytest
from groggy import Graph, DurableGraph, GraphBatch


def initial(graph):
    batch = GraphBatch(effective_at=10)
    batch.add_node(v=0)
    graph.apply_batch(batch, 0, "root", "test")
    return graph.current_state_id


def test_explicit_collector_drops_unreferenced_owned_future_without_changing_capture():
    graph = Graph()
    root = initial(graph)
    graph.set_node_attr(0, "v", 9)
    future = graph.current_state_id
    capture = graph.snapshot_at_commit(root, owned=True)
    copy = capture.to_graph()
    assert copy.gc_history() == 1
    with pytest.raises(ValueError):
        copy.snapshot_at_commit(future)
    assert capture.to_graph().snapshot_at_commit(future).node_attr(0, "v").value == 9
    assert graph.snapshot_at_commit(future).node_attr(0, "v").value == 9
    copy.set_node_attr(0, "v", 2)
    assert copy.current_state_id > future


def test_pruning_preserves_shared_ancestry_and_open_immutable_views():
    graph = Graph()
    root = initial(graph)
    graph.create_branch("discard")
    graph.checkout_branch("discard")
    graph.writer().set_node_attr(0, "v", 1, effective_at=20)
    shared = graph.current_state_id
    graph.create_branch("shared")
    graph.writer().set_node_attr(0, "v", 2, effective_at=30)
    gone = graph.current_state_id
    graph.checkpoint("no permanent pin", "test")
    pin = graph.read_view_at(30)
    snapshot = graph.snapshot_at_commit(gone)
    graph.checkout_branch("main")
    assert graph.prune_branch("discard") == 1
    assert graph.current_state_id == root and len(graph.commit_history()) == 2
    assert graph.snapshot_at_commit(shared).node_attr(0, "v").value == 1
    with pytest.raises(ValueError):
        graph.snapshot_at_commit(gone)
    assert pin.node_attr(0, "v") == 2 and snapshot.node_attr(0, "v").value == 2
    assert graph.checkpoints() == []
    assert graph.gc_history() == 0


def test_durable_prune_reopen_and_backup_keep_frozen_sources_and_reservations(tmp_path):
    path, backup = tmp_path / "graph.ggj", tmp_path / "backup.ggj"
    with DurableGraph.create(path) as graph:
        root = initial(graph)
        graph.checkpoint("root", "test")
        graph.create_branch("discard", root)
        graph.checkout_branch("discard", root)
        graph.writer().set_node_attr(0, "v", 1, effective_at=20)
        shared = graph.current_state_id
        graph.create_branch("shared", shared)
        batch = GraphBatch(effective_at=30)
        node = batch.add_node(payload=b"\x00\xff")
        batch.set_node_external_id(node, "side-only")
        graph.apply_batch(batch, shared, "extra", "test")
        group = graph.retain_group(graph.read_view_at(30))
        meta = graph.create_meta_node(group, effective_at=40)
        gone = graph.current_state_id
        graph.checkpoint("gone", "test")
        pin = graph.read_view_at(40)
        graph.checkout_branch("main", gone)
        before = path.read_bytes()
        with pytest.raises(ValueError):
            graph.prune_branch("discard", expected_state=gone)
        assert path.read_bytes() == before
        assert graph.prune_branch("discard", expected_state=root) == 2
        assert len(graph.checkpoints()) == 1
        before = path.read_bytes()
        assert graph.gc_history(expected_state=root) == 0
        assert path.read_bytes() == before
        graph.backup(backup)
    for location in [path, backup]:
        with DurableGraph.open(location) as graph:
            assert graph.current_state_id == root and len(graph.commit_history()) == 2
            assert len(graph.checkpoints()) == 1
            with pytest.raises(ValueError):
                graph.snapshot_at_commit(gone)
            assert graph.snapshot_at_commit(shared).node_attr(0, "v").value == 1
            assert graph.allocated_entity_for_external_id("side-only") == ("node", 1)
            assert graph.node_by_external_id("side-only") is None
            assert graph.writer().add_node() == meta + 1
            assert graph.current_state_id > gone
    assert pin.meta_subgraph(meta).node_attr(1, "payload") == b"\x00\xff"


@pytest.mark.parametrize("name", ["main", "unknown"])
def test_invalid_branch_pruning_preserves_state(name):
    graph = Graph()
    root = initial(graph)
    with pytest.raises(ValueError):
        graph.prune_branch(name)
    assert graph.current_state_id == root and len(graph.commit_history()) == 1


def test_current_and_dirty_branches_cannot_be_pruned_and_collection_is_explicit():
    graph = Graph(manual=True)
    graph.add_node()
    graph.commit("root", "test")
    graph.create_branch("other")
    graph.checkout_branch("other")
    with pytest.raises(ValueError):
        graph.prune_branch("other")
    graph.set_node_attr(0, "pending", 3)
    with pytest.raises(ValueError, match="uncommitted"):
        graph.gc_history()
    assert graph.has_uncommitted_changes() and len(graph.branches()) == 2
