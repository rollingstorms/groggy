"""Explicit complete pool purge preserves retained versions and frozen sources."""
import os

import pytest
from groggy import DurableGraph, Graph, GraphBatch


def fixture():
    graph = Graph()
    graph.add_node(value=0)
    graph.writer().set_node_external_id(0, "keep")
    graph.create_branch("discard")
    graph.checkout_branch("discard")
    node = graph.add_node(payload="frozen")
    edge = graph.add_edge(0, node)
    graph.writer().set_node_external_id(node, "free-node")
    graph.writer().set_edge_external_id(edge, "free-edge")
    pin = graph.read_view()
    group = graph.retain_group(pin)
    graph.checkout_branch("main")
    return graph, node, edge, group, pin


def test_preview_and_explicit_purge_release_orphans_and_keep_current_pin_write_authority(tmp_path):
    graph, node, edge, group, pin = fixture()
    # Every retained version protects entities; unattached groups need an owner reference.
    preview = graph.preview_purge()
    assert preview["purged_nodes"] == preview["purged_edges"] == 0
    assert group in preview["removed_groups"]
    graph.prune_branch("discard")
    source = graph.read_view()
    before = graph.check_integrity()
    preview = graph.preview_purge()
    assert graph.check_integrity() == before
    report = graph.purge_unreferenced()
    assert report == preview
    assert report["purged_nodes"] == report["purged_edges"] == 1
    assert report["node_ranges"] == [(node, node + 1)]
    assert report["edge_ranges"] == [(edge, edge + 1)]
    assert not graph.is_node_allocated(node) and not graph.is_edge_allocated(edge)
    assert graph.allocated_entity_for_external_id("free-node") is None
    assert graph.allocated_entity_for_external_id("keep") == ("node", 0)
    assert pin.node_attr(node, "payload") == "frozen"
    assert graph.check_integrity()["purged_nodes"] == 1
    batch = GraphBatch()
    batch.set_node_attr(0, "value", 7)
    assert graph.apply_view_batch(source, batch, "still authorized", "test")["view"].node_attr(0, "value") == 7
    path = tmp_path / "archive.gga"
    graph.save_archive(path)
    copy = Graph.load_archive(path)
    assert not copy.is_node_allocated(node)
    fresh = copy.add_node()
    assert fresh == node + 1
    copy.writer().set_node_external_id(fresh, "free-node")
    assert copy.node_by_external_id("free-node") == fresh


def test_current_and_historical_name_reservations_survive_if_entity_is_still_referenced():
    graph, node, edge, _, _ = fixture()
    graph.purge_unreferenced()
    assert graph.is_node_allocated(node) and graph.is_edge_allocated(edge)
    assert graph.allocated_entity_for_external_id("free-node") == ("node", node)
    graph.check_integrity()


@pytest.mark.skipif(os.name != "posix", reason="atomic purge replacement requires Unix")
def test_durable_purge_reopen_backup_and_immutable_source_survive(tmp_path):
    graph, node, edge, group, pin = fixture()
    graph.prune_branch("discard")
    path, backup = tmp_path / "graph.ggj", tmp_path / "backup.ggj"
    with DurableGraph.create_from_graph(path, graph) as store:
        state = store.current_state_id
        expected = store.preview_purge()
        before = path.read_bytes()
        with pytest.raises(ValueError, match="Stale"):
            store.purge_unreferenced(expected_state=0)
        assert path.read_bytes() == before
        report = store.purge_unreferenced()["purge"]
        assert report == expected
        assert group in report["removed_groups"]
        assert not store.is_node_allocated(node) and not store.is_edge_allocated(edge)
        assert store.current_state_id == state
        assert store.verify_storage()["graph"]["purged_nodes"] == 1
        assert store.purge_unreferenced()["purge"]["purged_nodes"] == 0
        store.backup(backup)
    for candidate in (path, backup):
        with DurableGraph.open(candidate) as store:
            assert not store.is_node_allocated(node)
            assert store.allocated_entity_for_external_id("free-node") is None
            batch = GraphBatch()
            added = batch.add_node()
            batch.set_node_external_id(added, "free-node")
            result = store.apply_batch(batch, state, "reuse name only", "test")
            assert result["added_nodes"] == [node + 1]
            assert store.node_by_external_id("free-node") == node + 1
    assert pin.node_attr(node, "payload") == "frozen"
