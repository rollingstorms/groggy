"""Durable exact-slot compaction changes storage layout without losing graph history."""
import os

import pytest
from groggy import DurableGraph, Graph, GraphBatch

pytestmark = pytest.mark.skipif(os.name != "posix", reason="atomic replacement requires Unix")


def test_durable_compaction_reopen_backup_and_continued_writes(tmp_path):
    graph = Graph()
    graph.add_node(value="payload" * 200)
    graph.create_branch("original")
    for _ in range(10):
        graph.set_node_attr(0, "value", "payload" * 200)
    state = graph.current_state_id
    path = tmp_path / "graph.ggj"
    backup = tmp_path / "backup.ggj"
    with DurableGraph.create_from_graph(path, graph) as store:
        before = store.check_integrity()
        pin = store.read_view_at(2**63 - 1)
        report = store.compact_storage()
        slots = report["pool_compaction"]
        assert slots["current_node_values"] < slots["previous_node_values"]
        assert report["current_bytes"] == path.stat().st_size < report["previous_bytes"]
        assert store.current_state_id == state
        assert store.check_integrity()["retained_versions"] == before["retained_versions"]
        assert pin.node_attr(0, "value") == "payload" * 200
        again = store.compact_storage()["pool_compaction"]
        assert again["previous_node_values"] == again["current_node_values"]
        assert "pool_compaction" not in store.checkpoint_storage()
        batch = GraphBatch()
        batch.add_node(value="after")
        store.apply_batch(batch, state, "after", "test")
        latest = store.current_state_id
        store.backup(backup)
    for candidate in (path, backup):
        with DurableGraph.open(candidate) as store:
            assert store.current_state_id == latest
            assert store.check_integrity()["active_nodes"] == 2
            store.checkout_branch("original", latest)
            assert store.check_integrity()["active_nodes"] == 1


def test_stale_durable_compaction_preserves_file_and_writer(tmp_path):
    path = tmp_path / "graph.ggj"
    graph = Graph()
    graph.add_node(value=1)
    with DurableGraph.create_from_graph(path, graph) as store:
        before = path.read_bytes()
        with pytest.raises(ValueError, match="Stale"):
            store.compact_storage(expected_state=0)
        assert path.read_bytes() == before
        assert store.current_state_id == 1
        store.compact_storage(expected_state=1)
