"""Explicit physical baselines preserve graph versions and writer ownership."""
import os

import pytest
from groggy import DurableGraph, GraphBatch

pytestmark = pytest.mark.skipif(os.name != "posix", reason="atomic replacement currently requires Unix")


def seed(store):
    batch = GraphBatch(effective_at=10)
    node = batch.add_node(v=1)
    batch.set_node_external_id(node, "original")
    store.apply_batch(batch, store.current_state_id, "root", "test")
    return store.current_state_id


def test_storage_checkpoint_keeps_versions_clocks_pins_schema_and_continued_backup(tmp_path):
    path = tmp_path / "graph.ggj"
    backup = tmp_path / "backup.ggj"
    with DurableGraph.create(path) as store:
        root = seed(store)
        store.create_branch("root", root)
        clocks = store.publication_metadata(root)
        store.set_schema(None, root, 0)
        group = store.retain_group(store.read_view_at(10))
        meta = store.create_meta_node(group, effective_at=15)
        batch = GraphBatch(effective_at=20)
        batch.unset_node_attr(0, "v")
        batch.set_node_external_id(0, "renamed")
        store.apply_batch(batch, store.current_state_id, "update", "test")
        state = store.current_state_id
        store.checkpoint("label", "test", expected_state=state)
        pin = store.read_view_at(20)
        report = store.checkpoint_storage(expected_state=state)
        assert report["selected_state"] == state
        assert report["previous_records"] > 1
        assert report["retained_versions"] == 3
        assert report["current_bytes"] == path.stat().st_size
        assert store.current_state_id == state and store.schema_revision == 1
        assert store.publication_metadata(root) == clocks
        assert pin.node_attr(0, "v") is None
        assert pin.meta_subgraph(meta).node_attr(0, "v") == 1
        with pytest.raises(Exception):
            DurableGraph.open(path)
        batch = GraphBatch(effective_at=30)
        batch.add_node()
        store.apply_batch(batch, state, "after", "test")
        latest = store.current_state_id
        store.backup(backup)
    for path in (path, backup):
        with DurableGraph.open(path) as store:
            assert store.current_state_id == latest and store.schema_revision == 1
            assert len(store.checkpoints()) == 1
            assert store.allocated_entity_for_external_id("original") == ("node", 0)
            assert store.allocated_entity_for_external_id("renamed") == ("node", 0)
            assert store.read_view_at(20).meta_subgraph(meta).node_attr(0, "v") == 1
            store.checkout_branch("root", latest)
            assert store.current_state_id == root


def test_storage_checkpoint_stale_state_rejects_without_rewrite_then_repeats(tmp_path):
    path = tmp_path / "graph.ggj"
    with DurableGraph.create(path) as store:
        root = seed(store)
        original = path.read_bytes()
        with pytest.raises(ValueError, match="Stale"):
            store.checkpoint_storage(expected_state=0)
        assert path.read_bytes() == original
        first = store.checkpoint_storage()
        second = store.checkpoint_storage()
        assert first["selected_state"] == root
        assert second["previous_records"] == 1
        assert second["retained_versions"] == 1
        assert store.current_state_id == root


def test_storage_checkpoint_refuses_changed_path_and_preserves_unrelated_file(tmp_path):
    path = tmp_path / "graph.ggj"
    moved = tmp_path / "moved.ggj"
    with DurableGraph.create(path) as store:
        root = seed(store)
        path.rename(moved)
        path.write_bytes(b"unrelated")
        with pytest.raises(ValueError, match="path changed"):
            store.checkpoint_storage()
        assert path.read_bytes() == b"unrelated"
        with pytest.raises(ValueError, match="reopen"):
            store.current_state_id
    with DurableGraph.open(moved) as store:
        assert store.current_state_id == root
        store.checkpoint_storage()


def test_physical_checkpoint_drops_pruned_operation_frames_and_keeps_raw_identity_reservations(tmp_path):
    path = tmp_path / "graph.ggj"
    with DurableGraph.create(path) as store:
        root = seed(store)
        store.create_branch("discard", root)
        store.checkout_branch("discard", root)
        batch = GraphBatch(effective_at=20)
        node = batch.add_node()
        batch.set_node_external_id(node, "retained-allocation")
        store.apply_batch(batch, root, "discarded", "test")
        discarded = store.current_state_id
        store.checkout_branch("main", discarded)
        assert store.prune_branch("discard") == 1
        report = store.checkpoint_storage()
        assert report["retained_versions"] == 1
    with DurableGraph.open(path) as store:
        with pytest.raises(ValueError):
            store.snapshot_at_commit(discarded)
        assert store.allocated_entity_for_external_id("retained-allocation") == ("node", 1)
        batch = GraphBatch()
        batch.add_node()
        result = store.apply_batch(batch, root, "after", "test")
        assert result["added_nodes"] == [2]
        assert store.current_state_id > discarded
