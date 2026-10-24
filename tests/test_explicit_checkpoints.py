"""Explicit checkpoints label published states without manufacturing mutations."""
import pytest
from groggy import DurableGraph, Graph


def test_writer_checkpoints_preserve_state_versions_clocks_and_retained_views():
    graph = Graph()
    writer = graph.writer(author="owner")
    node = writer.add_node(x=1)
    state = writer.state_id
    frozen = graph.read_view()
    before = graph.publication_metadata(state)
    assert writer.checkpoint("first") == state
    assert graph.commit("second", "owner") == state
    assert graph.checkpoint("third", "owner", expected_state=state) == state
    assert graph.current_state_id == state
    assert writer.state_id == state
    assert len(graph.commit_history()) == 1
    assert graph.publication_metadata(state) == before
    assert graph.checkpoints()[0]["message"] == "first"
    assert [r["state_id"] for r in graph.checkpoints()] == [state] * 3
    records = graph.checkpoints()
    records[0]["message"] = "edited copy"
    assert graph.checkpoints()[0]["message"] == "first"
    writer.set_node_attr(node, "x", 2)
    assert writer.state_id == state + 1
    assert frozen.node_attr(node, "x") == 1
    assert graph.snapshot_at_commit(state).node_attr(node, "x").value == 1


def test_empty_state_can_be_checkpointed_and_stale_writers_are_rejected():
    graph = Graph()
    assert graph.commit("empty", "owner") == 0
    writer = graph.writer()
    other = graph.writer()
    other.add_node()
    count = len(graph.checkpoints())
    with pytest.raises(ValueError, match="Stale"):
        writer.checkpoint("stale")
    assert len(graph.checkpoints()) == count
    assert graph.current_state_id == 1
    writer.refresh()
    assert writer.checkpoint("fresh") == 1


def test_legacy_pending_writes_require_publication_before_a_clean_checkpoint():
    graph = Graph(manual=True)
    graph.add_node(x=1)
    with pytest.raises(ValueError, match="uncommitted"):
        graph.checkpoint("pending", "owner")
    assert graph.checkpoints() == []
    state = graph.commit("legacy compatibility", "owner")
    assert graph.checkpoints()[0]["state_id"] == state
    assert graph.commit("label existing state", "owner") == state
    assert len(graph.commit_history()) == 1


def test_checkpoint_respects_a_writers_additional_schema():
    graph = Graph()
    graph.writer().add_node(labels=["Event"])
    writer = graph.writer(schema={"node_labels": {"Event": {"required": {"type": "int"}}}})
    with pytest.raises(ValueError, match="required"):
        writer.checkpoint("invalid under policy")
    assert graph.checkpoints() == []


def test_durable_checkpoints_restore_metadata_without_advancing_the_head(tmp_path):
    path = tmp_path / "graph.ggj"
    backup = tmp_path / "backup.ggj"
    with DurableGraph.create(path) as graph:
        assert graph.commit("empty", "owner") == 0
        writer = graph.writer(author="owner")
        node = writer.add_node(x=1)
        state = writer.state_id
        metadata = graph.publication_metadata(state)
        assert writer.checkpoint("retained") == state
        assert graph.commit("second label", "owner") == state
        records = graph.checkpoints()
        before = path.stat().st_size
        with pytest.raises(ValueError, match="Stale"):
            graph.checkpoint("bad", "owner", expected_state=state + 1)
        assert path.stat().st_size == before
        assert len(graph.commit_history()) == 1
        graph.backup(backup)
    for source in [path, backup]:
        with DurableGraph.open(source) as graph:
            assert graph.current_state_id == state
            assert graph.checkpoints() == records
            assert graph.publication_metadata(state) == metadata
            assert graph.get_node_attr(node, "x") == 1
            assert graph.snapshot_at_commit(state).node_attr(node, "x").value == 1
