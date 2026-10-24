import time
import pytest
from groggy import Graph, GraphBatch


def times_by_key(metadata):
    return {(row["kind"], row["entity_id"], row["name"]): row["effective_at"] for row in metadata["change_times"]}


def test_per_change_times_override_batch_default_and_creation_inheritance():
    graph = Graph()
    batch = GraphBatch(effective_at=100)
    node = batch.add_node(name="Ada", effective_at=50)
    batch.set_node_attr(node, "age", 42, effective_at=-10)
    before = time.time_ns() // 1000
    result = graph.apply_batch(batch, 0, "timed", "test")
    after = time.time_ns() // 1000
    metadata = graph.publication_metadata(result["state_id"])
    assert before <= metadata["recorded_at"] <= after
    assert metadata["effective_at"] == 100
    times = times_by_key(metadata)
    assert times[("node_added", 0, None)] == 50
    assert times[("node_attr", 0, "entity_type")] == 50
    assert times[("node_attr", 0, "name")] == 50
    assert times[("node_attr", 0, "age")] == -10


def test_writer_bulk_updates_capture_one_clock_and_explicit_bulk_time():
    graph = Graph()
    writer = graph.writer()
    nodes = writer.add_nodes(2)
    metadata = graph.publication_metadata(writer.state_id)
    assert metadata["effective_at"] == metadata["recorded_at"]
    assert all(row["effective_at"] == metadata["recorded_at"] for row in metadata["change_times"])
    writer.set_node_attrs({"score": {nodes[0]: 1, nodes[1]: 2}}, effective_at=123)
    metadata = graph.publication_metadata(writer.state_id)
    assert set(times_by_key(metadata).values()) == {123}
    assert len(metadata["change_times"]) == 2
    assert len(graph.commit_history()) == 2


def test_explicit_edge_removal_controls_time_instead_of_cascade_time():
    graph = Graph()
    writer = graph.writer()
    nodes = writer.add_nodes(2)
    edge = writer.add_edge(*nodes)
    batch = GraphBatch(effective_at=100)
    batch.remove_node(nodes[0], effective_at=20)
    batch.remove_edge(edge, effective_at=30)
    result = writer.apply_batch(batch)
    times = times_by_key(graph.publication_metadata(result["state_id"]))
    assert times[("node_removed", nodes[0], None)] == 20
    assert times[("edge_removed", edge, None)] == 30


@pytest.mark.parametrize("value", [True, 1.5, "yesterday", 2**80])
def test_invalid_timestamp_rejects_before_allocation(value):
    graph = Graph()
    batch = GraphBatch(effective_at=value)
    batch.add_node()
    with pytest.raises((TypeError, OverflowError)):
        graph.apply_batch(batch, 0, "invalid", "test")
    assert graph.add_node() == 0


def test_unknown_or_duplicate_time_overrides_are_not_silently_ignored():
    graph = Graph()
    raw = {"add_nodes": 1, "change_times": [{"kind": "node_attr", "entity": {"added": 0}, "name": "missing", "effective_at": 10}]}
    with pytest.raises(ValueError, match="no staged operation"):
        graph.apply_batch(raw, 0, "invalid", "test")
    raw["change_times"] = [{"kind": "node_added", "entity": {"added": 0}, "effective_at": 10}] * 2
    with pytest.raises(ValueError, match="duplicate time"):
        graph.apply_batch(raw, 0, "invalid", "test")
    assert graph.add_node() == 0


def test_legacy_commit_has_no_automatic_publication_metadata():
    graph = Graph(manual=True)
    graph.add_node()
    state = graph.commit("legacy", "test")
    assert graph.publication_metadata(state) is None
