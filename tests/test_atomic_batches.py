import pytest

from groggy import Graph, GraphBatch


def test_python_batch_stages_new_entities_and_publishes_one_state():
    graph = Graph()
    batch = GraphBatch()
    alice = batch.add_node(name="Alice")
    bob = batch.add_node(name="Bob")
    batch.add_edge(alice, bob, weight=1.5)

    result = graph.apply_batch(batch, graph.current_state_id, "people", "test")
    assert result["added_nodes"] == [0, 1]
    assert result["added_edges"] == [0]
    assert graph.current_state_id == result["state_id"]
    assert len(graph.commit_history()) == 1
    assert graph.get_node_attr(0, "name") == "Alice"
    assert graph.get_edge_attr(0, "weight") == 1.5

    previous = graph.snapshot_at_commit(result["state_id"])
    update = GraphBatch()
    update.set_node_attr(0, "name", "Alicia")
    graph.apply_batch(update, graph.current_state_id, "rename", "test")
    assert previous.node_attr(0, "name").value == "Alice"
    assert graph.get_node_attr(0, "name") == "Alicia"


def test_python_batch_collects_semantic_errors_without_partial_write():
    graph = Graph()
    batch = GraphBatch()
    new_node = batch.add_node(name="orphan")
    batch.add_edge(new_node, 999)
    batch.set_node_attr(1000, "score", 1)

    with pytest.raises(ValueError) as error:
        graph.apply_batch(batch, 0, "bad", "test")
    assert "add_edges[0]" in str(error.value)
    assert "node_attrs.score[0]" in str(error.value)
    assert graph.current_state_id == 0
    assert graph.add_node() == 0


def test_python_batch_rejects_handle_from_another_batch():
    first = GraphBatch()
    handle = first.add_node()
    second = GraphBatch()
    second.add_edge(handle, 0)
    with pytest.raises(ValueError, match="another GraphBatch"):
        second.to_dict()
