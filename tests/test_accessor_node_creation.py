"""Node append/extend marshal completely, then use one native creation operation."""
import pytest
from groggy import Graph


def test_node_accessor_append_and_extend_each_publish_one_complete_version():
    graph = Graph()
    node = graph.nodes.append({"name": "Ada", "score": 1.25})
    assert node == 0 and graph.current_state_id == 1
    snapshot = graph.snapshot_at_commit(1)
    assert snapshot.node_attr(node, "name").value == "Ada"
    assert snapshot.node_attr(node, "score").value == 1.25
    ids = graph.nodes.extend([{"name": "Grace", "score": 2.5}, {"name": "Katherine"}])
    assert ids == [1, 2] and graph.current_state_id == 2
    assert len(graph.commit_history()) == 2
    assert not graph.has_uncommitted_changes()
    assert graph.get_node_attr(2, "name") == "Katherine"
    assert graph.nodes.extend([]) == [] and graph.current_state_id == 2


@pytest.mark.parametrize("rows", [
    [{"ok": 1}, object()],
    [{"ok": 1}, {1: "bad key"}],
    [{"ok": 1}, {"invalid": object()}],
    [{"ok": 1}, {"entity_type": "meta"}],
    [{"ok": 1}, {"labels": [1]}],
    [{"ok": 1}, {"external_id": "managed"}],
])
def test_node_accessor_extend_rejects_complete_input_before_allocating(rows):
    graph = Graph()
    before = graph.current_state_id
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        graph.nodes.extend(rows)
    assert graph.current_state_id == before and len(graph.nodes) == 0
    assert not graph.has_uncommitted_changes()
    assert graph.nodes.append({"valid": True}) == 0


@pytest.mark.parametrize("attrs", [{"invalid": object()}, {1: "bad"}, {"entity_type": "meta"}])
def test_node_accessor_append_rejects_before_allocating(attrs):
    graph = Graph()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        graph.nodes.append(attrs)
    assert graph.current_state_id == 0 and len(graph.nodes) == 0
    assert graph.nodes.append({}) == 0


def test_node_accessor_creation_honors_explicit_manual_policy():
    graph = Graph(manual=True)
    assert graph.nodes.append({"v": 1}) == 0
    assert graph.nodes.extend([{"v": 2}, {"v": 3}]) == [1, 2]
    assert graph.current_state_id == 0 and graph.has_uncommitted_changes()
    graph.commit("manual nodes", "test")
    assert graph.current_state_id == 1
    assert graph.snapshot_at_commit(1).node_attr(2, "v").value == 3
