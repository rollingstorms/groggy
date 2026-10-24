"""Ordinary Python Graph writes publish validated versions by default."""
import time
import pytest
from groggy import Graph, GraphBatch


def test_default_direct_and_bulk_operations_publish_once_with_compute_time():
    graph = Graph()
    assert graph.automatic_publication
    before = time.time_ns() // 1000
    nodes = graph.add_nodes([{"x": 1}, {"x": 2}, {"x": 3}])
    after = time.time_ns() // 1000
    assert graph.current_state_id == 1
    clock = graph.publication_metadata(1)
    assert before <= clock["recorded_at"] <= after
    assert clock["effective_at"] == clock["recorded_at"]
    assert all(row["effective_at"] == clock["recorded_at"] for row in clock["change_times"])
    edges = graph.add_edges([(nodes[0], nodes[1], {"w": 1}), (nodes[1], nodes[2], {"w": 2})])
    assert graph.current_state_id == 2
    frozen = graph.read_view()
    assert graph.snapshot_at_commit(2).edge_attr(edges[0], "w").value == 1
    graph.set_node_attrs({"x": {nodes[0]: 4, nodes[1]: 5}})
    assert graph.current_state_id == 3
    graph.set_edge_attrs({"w": {edges[0]: 3, edges[1]: 4}})
    assert graph.current_state_id == 4
    graph.remove_node(nodes[1])
    assert graph.current_state_id == 5
    assert len(graph.edge_ids) == 0
    assert frozen.node_attr(nodes[1], "x") == 2
    assert graph.commit("checkpoint", "owner") == 5
    assert len(graph.commit_history()) == 5
    assert not graph.has_uncommitted_changes()


def test_creation_has_initial_properties_in_the_same_state_and_failures_consume_no_ids():
    graph = Graph()
    node = graph.add_node(name="one")
    assert graph.current_state_id == 1
    edge = graph.add_edge(node, node, relation="loop")
    assert graph.current_state_id == 2
    snapshot = graph.snapshot_at_commit(2)
    assert snapshot.node_attr(node, "name").value == "one"
    assert snapshot.edge_attr(edge, "relation").value == "loop"
    state = graph.current_state_id
    with pytest.raises(ValueError):
        graph.add_edges([(node, node), (node, 99)])
    with pytest.raises(ValueError):
        graph.add_nodes([{"name": "valid"}, {"entity_type": "meta"}])
    assert graph.current_state_id == state
    assert len(graph.node_ids) == len(graph.edge_ids) == 1
    assert graph.add_node() == 1
    assert graph.add_edge(node, node) == 1
    state = graph.current_state_id
    assert graph.add_nodes(0) == []
    assert graph.add_edges([]) == []
    graph.set_node_attrs({})
    graph.remove_nodes([])
    assert graph.current_state_id == state


def test_ordinary_writes_invalidate_optimistic_sessions_but_batch_times_remain_explicit():
    graph = Graph()
    writer = graph.writer()
    graph.add_node()
    with pytest.raises(ValueError, match="Stale"):
        writer.add_node()
    writer.refresh()
    writer.set_node_attr(0, "x", 1, effective_at=-123)
    clock = graph.publication_metadata(graph.current_state_id)
    assert all(row["effective_at"] == -123 for row in clock["change_times"])
    batch = GraphBatch(effective_at=456)
    batch.set_node_attr(0, "x", 2, effective_at=789)
    state = graph.current_state_id
    graph.apply_batch(batch, state, "explicit", "test")
    assert graph.current_state_id == state + 1
    assert graph.publication_metadata(state + 1)["change_times"][0]["effective_at"] == 789
    graph.set_node_attr(0, "x", 3)
    assert not graph.has_uncommitted_changes()


def test_identity_upserts_and_graph_copy_publish_one_state():
    graph = Graph()
    rows = [{"external_id": "a", "x": 1}, {"external_id": "b", "x": 2}]
    assert graph.upsert_nodes(rows) == [0, 1]
    assert graph.current_state_id == 1
    graph.upsert_nodes(rows)
    assert graph.current_state_id == 1
    graph.set_node_external_id(0, "renamed")
    assert graph.current_state_id == 2
    source = Graph()
    source.add_nodes(2)
    source.add_edge(0, 1, w=3)
    target = Graph()
    target.add_graph(source)
    assert target.current_state_id == 1
    assert target.get_edge_attr(0, "w") == 3


def test_manual_compatibility_and_active_import_have_explicit_boundaries(tmp_path):
    manual = Graph(manual=True)
    assert not manual.automatic_publication
    manual.add_node(x=1)
    assert manual.current_state_id == 0
    assert manual.has_uncommitted_changes()
    with pytest.raises(ValueError, match="uncommitted"):
        manual.writer().add_node()
    manual.commit("legacy", "test")
    path = tmp_path / "active.json"
    manual.save_to_path(path)
    imported = Graph.load_from_path(path)
    assert imported.automatic_publication
    assert not imported.has_uncommitted_changes()
    assert imported.current_state_id == 1
    assert imported.checkpoints() == []
    assert imported.publication_metadata(1) is not None
    imported.set_node_attr(0, "x", 2)
    assert imported.current_state_id == 2
    legacy_import = Graph.load_from_path(path, manual=True)
    assert not legacy_import.automatic_publication
    assert legacy_import.has_uncommitted_changes()


def test_table_graph_conversion_returns_a_clean_publishing_graph():
    source = Graph()
    source.add_nodes(2)
    source.add_edge(0, 1, w=3)
    copied = source.table().to_graph()
    assert copied.automatic_publication
    assert not copied.has_uncommitted_changes()
    assert copied.current_state_id == 1
    assert copied.get_edge_attr(0, "w") == 3
    copied.set_node_attr(0, "x", 4)
    assert copied.current_state_id == 2
