import pytest
from groggy import Graph


def test_writer_publishes_each_scalar_or_bulk_call_as_one_state():
    graph = Graph()
    writer = graph.writer(author="test")
    nodes = writer.add_nodes_with_attrs([{"name": "Ada"}, {}])
    assert len(graph.commit_history()) == 1
    before = graph.snapshot_at_commit(writer.state_id)
    edge = writer.add_edge(*nodes, weight=0.5)
    writer.set_node_attrs({"name": {nodes[0]: "Grace", nodes[1]: "Lin"}, "age": [(nodes[0], 42)]})
    assert len(graph.commit_history()) == 3
    assert not graph.has_uncommitted_changes()
    assert before.node_attr(nodes[0], "name").value == "Ada"
    writer.remove_node(nodes[0])
    assert len(graph.edges) == 0
    assert len(graph.commit_history()) == 4
    assert not graph.contains_edge(edge)


def test_writer_rejects_stale_session_until_explicit_refresh():
    graph = Graph()
    first = graph.writer()
    second = graph.writer()
    assert first.add_node() == 0
    with pytest.raises(ValueError, match="Stale"):
        second.add_node()
    assert second.state_id == 0
    second.refresh()
    assert second.add_node() == 1
    graph.set_node_attr(0, "ordinary", True)
    with pytest.raises(ValueError, match="Stale"):
        second.add_node()
    assert graph.get_node_attr(0, "ordinary") is True


def test_writer_copies_policy_and_enforces_it_for_each_call():
    graph = Graph()
    schema = {"node_labels": {"Entity": {"name": {"type": "text"}}}}
    writer = graph.writer(schema=schema)
    schema.clear()
    with pytest.raises(ValueError, match="name"):
        writer.add_node(labels=["Entity"])
    assert writer.add_node(labels=["Entity"], name="Ada") == 0
    state = writer.state_id
    with pytest.raises(ValueError, match="name"):
        writer.set_node_attr(0, "name", None)
    assert writer.state_id == state
    assert not graph.has_uncommitted_changes()


def test_writer_handles_bulk_edges_kinds_and_rejected_columns_without_partial_writes():
    graph = Graph()
    writer = graph.writer()
    nodes = writer.add_nodes(2)
    edges = writer.add_edges([(nodes[0], nodes[1], {"relationship_type": "KNOWS"}), (nodes[0], nodes[1])])
    assert len(graph.commit_history()) == 2
    writer.set_node_labels(nodes[0], ["Entity"])
    writer.set_edge_relationship_type(edges[1], "KNOWS")
    assert graph.edges_with_relationship_type("KNOWS") == edges
    state = writer.state_id
    with pytest.raises(ValueError):
        writer.set_edge_attrs({"weight": {edges[0]: 2, 999: 3}})
    assert writer.state_id == state
    assert graph.get_edge_attr(edges[0], "weight") is None
    writer.remove_edges(edges)
    assert len(graph.edges) == 0
