import pytest

from groggy import Graph


def test_python_commit_diff_reports_typed_values_and_topology():
    graph = Graph()
    alice = graph.add_node()
    graph.set_node_attr(alice, "score", 1)
    first = graph.commit("first", "test")

    bob = graph.add_node()
    edge = graph.add_edge(alice, bob)
    graph.set_node_attr(alice, "score", "later")
    second = graph.commit("second", "test")

    diff = graph.diff_commits(first, second)
    assert diff["nodes_added"] == [bob]
    assert diff["edges_added"] == [(edge, alice, bob)]
    assert diff["edge_endpoint_changes"] == []
    score = next(
        change
        for change in diff["attribute_changes"]
        if change["entity_type"] == "node" and change["name"] == "score"
    )
    assert score["old"] == 1
    assert score["new"] == "later"

    reverse = graph.diff_commits(second, first)
    assert reverse["nodes_removed"] == [bob]
    assert reverse["edges_removed"] == [edge]
    with pytest.raises(ValueError, match="not found"):
        graph.diff_commits(first, 999)
