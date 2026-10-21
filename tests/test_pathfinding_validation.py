import pytest

from groggy import Graph


@pytest.mark.parametrize("operation", ["bfs", "dfs", "shortest_path"])
def test_removed_or_unknown_starts_reject_instead_of_returning_a_phantom_node(operation):
    graph = Graph()
    removed = graph.add_node()
    graph.remove_node(removed)
    for node in (removed, 999):
        with pytest.raises(ValueError):
            if operation == "shortest_path":
                graph.shortest_path(node, node)
            else:
                getattr(graph, operation)(node)


@pytest.mark.parametrize("weight", [-1, -0.5, float("nan"), float("inf"), "bad", None, True])
def test_bad_weights_reject_before_shortest_path_early_exit(weight):
    graph = Graph(directed=True)
    a, b, c = [graph.add_node() for _ in range(3)]
    edge = graph.add_edge(b, c, weight=weight)
    with pytest.raises(ValueError, match="weight"):
        graph.shortest_path(a, a, weight_attribute="weight")
    graph.remove_edge(edge)
    assert list(graph.shortest_path(a, a, weight_attribute="weight").node_ids) == [a]


def test_compact_integer_weights_and_parallel_edges_keep_selected_edge_identity():
    graph = Graph()
    a, b, c = [graph.add_node() for _ in range(3)]
    graph.add_edge(a, b, weight=9)
    cheap = graph.add_edge(a, b, weight=3)
    zero = graph.add_edge(b, c, weight=0)
    path = graph.shortest_path(c, a, weight_attribute="weight")
    assert set(path.edge_ids) == {cheap, zero}
    missing = graph.add_edge(a, c)
    assert list(graph.shortest_path(a, c, weight_attribute="weight").edge_ids) == [missing]
