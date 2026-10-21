import pytest

from groggy import Graph, algorithms


def fixture():
    graph = Graph()
    nodes = [graph.add_node() for _ in range(3)]
    graph.set_node_attr(nodes[2], "start", True)
    graph.set_node_attr(nodes[0], "goal", True)
    for a, b, w in [(0, 2, 9), (0, 1, 2), (1, 2, 2), (0, 1, 8)]:
        graph.add_edge(nodes[a], nodes[b], w=w)
    return graph, nodes


def execute(graph, nodes, heuristic=None):
    return graph.induced_subgraph(nodes).apply(
        algorithms.pathfinding.astar(
            start_attr="start", goal_attr="goal", weight_attr="w",
            heuristic_attr=heuristic, output_attr="order",
        )
    )


def test_astar_reverse_parallel_weights_with_consistent_integer_heuristic():
    graph, nodes = fixture()
    for node, h in zip(nodes, [0, 2, 4]):
        graph.set_node_attr(node, "h", h)
    execute(graph, nodes, "h")
    assert [graph.get_node_attr(n, "order") for n in nodes] == [2, 1, 0]


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), None, True, "bad", 100])
def test_astar_rejects_invalid_or_inconsistent_heuristics_without_writing(bad):
    graph, nodes = fixture()
    graph.set_node_attr(nodes[1], "h", bad)
    with pytest.raises(Exception, match="heuristic"):
        execute(graph, nodes, "h")
    assert all(graph.get_node_attr(n, "order") is None for n in nodes)


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), None, True, "bad"])
def test_astar_uses_the_checked_weight_contract(bad):
    graph, nodes = fixture()
    graph.add_edge(nodes[0], nodes[1], w=bad)
    with pytest.raises(Exception, match="weight"):
        execute(graph, nodes)
    assert all(graph.get_node_attr(n, "order") is None for n in nodes)
