import pytest

from groggy import Graph, algorithms


def execute(graph, nodes, weighted=True, normalized=False):
    graph.induced_subgraph(nodes).apply(
        algorithms.centrality.betweenness(
            normalized=normalized, weight_attr="w" if weighted else None,
            output_attr="score",
        )
    )


@pytest.mark.parametrize("normalized", [False, True])
def test_weighted_betweenness_counts_equal_parallel_edge_paths(normalized):
    graph = Graph(directed=True)
    nodes = [graph.add_node() for _ in range(4)]
    for u, v, w in [(0, 1, 1), (0, 1, 1), (1, 3, 1), (0, 2, 1), (2, 3, 1), (0, 2, 8)]:
        graph.add_edge(nodes[u], nodes[v], w=w)
    execute(graph, nodes, normalized=normalized)
    scale = 1 / 6 if normalized else 1
    assert graph.get_node_attr(nodes[1], "score") == pytest.approx(scale * 2 / 3)
    assert graph.get_node_attr(nodes[2], "score") == pytest.approx(scale / 3)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_path_scores_respect_direction_and_missing_unit_weights(directed, normalized):
    graph = Graph(directed=directed)
    nodes = [graph.add_node() for _ in range(3)]
    graph.add_edge(nodes[2], nodes[1], w=2)
    graph.add_edge(nodes[1], nodes[0])
    execute(graph, nodes, normalized=normalized)
    expected = 0.5 if directed and normalized else 1
    assert graph.get_node_attr(nodes[1], "score") == pytest.approx(expected)


@pytest.mark.parametrize("bad", [0, -1, None, True, "bad", float("nan"), float("inf")])
def test_invalid_betweenness_weights_do_not_replace_previous_scores(bad):
    graph = Graph()
    nodes = [graph.add_node() for _ in range(3)]
    edge = graph.add_edge(nodes[0], nodes[1], w=1)
    graph.add_edge(nodes[1], nodes[2], w=1)
    execute(graph, nodes)
    before = [graph.get_node_attr(n, "score") for n in nodes]
    graph.set_edge_attr(edge, "w", bad)
    with pytest.raises(Exception, match="weight"):
        execute(graph, nodes)
    assert [graph.get_node_attr(n, "score") for n in nodes] == before
