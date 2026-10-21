import pytest
from groggy import Graph, algorithms
from groggy.pipeline import apply


def test_undirected_reverse_edges_and_directed_dangling_scores():
    for directed, expected in [(False, [0.5, 0.5]), (True, [1 / 2.85, 1.85 / 2.85])]:
        graph = Graph(directed=directed)
        a, b = graph.add_node(), graph.add_node()
        graph.add_edge(a, b)
        graph.induced_subgraph([a, b]).apply(
            algorithms.centrality.pagerank(tolerance=1e-12, max_iter=1000)
        )
        got = [graph.get_node_attr(n, "pagerank") for n in [a, b]]
        assert got == pytest.approx(expected, abs=1e-7)


def test_personalization_controls_dangling_distribution_and_returns_f64_scores():
    graph = Graph(directed=True)
    nodes = [graph.add_node() for _ in range(3)]
    for node, weight in zip(nodes, [1, 3, 0]):
        graph.set_node_attr(node, "p", weight)
    result, profile = apply(
        graph.induced_subgraph(nodes),
        algorithms.centrality.pagerank(personalization_attr="p", tolerance=1e-13),
        persist=False,
        return_profile=True,
    )
    scores = profile["outputs"]["centrality.pagerank.scores"]
    assert [pair[0] for pair in scores] == nodes
    assert [pair[1] for pair in scores] == pytest.approx([0.25, 0.75, 0.0], abs=1e-14)
    assert profile["stats"]["pr.converged"] == 1.0
    assert "pagerank" not in result.nodes.attribute_names()
    assert all(graph.get_node_attr(n, "pagerank") is None for n in nodes)


@pytest.mark.parametrize("value", [-1, True, None, "bad", float("nan"), float("inf")])
def test_invalid_personalization_preserves_existing_output(value):
    graph = Graph()
    a, b = graph.add_node(), graph.add_node()
    graph.set_node_attr(a, "p", value)
    graph.set_node_attr(a, "pagerank", 7)
    with pytest.raises(RuntimeError, match="personalization"):
        graph.induced_subgraph([a, b]).apply(
            algorithms.centrality.pagerank(personalization_attr="p")
        )
    assert graph.get_node_attr(a, "pagerank") == 7


@pytest.mark.parametrize("kwargs", [
    {"max_iter": 0}, {"max_iter": -1}, {"damping": -0.1}, {"damping": 1.0},
    {"damping": float("nan")}, {"damping": float("inf")},
    {"tolerance": 0.0}, {"tolerance": -1.0},
    {"tolerance": float("nan")}, {"tolerance": float("inf")},
])
def test_invalid_controls_reject(kwargs):
    graph = Graph()
    node = graph.add_node()
    with pytest.raises(RuntimeError):
        graph.induced_subgraph([node]).apply(algorithms.centrality.pagerank(**kwargs))


def test_exhausted_budget_is_explicitly_an_approximation():
    graph = Graph(directed=True)
    a, b = graph.add_node(), graph.add_node()
    graph.add_edge(a, b)
    _, profile = apply(
        graph.induced_subgraph([a, b]),
        algorithms.centrality.pagerank(max_iter=1, tolerance=1e-12),
        persist=False, return_profile=True,
    )
    assert profile["stats"]["pr.converged"] == 0.0
    assert profile["stats"]["pr.count.iteration"] == 1.0
    scores = profile["outputs"]["centrality.pagerank.scores"]
    assert sum(value for _, value in scores) == pytest.approx(1.0)
