"""Persistent scores retain the same f64 values as owned algorithm outputs."""
import pytest

from groggy import AttrValue, Graph, algorithms
from groggy.pipeline import apply


@pytest.mark.parametrize("name", ["dijkstra", "closeness", "betweenness", "pagerank"])
def test_persistent_scores_equal_owned_output_without_narrowing(name):
    graph = Graph(directed=True)
    nodes = graph.add_nodes(4)
    graph.set_node_attr(nodes[0], "source", True)
    for u, v, w in [(0, 1, 1 + 2**-40), (1, 2, 2), (0, 2, 4), (2, 3, 1)]:
        graph.add_edge(nodes[u], nodes[v], w=AttrValue.float64(w))
    choices = {
        "dijkstra": (
            algorithms.pathfinding.dijkstra("source", "w", "result"),
            "pathfinding.dijkstra.distances",
        ),
        "closeness": (
            algorithms.centrality.closeness("w", "result"),
            "centrality.closeness.scores",
        ),
        "betweenness": (
            algorithms.centrality.betweenness(normalized=True, output_attr="result"),
            "centrality.betweenness.scores",
        ),
        "pagerank": (
            algorithms.centrality.pagerank(
                tolerance=1e-13, max_iter=200, output_attr="result"
            ),
            "centrality.pagerank.scores",
        ),
    }
    algorithm, key = choices[name]
    subgraph = graph.induced_subgraph(nodes)
    _, profile = apply(subgraph, algorithm, persist=False, return_profile=True)
    assert all(graph.get_node_attr(node, "result") is None for node in nodes)
    subgraph.apply(algorithm)
    for node, score in profile["outputs"][key]:
        assert graph.get_node_attr(node, "result") == score
