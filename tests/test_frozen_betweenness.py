"""Captured Brandes shares kernels and preserves parallel shortest-path counts."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest
from groggy import algorithms
from groggy.pipeline import apply


def test_captured_parallel_path_counts_lengths_and_isolate_normalization():
    graph = gr.Graph(directed=True)
    a, b, c, d, isolate = graph.add_nodes(5)
    edges = [graph.add_edge(a, b), graph.add_edge(a, b), graph.add_edge(a, b),
             graph.add_edge(a, c), graph.add_edge(b, d), graph.add_edge(c, d)]
    graph.set_edge_attr(edges[2], "w", 10)  # Exclude this instance only for weighted paths.
    view = graph.read_view()
    graph.add_edge(a, d)
    graph.set_edge_attr(edges[0], "w", 0)
    del graph
    expected = {a: 0, b: 2/3, c: 1/3, d: 0, isolate: 0}
    assert dict(view.betweenness(normalized=False, weight_attr="w")) == pytest.approx(expected)
    assert dict(view.betweenness(weight_attr="w")) == pytest.approx({node: value/12 for node, value in expected.items()})
    assert dict(view.betweenness(normalized=False)) == pytest.approx({a: 0, b: 0.75, c: 0.25, d: 0, isolate: 0})
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: view.betweenness(weight_attr="w"), range(4)))
    assert results == [results[0]] * 4
    results[0].clear()
    assert len(view.betweenness()) == 5
    assert all("betweenness" not in view.node_attribute_keys(node) for node in [a, b, c, d, isolate])


@pytest.mark.parametrize("value", [None, True, "bad", [1.0], 0, -1, float("nan"), float("inf")])
def test_invalid_selected_lengths_fail_even_on_loops(value):
    graph = gr.Graph()
    node = graph.add_node()
    edge = graph.add_edge(node, node)
    graph.set_edge_attr(edge, "w", value)
    with pytest.raises((ValueError, RuntimeError)):
        graph.read_view().betweenness(weight_attr="w")
    assert graph.read_view().betweenness() == [(node, 0)]


def test_empty_directed_undirected_and_transient_live_output():
    assert gr.Graph().read_view().betweenness() == []
    for directed in [False, True]:
        graph = gr.Graph(directed=directed)
        a, b, c = graph.add_nodes(3)
        graph.add_edge(a, b)
        graph.add_edge(b, c)
        expected = [(a, 0), (b, 0.5 if directed else 1), (c, 0)]
        assert graph.read_view().betweenness() == expected
        _, profile = apply(
            graph.induced_subgraph([a, b, c]),
            algorithms.centrality.betweenness(),
            persist=False,
            return_profile=True,
        )
        assert profile["outputs"]["centrality.betweenness.scores"] == expected
        assert all("betweenness" not in graph.read_view().node_attribute_keys(node) for node in [a, b, c])
