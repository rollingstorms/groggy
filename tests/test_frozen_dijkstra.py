"""Pinned weighted distances share checked live-kernel rules."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


@pytest.mark.parametrize("directed", [False, True])
def test_captured_weights_and_scope_survive_owner_mutations(directed):
    graph = gr.Graph(directed=directed)
    a, b, c, isolated = graph.add_nodes(4)
    first = graph.add_edge(a, b)
    second = graph.add_edge(b, c)
    excluded = graph.add_edge(a, c)
    graph.set_edge_attr(first, "w", 4)
    graph.set_edge_attr(second, "w", 0.5)
    graph.set_edge_attr(excluded, "w", 0)
    view = graph.read_view().select_edges([first, second])
    graph.set_edge_attr(first, "w", 999)
    graph.remove_node(b)
    del graph
    expected = [(a, 0.0), (b, 4.0), (c, 4.5)]
    assert view.dijkstra_distances(a, weight_attr="w") == expected
    assert view.dijkstra_distances(c, weight_attr="w") == ([(c, 0.0)] if directed else [(a, 4.5), (b, 0.5), (c, 0.0)])
    assert all(type(value) is float for _, value in expected)
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(lambda _: view.dijkstra_distances(a, "w"), range(4))) == [expected] * 4
    with pytest.raises((ValueError, RuntimeError), match="captured selection"):
        view.dijkstra_distances(isolated, "w")


@pytest.mark.parametrize("bad", [None, True, "2", -1, float("nan"), float("inf")])
def test_bad_selected_weights_fail_even_when_unreachable(bad):
    graph = gr.Graph()
    a, b, c = graph.add_nodes(3)
    edge = graph.add_edge(b, c)
    graph.set_edge_attr(edge, "w", bad)
    view = graph.read_view()
    with pytest.raises((ValueError, RuntimeError), match="weight"):
        view.dijkstra_distances(a, "w")
    assert view.dijkstra_distances(a) == [(a, 0.0)]


def test_zero_missing_parallel_weights_and_singleton_preserve_existing_rules():
    graph = gr.Graph()
    a, b, c = graph.add_nodes(3)
    zero = graph.add_edge(a, b)
    parallel = graph.add_edge(a, b)
    graph.add_edge(b, c)
    graph.set_edge_attr(zero, "w", 0)
    graph.set_edge_attr(parallel, "w", 7)
    view = graph.read_view()
    assert view.dijkstra_distances(a, "w") == [(a, 0.0), (b, 0.0), (c, 1.0)]
    assert view.select_nodes([c]).dijkstra_distances(c, "w") == [(c, 0.0)]
    with pytest.raises((ValueError, RuntimeError)):
        graph.read_view().select_nodes([]).dijkstra_distances(a)
