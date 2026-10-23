"""Captured PageRank retains selected topology and exposes finite-budget status."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


def test_frozen_personalization_scope_and_owned_result():
    graph = gr.Graph(directed=True)
    a, b, c = graph.add_nodes(3)
    edges = [graph.add_edge(a, b), graph.add_edge(a, b), graph.add_edge(a, c)]
    graph.add_edge(b, a)  # Excluded.
    for node, value in [(a, 1), (b, 0), (c, 0)]:
        graph.set_node_attr(node, "p", value)
    view = graph.read_view().select_edges(edges)
    graph.set_node_attr(a, "p", 0)
    del graph
    result = view.pagerank(damping=0.5, max_iter=1000, tolerance=1e-14, personalization_attr="p")
    assert isinstance(result, gr.PageRankResult)
    assert result.converged
    assert dict(result.scores) == pytest.approx({a: 2/3, b: 2/9, c: 1/9}, abs=1e-13)
    assert result.max_update <= 1e-14
    assert 0 < result.iterations <= 1000
    copy = result.scores
    copy.clear()
    assert len(result.scores) == 3
    with pytest.raises(AttributeError):
        result.converged = False
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: view.pagerank(max_iter=1000).scores, range(4)))
    assert results == [results[0]] * 4
    assert all("pagerank" not in view.node_attribute_keys(node) for node in [a, b, c])


def test_budget_empty_isolates_and_no_property_writes():
    graph = gr.Graph(directed=True)
    empty = graph.read_view().pagerank()
    assert empty.scores == []
    assert empty.converged and empty.iterations == 0 and empty.max_update == 0
    a, b = graph.add_nodes(2)
    assert dict(graph.read_view().pagerank().scores) == {a: 0.5, b: 0.5}
    graph.add_edge(a, b)
    limited = graph.read_view().pagerank(max_iter=1, tolerance=1e-14)
    assert not limited.converged and limited.iterations == 1
    assert sum(score for _, score in limited.scores) == pytest.approx(1)
    assert all("pagerank" not in graph.read_view().node_attribute_keys(node) for node in [a, b])


@pytest.mark.parametrize("value", [None, True, "text", [1.0], -1, float("nan"), float("inf")])
def test_invalid_personalization_fails_even_on_isolates(value):
    graph = gr.Graph()
    node = graph.add_node()
    graph.set_node_attr(node, "p", value)
    with pytest.raises((ValueError, RuntimeError), match="personalization"):
        graph.read_view().pagerank(personalization_attr="p")


@pytest.mark.parametrize("kwargs", [
    {"damping": -0.1}, {"damping": 1}, {"damping": float("nan")},
    {"max_iter": 0}, {"tolerance": 0}, {"tolerance": float("inf")},
])
def test_invalid_parameters_are_checked_for_empty_graphs(kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        gr.Graph().read_view().pagerank(**kwargs)


def test_missing_personalization_defaults_to_one_and_zero_total_fails():
    graph = gr.Graph()
    a, b = graph.add_nodes(2)
    graph.set_node_attr(a, "p", 0)
    result = graph.read_view().pagerank(damping=0, personalization_attr="p")
    assert dict(result.scores) == {a: 0, b: 1}
    graph.set_node_attr(b, "p", 0)
    with pytest.raises((ValueError, RuntimeError), match="positive total"):
        graph.read_view().pagerank(personalization_attr="p")
