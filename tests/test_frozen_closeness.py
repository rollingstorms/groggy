"""Frozen closeness shares the checked live distance/scoring implementation."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


def test_scope_frozen_lengths_direction_and_result_ownership():
    graph = gr.Graph(directed=True)
    a, b, c = graph.add_nodes(3)
    first = graph.add_edge(a, b)
    second = graph.add_edge(b, c)
    graph.add_edge(a, c)  # Excluded shortcut.
    graph.set_edge_attr(first, "w", 2)
    view = graph.read_view().select_edges([first, second])
    graph.set_edge_attr(first, "w", 0)
    del graph
    assert dict(view.closeness(weight_attr="w")) == pytest.approx({a: 5/6, b: 1, c: 0})
    assert dict(view.closeness(weight_attr="w", direction="incoming")) == pytest.approx({a: 0, b: 0.5, c: 4/3})
    assert dict(view.closeness(harmonic=False, weight_attr="w", normalized=True)) == pytest.approx({a: 2/5, b: 0.5, c: 0})
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: view.closeness(), range(4)))
    assert results == [[(a, 1.5), (b, 1), (c, 0)]] * 4
    results[0].clear()
    assert len(view.closeness()) == 3
    assert all("closeness" not in view.node_attribute_keys(node) for node in [a, b, c])
    with pytest.raises(ValueError, match="direction"):
        view.closeness(direction="sideways")


@pytest.mark.parametrize("value", [None, True, "bad", [1.0], 0, -1, float("nan"), float("inf")])
def test_invalid_selected_lengths_fail(value):
    graph = gr.Graph()
    a, b = graph.add_nodes(2)
    edge = graph.add_edge(a, b)
    graph.set_edge_attr(edge, "w", value)
    with pytest.raises((ValueError, RuntimeError)):
        graph.read_view().closeness(weight_attr="w")
    # With no weight attribute requested, stored lengths do not affect hop distance.
    assert graph.read_view().closeness() == [(a, 1), (b, 1)]


def test_empty_isolates_and_parallel_minimum_lengths():
    graph = gr.Graph()
    assert graph.read_view().closeness() == []
    a, b, isolate = graph.add_nodes(3)
    first = graph.add_edge(a, b)
    second = graph.add_edge(a, b)
    graph.set_edge_attrs({"w": {first: 2, second: 5}})
    result = graph.read_view().closeness(weight_attr="w")
    assert result == [(a, 0.5), (b, 0.5), (isolate, 0)]
    assert all("closeness" not in graph.read_view().node_attribute_keys(node) for node in [a, b, isolate])
