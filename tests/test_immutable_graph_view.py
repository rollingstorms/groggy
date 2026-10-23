"""Immutable state and edge-scoped selections remain independent of live mutations."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


def test_read_view_freezes_dirty_properties_and_nested_scope():
    graph = gr.Graph()
    a, b, isolated = graph.add_nodes(3)
    first = graph.add_edge(a, b)
    second = graph.add_edge(a, b)
    loop = graph.add_edge(a, a)
    graph.set_node_attr(a, "value", 2**63 - 1)
    graph.set_edge_attr(first, "weight", 4)
    root = graph.read_view()
    assert isinstance(root, gr.GraphView)
    selected = root.select_edges([first, loop, first]).select_nodes([a, b, b])
    assert selected.edge_ids == [first, loop]
    assert selected.neighbors(a) == [a, b]
    assert not selected.contains_edge(second)
    graph.set_node_attr(a, "value", None)
    graph.remove_node(a)
    assert selected.node_attr(a, "value") == 2**63 - 1
    assert selected.edge_attr(first, "weight") == 4
    assert selected.edge_endpoints(first) == (a, b)
    assert selected.select_nodes([a, b]).neighbors(a) == [a, b]
    assert root.degree(isolated) == 0
    assert not graph.read_view().contains_node(a)
    with pytest.raises((ValueError, RuntimeError)):
        selected.select_nodes([isolated])
    with pytest.raises((ValueError, RuntimeError)):
        selected.edge_attr(second, "weight")


def test_directed_instance_degrees_and_loops():
    graph = gr.Graph(directed=True)
    a, b, c = graph.add_nodes(3)
    graph.add_edge(a, b)
    graph.add_edge(a, b)
    graph.add_edge(c, a)
    graph.add_edge(a, a)
    view = graph.read_view()
    assert view.is_directed
    assert view.neighbors(a) == [a, b, b]
    assert view.degree(a) == 4
    assert view.out_degree(a) == 3
    assert view.in_degree(a) == 2
    selected = view.select_nodes([a, b])
    assert selected.in_degree(a) == 1
    assert selected.in_degree(b) == 2
    assert selected.neighbors(b) == []


def test_explicit_pin_keeps_legacy_edge_scope_and_outlives_owner():
    graph = gr.Graph()
    a, b = graph.add_nodes(2)
    first = graph.add_edge(a, b)
    graph.add_edge(a, b)
    legacy = graph.subgraph_from_edges([first])
    frozen = legacy.pin_view()
    assert frozen.edge_ids == [first]
    assert frozen.select_nodes([a, b]).edge_ids == [first]
    graph.remove_edge(first)
    with pytest.raises((ValueError, RuntimeError)):
        legacy.pin_view()
    del legacy, graph
    assert frozen.neighbors(a) == [b]
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(lambda _: frozen.copy().select_nodes([a, b]).degree(a), range(4))) == [1] * 4


def test_empty_single_node_and_null_versus_absent():
    graph = gr.Graph()
    assert graph.read_view().node_ids == []
    a = graph.add_node()
    graph.set_node_attr(a, "value", None)
    view = graph.read_view()
    assert view.node_count() == 1
    assert view.edge_count() == view.degree(a) == 0
    assert view.node_attr(a, "value") is None
    assert view.node_attr(a, "missing") is None
    assert view.has_node_attribute(a, "value")
    assert not view.has_node_attribute(a, "missing")
    assert "value" in view.node_attribute_keys(a)
    assert view.select_nodes([]).node_count() == 0
    with pytest.raises((ValueError, RuntimeError)):
        view.node_attr(a + 1, "value")
    assert not hasattr(view, "set_node_attr")
