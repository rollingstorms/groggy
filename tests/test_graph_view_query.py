"""Shared live/frozen filtering honors captured values and nested edge scope."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


@pytest.mark.parametrize("directed", [False, True])
def test_legacy_and_frozen_queries_cannot_expand_parent_membership(directed):
    graph = gr.Graph(directed=directed)
    a, b, outside = graph.add_nodes(3)
    first = graph.add_edge(a, b)
    excluded = graph.add_edge(a, b)
    graph.add_edge(b, outside)
    for node in [a, b, outside]:
        graph.set_node_attr(node, "keep", True)
    for edge in graph.read_view().edge_ids:
        graph.set_edge_attr(edge, "keep", True)
    legacy = graph.subgraph_from_edges([first])
    frozen = legacy.pin_view()
    for parent in [legacy, frozen]:
        child = parent.filter_nodes("keep == true")
        assert set(child.node_ids) == {a, b}
        assert list(child.edge_ids) == [first]
        assert list(child.filter_edges("keep == true").edge_ids) == [first]
        assert list(parent.filter_nodes(gr.NodeFilter.degree_range(1, 1)).edge_ids) == [first]
        assert parent.filter_nodes(gr.NodeFilter.has_neighbor(outside)).node_count() == 0
        assert parent.filter_edges(gr.EdgeFilter.connects_nodes(b, a)).edge_count() == int(not directed)
    graph.set_node_attr(a, "keep", False)
    assert frozen.filter_nodes("keep == true").node_ids == [a, b]
    assert list(legacy.filter_nodes("keep == true").node_ids) == [b]
    graph.remove_edge(first)
    with pytest.raises((ValueError, RuntimeError)):
        legacy.filter_nodes("keep == true")
    del legacy, graph
    assert frozen.filter_edges("keep == true").edge_ids == [first]
    assert excluded not in frozen.edge_ids


def test_exact_large_integer_ranges_and_null_presence_match_live_queries():
    graph = gr.Graph()
    high, low, null, absent = graph.add_nodes(4)
    graph.set_node_attr(high, "value", 2**63 - 1)
    graph.set_node_attr(low, "value", 2**63 - 2)
    graph.set_node_attr(null, "value", None)
    frozen = graph.read_view()
    for parent in [graph, frozen]:
        assert set(parent.filter_nodes("value > 9223372036854775806").node_ids) == {high}
        assert set(parent.filter_nodes(gr.NodeFilter.has_attribute("value")).node_ids) == {high, low, null}
        assert set(parent.filter_nodes(gr.NodeFilter.attribute_equals("value", None)).node_ids) == {null}
        assert set(parent.filter_nodes(gr.NodeFilter.not_filter(gr.NodeFilter.has_attribute("value"))).node_ids) == {absent}
    assert frozen.filter_nodes(gr.NodeFilter.and_filters([])).node_ids == []
    assert frozen.filter_nodes(gr.NodeFilter.or_filters([])).node_ids == []
    assert frozen.filter_nodes(gr.NodeFilter.node_set([high, high, 999999])).node_ids == [high]
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(lambda _: frozen.filter_nodes("value > 9223372036854775806").node_ids, range(4))) == [[high]] * 4


def test_incident_neighbor_predicate_distinguishes_a_self_loop_from_any_incident_edge():
    graph = gr.Graph(directed=True)
    a, b = graph.add_nodes(2)
    graph.add_edge(a, b)
    graph.add_edge(b, b)
    for parent in [graph, graph.read_view()]:
        assert set(parent.filter_nodes(gr.NodeFilter.has_neighbor(a)).node_ids) == {b}
        assert set(parent.filter_nodes(gr.NodeFilter.has_neighbor(b)).node_ids) == {a, b}
    # HasNeighbor keeps the existing incident query semantics for directed graphs.
    assert graph.read_view().neighbors(b) == [b]


def test_query_validation_and_empty_edge_boolean_conventions():
    graph = gr.Graph()
    a, b = graph.add_nodes(2)
    edge = graph.add_edge(a, b)
    frozen = graph.read_view()
    assert frozen.filter_edges(gr.EdgeFilter.and_filters([])).edge_ids == [edge]
    assert frozen.filter_edges(gr.EdgeFilter.or_filters([])).edge_ids == []
    assert frozen.filter_nodes(gr.NodeFilter.has_attribute("missing")).node_ids == []
    for kind in ["filter_nodes", "filter_edges"]:
        with pytest.raises(TypeError):
            getattr(frozen, kind)(123)
        with pytest.raises(ValueError):
            getattr(frozen, kind)("")
