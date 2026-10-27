"""Null query syntax preserves explicit absence and existing identifier syntax."""
import pytest
from groggy import Graph


def test_null_query_literal_preserves_missing_text_and_named_properties():
    graph = Graph()
    a = graph.add_node(x=None, null=None)
    b = graph.add_node(x="null")
    graph.add_node()
    graph.add_edge(a, b, x=None)
    graph.add_edge(a, b, x="null")
    view = graph.read_view()
    assert view.filter_nodes("x == null").node_ids == [a]
    assert view.filter_nodes("x != NULL").node_ids == [b]
    assert view.filter_nodes("NOT (x == null)").node_ids == [b, 2]
    assert view.filter_nodes("null == null").node_ids == [a]
    assert view.filter_nodes("x == 'null'").node_ids == [b]
    assert view.filter_edges("x == nUlL").edge_ids == [0]
    assert view.filter_edges("x != null").edge_ids == [1]
    for query in ["x == null extra", "x == null null", "x == null)", "x == nul"]:
        with pytest.raises(ValueError):
            view.filter_nodes(query)
        with pytest.raises(ValueError):
            view.filter_edges(query)
