"""Public Python reads use the graph's active membership."""

import pytest

import groggy as gr


def test_removed_entities_do_not_appear_in_live_reads():
    graph = gr.Graph()
    source = graph.add_node()
    target = graph.add_node()
    edge = graph.add_edge(source, target)

    graph.remove_edge(edge)
    assert not graph.contains_edge(edge)
    assert len(graph.edges) == 0
    assert not graph.has_edge_attribute(edge, "weight")
    assert graph.edge_attribute_keys(edge) == []
    with pytest.raises(Exception):
        graph.edge_endpoints(edge)

    graph.remove_node(source)
    assert not graph.contains_node(source)
    assert len(graph.nodes) == 1
    assert not graph.has_node_attribute(source, "entity_type")
    assert graph.node_attribute_keys(source) == []
    with pytest.raises(Exception):
        graph.get_node_attr(source, "entity_type")
