"""Small cross-language checks for Groggy's established graph contract."""

import groggy as gr


def test_empty_graph_is_a_valid_construction_state():
    graph = gr.Graph()

    assert graph.is_undirected
    assert not graph.is_directed
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


def test_one_node_without_edges_is_the_minimal_graph():
    graph = gr.Graph()
    node = graph.add_node()

    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0
    assert graph.contains_node(node)
    assert graph.get_node_attr(node, "entity_type") == "base"


def test_direction_is_selected_when_the_graph_is_created():
    graph = gr.Graph(directed=True)

    assert graph.is_directed
    assert not graph.is_undirected
    assert len(graph.nodes) == 0
