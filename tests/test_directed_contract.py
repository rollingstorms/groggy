"""Directed reads use edge orientation in the Python graph API."""

import groggy as gr


def test_directed_paths_do_not_traverse_an_edge_backward():
    graph = gr.Graph(directed=True)
    source = graph.add_node()
    target = graph.add_node()
    graph.add_edge(source, target)

    assert graph.has_edge_between(source, target)
    assert not graph.has_edge_between(target, source)
    assert graph.shortest_path(source, target) is not None
    assert graph.shortest_path(target, source) is None
    assert len(graph.bfs(target).nodes) == 1


def test_directed_adjacency_uses_the_same_node_order_as_graph_ids():
    graph = gr.Graph(directed=True)
    source = graph.add_node()
    target = graph.add_node()
    graph.add_edge(source, target)

    node_ids = list(graph.node_ids)
    source_index = node_ids.index(source)
    target_index = node_ids.index(target)
    adjacency = graph.adjacency_matrix().to_numpy()

    assert adjacency[source_index, target_index] == 1.0
    assert adjacency[target_index, source_index] == 0.0
