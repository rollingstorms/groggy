"""Parallel facts and self-loops retain their own edge IDs in Python."""

import groggy as gr


def test_weighted_path_keeps_the_chosen_parallel_edge():
    graph = gr.Graph(directed=True)
    source = graph.add_node()
    target = graph.add_node()
    expensive = graph.add_edge(source, target, weight=10.0)
    cheap = graph.add_edge(source, target, weight=1.0)

    assert expensive != cheap
    assert len(graph.edges) == 2
    path = graph.shortest_path(source, target, weight_attribute="weight")
    assert list(path.edge_ids) == [cheap]


def test_self_loop_keeps_one_edge_id():
    graph = gr.Graph()
    node = graph.add_node()
    loop_edge = graph.add_edge(node, node)

    assert list(graph.edge_ids) == [loop_edge]
    assert graph.edge_endpoints(loop_edge) == (node, node)
