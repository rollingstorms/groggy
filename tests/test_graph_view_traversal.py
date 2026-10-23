"""Frozen traversal uses captured topology and the existing exact integer kernels."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


@pytest.mark.parametrize("directed", [False, True])
def test_frozen_distances_and_order_keep_scope_after_owner_changes(directed):
    graph = gr.Graph(directed=directed)
    a, b, c, outside = graph.add_nodes(4)
    first = graph.add_edge(a, c)
    onward = graph.add_edge(c, b)
    graph.add_edge(a, b)  # Excluded shortcut.
    graph.add_edge(b, outside)
    parallel = graph.add_edge(a, c)
    loop = graph.add_edge(c, c)
    graph.set_node_attr(a, "score", 7)
    view = graph.read_view().select_edges([first, onward, parallel, loop])
    graph.remove_node(a)
    del graph
    assert view.bfs_distances(a) == [(a, 0), (b, 2), (c, 1)]
    assert view.dfs_order(a) == [(a, 0), (c, 1), (b, 2)]
    assert view.bfs_distances(b) == ([(b, 0)] if directed else [(a, 2), (b, 0), (c, 1)])
    assert view.node_attr(a, "score") == 7
    with ThreadPoolExecutor(max_workers=2) as workers:
        rows = list(workers.map(lambda _: view.bfs_distances(a), range(4)))
    assert rows == [[(a, 0), (b, 2), (c, 1)]] * 4
    assert all(type(value) is int for _, value in view.dfs_order(a))
    assert outside not in [node for node, _ in view.bfs_distances(a)]


def test_dfs_uses_ascending_neighbor_recursive_preorder():
    graph = gr.Graph(directed=True)
    nodes = graph.add_nodes(5)
    for a, b in [(0, 2), (0, 1), (1, 3), (2, 3), (3, 0), (0, 2)]:
        graph.add_edge(nodes[a], nodes[b])
    view = graph.read_view()
    assert view.dfs_order(nodes[0]) == [(nodes[0], 0), (nodes[1], 1), (nodes[3], 2), (nodes[2], 3)]
    assert view.bfs_distances(nodes[0]) == [(nodes[0], 0), (nodes[1], 1), (nodes[2], 1), (nodes[3], 2)]


def test_empty_single_node_unknown_and_unselected_starts():
    graph = gr.Graph()
    with pytest.raises((ValueError, RuntimeError), match="captured selection"):
        graph.read_view().bfs_distances(0)
    a, b = graph.add_nodes(2)
    view = graph.read_view().select_nodes([b])
    assert view.bfs_distances(b) == [(b, 0)]
    assert view.dfs_order(b) == [(b, 0)]
    for start in [a, 999999]:
        with pytest.raises((ValueError, RuntimeError), match="captured selection"):
            view.bfs_distances(start)
        with pytest.raises((ValueError, RuntimeError), match="captured selection"):
            view.dfs_order(start)


def test_traversal_reads_do_not_add_or_change_graph_properties():
    graph = gr.Graph()
    a, b = graph.add_nodes(2)
    graph.add_edge(a, b)
    graph.set_node_attr(a, "score", 3)
    view = graph.read_view()
    before = {node: view.node_attribute_keys(node) for node in [a, b]}
    view.bfs_distances(a)
    view.dfs_order(a)
    after = graph.read_view()
    assert {node: after.node_attribute_keys(node) for node in [a, b]} == before
    assert after.node_attr(a, "score") == 3
