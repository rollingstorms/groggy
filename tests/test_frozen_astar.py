"""Captured A* paths preserve selected edge instances and checked heuristics."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


@pytest.mark.parametrize("directed", [False, True])
def test_captured_path_uses_selected_weights_heuristics_and_actual_edges(directed):
    graph = gr.Graph(directed=directed)
    a, b, c = graph.add_nodes(3)
    first = graph.add_edge(a, b)
    parallel = graph.add_edge(a, b)
    second = graph.add_edge(b, c)
    graph.add_edge(a, c)  # Excluded shortcut.
    graph.set_edge_attr(first, "w", 4)
    graph.set_edge_attr(parallel, "w", 4)
    graph.set_edge_attr(second, "w", 1)
    for node, h in [(a, 5), (b, 1), (c, 0)]:
        graph.set_node_attr(node, "h", h)
    view = graph.read_view().select_edges([first, parallel, second])
    graph.set_edge_attr(first, "w", 999)
    graph.set_node_attr(a, "h", 999)
    graph.remove_node(b)
    del graph
    path = view.astar_path(a, c, weight_attr="w", heuristic_attr="h")
    assert isinstance(path, gr.GraphPath)
    assert path.nodes == [a, b, c]
    assert path.edges == [first, second]
    assert path.total_weight == 5.0
    assert path.length == 2
    assert path.algorithm == "astar"
    copy = path.nodes
    copy.clear()
    assert path.nodes == [a, b, c]
    with pytest.raises(AttributeError):
        path.total_weight = 0
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(
            workers.map(lambda _: view.astar_path(a, c, "w", "h").edges, range(4))
        )
        assert results == [[first, second]] * 4
    if not directed:
        assert view.astar_path(c, a, "w").edges == [second, first]
    else:
        with pytest.raises((ValueError, RuntimeError), match="unreachable"):
            view.astar_path(c, a, "w")


@pytest.mark.parametrize("bad", [None, True, "2", -1, float("inf"), float("nan")])
def test_invalid_isolated_heuristics_are_checked_even_for_a_trivial_path(bad):
    graph = gr.Graph()
    a, isolated = graph.add_nodes(2)
    graph.set_node_attr(isolated, "h", bad)
    with pytest.raises((ValueError, RuntimeError), match="heuristic"):
        graph.read_view().astar_path(a, a, heuristic_attr="h")


def test_heuristic_consistency_goal_zero_and_start_membership():
    graph = gr.Graph(directed=True)
    a, b = graph.add_nodes(2)
    edge = graph.add_edge(a, b)
    graph.set_edge_attr(edge, "w", 2)
    graph.set_node_attr(a, "h", 2)
    graph.set_node_attr(b, "h", 0)
    view = graph.read_view()
    assert view.astar_path(a, b, "w", "h").total_weight == 2
    with pytest.raises((ValueError, RuntimeError), match="goal must be zero"):
        view.astar_path(a, a, "w", "h")
    graph.set_node_attr(a, "h", 3)
    with pytest.raises((ValueError, RuntimeError), match="inconsistent"):
        graph.read_view().astar_path(a, b, "w", "h")
    with pytest.raises((ValueError, RuntimeError), match="captured selection"):
        view.select_nodes([a]).astar_path(a, b)
    trivial = view.select_nodes([a]).astar_path(a, a)
    assert trivial.nodes == [a]
    assert trivial.edges == []
    assert trivial.total_weight == 0
