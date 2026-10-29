"""Adjacency export shares selected directed/undirected edge-instance semantics."""
import pytest
from groggy import Graph


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("source", ["graph", "connected", "captured"])
def test_adjacency_export_preserves_direction_isolates_loops_and_parallel_instances(directed, source):
    graph = Graph(directed=directed)
    graph.add_nodes(5)
    for edge in [(0, 2), (0, 1), (0, 2), (2, 2), (3, 0)]:
        graph.add_edge(*edge)
    subject = graph if source == "graph" else graph.view() if source == "connected" else graph.read_view()
    expected = {0: [1, 2, 2], 1: [], 2: [2], 3: [0], 4: []} if directed else {0: [1, 2, 2, 3], 1: [0], 2: [0, 0, 2], 3: [0], 4: []}
    assert subject.adjacency_list() == expected
    assert list(subject.adjacency_list()) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("directed", [False, True])
def test_selected_adjacency_does_not_induce_excluded_parallel_instances(directed):
    graph = Graph(directed=directed)
    graph.add_nodes(4)
    graph.add_edge(0, 1)
    graph.add_edge(0, 1)
    graph.add_edge(1, 1)
    graph.add_edge(2, 3)
    selected = graph.subgraph_from_edges([1, 2])
    frozen = selected.pin_view()
    expected = {0: [1], 1: [1] if directed else [0, 1]}
    assert selected.adjacency_list() == frozen.adjacency_list() == expected
    graph.remove_edge(1)
    with pytest.raises((ValueError, RuntimeError)):
        selected.adjacency_list()
    assert frozen.adjacency_list() == expected


def test_empty_and_isolated_adjacency_and_returned_dictionary_are_detached():
    graph = Graph()
    assert graph.adjacency_list() == {}
    graph.add_nodes(2)
    frozen = graph.read_view()
    rows = frozen.adjacency_list()
    rows[0].append(1)
    assert frozen.adjacency_list() == {0: [], 1: []}
    del graph
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(lambda _: frozen.adjacency_list(), range(4))) == [{0: [], 1: []}] * 4
