"""Traversal trees/path membership comes from the selected capture, never owner shortcuts."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


def fixture(directed=True):
    graph = gr.Graph(directed=directed)
    graph.add_nodes(6)
    for source, target in [(0, 1), (1, 2), (0, 3), (3, 2), (0, 1), (1, 1), (2, 4)]:
        graph.add_edge(source, target)
    return graph


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_no_owner_route_through_excluded_intermediate_or_edge(directed, captured):
    graph = fixture(directed)
    selected = graph.induced_subgraph([0, 2, 4])
    if captured:
        selected = selected.pin_view()
    assert not selected.has_path(0, 2)
    assert selected.shortest_path_subgraph(0, 2) is None
    assert selected.bfs(0).node_count() == 1
    assert selected.dfs(0).node_count() == 1
    exact = graph.subgraph_from_edges([2, 3, 6])
    if captured:
        exact = exact.pin_view()
    path = exact.shortest_path_subgraph(0, 4)
    assert path.node_count() == 4
    assert sorted(path.edge_ids if captured else path.edge_ids.to_list()) == [2, 3, 6]


@pytest.mark.parametrize("method", ["bfs", "dfs"])
@pytest.mark.parametrize("depth,nodes,edges", [
    (None, [0, 1, 2, 3, 4], [0, 1, 2, 6]),
    (0, [0], []), (1, [0, 1, 3], [0, 2]),
])
def test_discovery_trees_have_deterministic_neighbors_parallel_ties_and_depth(method, depth, nodes, edges):
    graph = fixture()
    before = graph.current_state_id
    tree = getattr(graph.read_view(), method)(0, depth)
    assert tree.node_ids == nodes
    assert tree.edge_ids == edges
    assert graph.current_state_id == before
    assert tree.edge_count() == tree.node_count() - 1


@pytest.mark.parametrize("depth", [True, -1, 1.5, "2", 2**100])
@pytest.mark.parametrize("method", ["bfs", "dfs"])
def test_frozen_traversal_depth_requires_nonnegative_integer(method, depth):
    with pytest.raises((ValueError, RuntimeError, TypeError, OverflowError)):
        getattr(fixture().read_view(), method)(0, depth)


def test_exact_parent_edge_scope_survives_connected_induction_and_edge_selection():
    graph = fixture(directed=False)
    selected = graph.subgraph_from_edges([0])
    assert sorted(selected.induced_subgraph([0, 1]).edge_ids.to_list()) == [0]
    assert sorted(selected.subgraph_from_edges([0, 4]).edge_ids.to_list()) == [0]
    assert selected.pin_view().select_nodes([0, 1]).edge_ids == [0]


def test_frozen_paths_survive_owner_removal_and_workers():
    graph = fixture()
    root = graph.read_view()
    path = root.shortest_path_subgraph(0, 4)
    assert path.edge_ids == [0, 1, 6]
    assert path.node_ids == [0, 1, 2, 4]
    graph.remove_node(1)
    del graph
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(lambda _: root.bfs(0).edge_ids, range(4))) == [[0, 1, 2, 6]] * 4
    assert root.shortest_path_subgraph(0, 4).edge_ids == path.edge_ids


def test_empty_single_node_outside_scope_and_direction_contracts():
    graph = gr.Graph(directed=True)
    assert not graph.read_view().is_connected()
    assert not graph.induced_subgraph([]).is_connected()
    node = graph.add_node()
    view = graph.read_view()
    assert view.is_connected()
    assert view.shortest_path_subgraph(node, node).node_ids == [node]
    assert view.shortest_path_subgraph(node, node).edge_ids == []
    assert not view.has_path(99, 99)
    assert view.shortest_path_subgraph(99, node) is None
    with pytest.raises((ValueError, RuntimeError)):
        view.bfs(99)
    with pytest.raises((ValueError, RuntimeError)):
        view.dfs(99)
    graph = fixture()
    view = graph.read_view().select_nodes([0, 1])
    assert view.is_connected("weak")
    assert not view.is_connected("strong")
    assert view.has_path(0, 1)
    assert not view.has_path(1, 0)
    assert view.shortest_path_subgraph(1, 0) is None


def test_connected_components_fallback_and_cache_do_not_use_unselected_bridge():
    graph = fixture(directed=False)
    selected = graph.subgraph_from_edges([0, 6])
    for _ in range(2):
        assert len(selected.connected_components()) == 2
    graph.set_node_attr(0, "new_value", 1)
    assert len(selected.connected_components()) == 2
    graph.remove_edge(0)
    with pytest.raises((ValueError, RuntimeError)):
        selected.connected_components()


def test_restoration_invalidates_scope_csr_and_components_even_when_space_versions_repeat():
    graph = gr.Graph()
    graph.add_nodes(3)
    graph.add_edge(0, 1)
    first = graph.current_state_id
    graph.set_edge_endpoints(0, 0, 2)
    second = graph.current_state_id
    graph.restore_state(first, mode="switch")
    selected = graph.induced_subgraph([0, 1, 2])
    assert selected.neighbors(0).to_list() == [1]
    assert not selected.has_path(0, 2)
    assert len(selected.connected_components()) == 2
    graph.restore_state(second, mode="switch")
    assert selected.neighbors(0).to_list() == [2]
    assert selected.has_path(0, 2)
    assert not selected.has_path(0, 1)
    assert selected.bfs(0).has_node(2)
    assert not selected.bfs(0).has_node(1)
