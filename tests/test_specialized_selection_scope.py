"""Neighborhood-to-base conversion preserves selected scope and frozen ownership."""
import groggy as gr
import pytest


@pytest.mark.parametrize("directed", [False, True])
def test_neighborhood_bridge_keeps_exact_scope_and_frozen_read_lifetime(directed):
    graph = gr.Graph(directed=directed)
    graph.add_nodes(5)
    for source, target in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        graph.add_edge(source, target)
    neighborhood = graph.neighborhood([0], radius=2)[0]
    frozen = neighborhood.pin_view()
    before = graph.current_state_id
    assert frozen.node_ids == [0, 1, 2]
    assert frozen.edge_ids == [0, 1]
    assert neighborhood.bfs(0).node_count() == 3
    assert neighborhood.shortest_path_subgraph(0, 4) is None
    assert neighborhood.induced_subgraph([0, 1, 2, 3]).node_count() == 3
    assert graph.current_state_id == before
    graph.remove_node(2)
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        neighborhood.neighbors(0)
    assert frozen.bfs(0).node_ids == [0, 1, 2]
    assert frozen.shortest_path_subgraph(0, 2).edge_ids == [0, 1]


def test_component_and_neighborhood_wrappers_expose_explicit_capture():
    # Native constructors/metadata and actual pinned lifetime are exercised in
    # specialized_selection_scope.rs; these are the curated Python class adapters.
    from groggy import _groggy as native

    assert callable(native.ComponentSubgraph.pin_view)
    assert callable(native.NeighborhoodSubgraph.pin_view)
