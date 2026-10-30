"""Default selections retain their source; only explicit methods write the owner."""
import gc
import weakref

import pytest
from groggy import Graph


def graph_with_values():
    graph = Graph(directed=True)
    graph.add_nodes([{"value": 7}, {"value": 8}, {"value": 9}])
    graph.add_edge(0, 1, weight=2.5)
    return graph


def test_default_reads_and_derivations_stay_frozen_after_owner_changes():
    graph = graph_with_values()
    selected = graph.induced_subgraph([0, 1])
    nodes = selected.nodes
    edges = selected.edges
    graph.set_node_attr(0, "value", 99)
    graph.set_edge_endpoints(0, 2, 1)
    graph.remove_node(1)
    assert selected["value"].to_list() == [7, 8]
    assert selected.value == {0: 7, 1: 8}
    assert nodes[0]["value"] == 7
    assert edges[0].endpoints == (0, 1)
    assert selected.neighbors(0).to_list() == [1]
    assert all(type(node) is int for node in selected.neighbors(0).to_list())
    assert selected.degree(0) == 1
    assert selected.bfs(0).node_ids.to_list() == [0, 1]
    assert selected.dfs(0).node_ids.to_list() == [0, 1]
    assert selected.shortest_path_subgraph(0, 1).node_ids.to_list() == [0, 1]
    assert selected.induced_subgraph([0])["value"].to_list() == [7]
    assert selected.subgraph_from_edges([0]).edge_ids.to_list() == [0]
    assert selected.neighborhood([0], hops=1)[0].node_ids.to_list() == [0, 1]
    assert selected.sample(2, seed=1).edge_ids.to_list() == [0]
    assert selected.adjacency_matrix(sparse=True).shape == (2, 2)
    assert selected.to_graph().get_node_attr(0, "value") == 7
    assert selected.to_nodes()[0]["value"] == 7
    assert selected.to_edges()[0].endpoints == (0, 1)
    assert "0" in str(selected)


def test_lazy_components_keep_the_source_before_first_item_access():
    graph = graph_with_values()
    components = graph.connected_components()
    graph.set_node_attr(0, "value", 99)
    graph.remove_node(1)
    component = next(item for item in components if 0 in item.node_ids.to_list())
    assert component["value"].to_list() == [7, 8]
    assert component.edge_ids.to_list() == [0]


def test_explicit_bulk_writes_preserve_original_and_require_fresh_source():
    graph = graph_with_values()
    selected = graph.induced_subgraph([0, 1])
    before = graph.current_state_id
    selected.set_node_attrs({"value": {0: 11, 1: 12}})
    assert graph.current_state_id == before + 1
    assert graph.get_node_attr(0, "value") == 11
    assert selected["value"].to_list() == [7, 8]
    with pytest.raises((ValueError, RuntimeError)):
        selected.set_node_attrs({"value": {0: 13}})
    assert graph.get_node_attr(0, "value") == 11
    fresh = graph.induced_subgraph([0, 1])
    fresh.set_edge_attrs({"weight": {0: 4.5}})
    assert fresh.edges[0]["weight"] == 2.5
    assert graph.get_edge_attr(0, "weight") == 4.5


def test_default_cache_refreshes_values_and_does_not_keep_unowned_view_alive():
    graph = graph_with_values()
    old = graph.view()
    assert graph.view() is old
    graph.set_node_attr(0, "value", 99)
    new = graph.view()
    assert new is not old
    assert old.nodes[0]["value"] == 7
    assert new.nodes[0]["value"] == 99
    ref = weakref.ref(new)
    del new
    gc.collect()
    assert ref() is None
    assert graph.view().nodes[0]["value"] == 99


def test_related_default_selections_share_source_but_explicit_captures_do_not():
    graph = graph_with_values()
    left = graph.induced_subgraph([0, 1])
    right = graph.induced_subgraph([1, 2])
    assert left.merge_with(right).node_ids.to_list() == [0, 1, 2]
    graph.set_node_attr(0, "value", 99)
    with pytest.raises((ValueError, RuntimeError)):
        left.merge_with(graph.induced_subgraph([1, 2]))


def test_explicit_connected_compatibility_view_reads_current_owner():
    graph = graph_with_values()
    connected = graph.connected_view()
    captured = graph.view()
    graph.set_node_attr(0, "value", 99)
    assert connected["value"].to_list() == [99, 8, 9]
    assert captured["value"].to_list() == [7, 8, 9]
    graph.remove_node(1)
    with pytest.raises((ValueError, RuntimeError)):
        connected.adjacency_matrix(sparse=True)
