"""The connected full-view cache follows revisions, not aggregate entity counts."""
import pytest
from groggy import Graph


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("operation", ["node", "edge", "trade"])
def test_full_view_refreshes_after_equal_cardinality_membership_changes(manual, operation):
    graph = Graph(manual=manual)
    graph.add_nodes(3)
    graph.add_edge(0, 1)
    before = graph.view()
    assert graph.view() is before
    if operation == "node":
        graph.remove_node(2)
        graph.add_node()
    elif operation == "edge":
        graph.remove_edge(0)
        graph.add_edge(0, 1)
    else:
        graph.remove_edge(0)
        graph.add_node()
    current = graph.view()
    assert sorted(current.node_ids.to_list()) == sorted(graph.node_ids)
    assert sorted(current.edge_ids.to_list()) == sorted(graph.edge_ids)
    assert current is not before
    assert graph.view() is current
    assert current.pin_view().node_ids == sorted(graph.node_ids)


def test_full_view_cache_restoration_does_not_reuse_equal_numeric_space_versions():
    graph = Graph()
    graph.add_nodes(3)
    graph.add_edge(0, 1)
    first = graph.current_state_id
    graph.remove_edge(0)
    graph.add_edge(1, 2)
    second = graph.current_state_id
    graph.restore_state(first, mode="switch")
    before = graph.view()
    assert before.edge_ids.to_list() == [0]
    graph.restore_state(second, mode="switch")
    after = graph.view()
    assert after.edge_ids.to_list() == [1]
    assert after.pin_view().edge_endpoints(1) == (1, 2)
    assert after is not before


def test_property_and_endpoint_only_writes_preserve_cached_membership_and_current_reads():
    graph = Graph(directed=True)
    graph.add_nodes(3)
    graph.add_edge(0, 1)
    original = graph.view()
    assert original.neighbors(0).to_list() == [1]
    graph.set_node_attr(0, "value", 7)
    assert graph.view() is original
    assert graph.view().pin_view().node_attr(0, "value") == 7
    graph.set_edge_endpoints(0, 2, 1)
    assert graph.view() is original
    assert graph.view().neighbors(0).to_list() == []
    assert graph.view().neighbors(2).to_list() == [1]
    assert graph.view().pin_view().edge_endpoints(0) == (2, 1)
