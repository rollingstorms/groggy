"""Connected property access must respect selection independently of owner liveness."""
import pytest
from groggy import Graph


def fixture():
    graph = Graph()
    for i in range(3):
        graph.add_node(score=i, included=i < 2)
    first = graph.add_edge(0, 1, weight=1)
    second = graph.add_edge(0, 1, weight=2)
    return graph, first, second


def test_connected_property_reads_reject_other_active_nodes():
    graph, _, _ = fixture()
    selected = graph.filter_nodes("included == true")
    assert selected.get_node_attribute(0, "score") == 0
    with pytest.raises(RuntimeError, match="outside.*selection"):
        selected.get_node_attribute(2, "score")
    assert graph.get_node_attr(2, "score") == 2


def test_edge_property_reads_reject_excluded_parallel_instances():
    graph, first, second = fixture()
    selected = graph.filter_edges("weight == 1")
    assert selected.get_edge_attribute(first, "weight") == 1
    with pytest.raises(RuntimeError, match="outside.*selection"):
        selected.get_edge_attribute(second, "weight")
    assert graph.get_edge_attr(second, "weight") == 2


@pytest.mark.parametrize("edge", [False, True])
def test_connected_bulk_property_errors_preserve_the_selected_prefix_and_history(edge):
    graph, first, second = fixture()
    head = graph.current_state_id
    if edge:
        selected = graph.filter_edges("weight == 1")
        call = lambda: selected.set_edge_attrs({"weight": {first: 7, second: 8}})
    else:
        selected = graph.filter_nodes("included == true")
        call = lambda: selected.set_node_attrs({"score": {0: 7, 2: 8}})
    with pytest.raises((ValueError, RuntimeError, KeyError, PermissionError)):
        call()
    assert graph.get_node_attr(0, "score") == 0
    assert graph.get_edge_attr(first, "weight") == 1
    assert graph.current_state_id == head
