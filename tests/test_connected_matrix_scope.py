"""Connected numerical exports must not silently shrink stale graph selections."""
import pytest
from groggy import Graph


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("operation", ["node", "edge"])
def test_stale_connected_matrix_rejects_instead_of_filtering_inactive_selection(sparse, operation):
    graph = Graph(directed=True)
    graph.add_nodes(3)
    graph.add_edge(0, 1, weight=2.5)
    selection = graph.view()
    frozen = selection.pin_view()
    before = selection.adjacency_matrix(sparse=sparse, weight_attr="weight")
    assert before.shape == (3, 3)
    if operation == "node":
        graph.remove_node(2)
    else:
        graph.remove_edge(0)
    with pytest.raises((ValueError, RuntimeError)):
        selection.adjacency_matrix(sparse=sparse, weight_attr="weight")
    assert frozen.adjacency_matrix(sparse=sparse, weight_attr="weight").shape == (3, 3)
    assert before.shape == (3, 3)
    if not sparse:
        with pytest.raises((ValueError, RuntimeError)):
            selection.to_matrix()
        with pytest.raises((ValueError, RuntimeError)):
            selection.adj()


def test_rewired_outside_endpoints_reject_while_valid_inside_revisions_remain_live():
    graph = Graph(directed=True)
    graph.add_nodes(4)
    graph.add_edge(0, 1)
    selection = graph.induced_subgraph([0, 1, 2])
    graph.set_edge_endpoints(0, 2, 1)
    assert selection.adjacency_matrix(sparse=True).shape == (3, 3)
    graph.set_edge_endpoints(0, 3, 1)
    with pytest.raises((ValueError, RuntimeError)):
        selection.adjacency_matrix(sparse=True)
