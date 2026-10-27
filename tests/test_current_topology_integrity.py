"""Explicit integrity checks include the captured current topology cache."""
import pytest

from groggy import Graph


@pytest.mark.parametrize("directed", [False, True])
def test_warm_topology_integrity_preserves_state_history_and_pinned_results(directed):
    graph = Graph(directed=directed)
    graph.add_nodes(4)
    graph.add_edge(0, 1)
    graph.add_edge(0, 1)
    graph.add_edge(1, 1)
    graph.neighbors(0)
    head = graph.current_state_id
    pin = graph.read_view()
    versions = len(graph.commit_history())
    report = graph.check_integrity()
    assert report["active_nodes"] == 4
    assert report["active_edges"] == 3
    assert graph.current_state_id == head
    assert len(graph.commit_history()) == versions
    with pytest.raises(ValueError):
        graph.set_edge_endpoints(0, 999, 1)
    assert graph.current_state_id == head
    graph.check_integrity()
    graph.set_edge_endpoints(0, 2, 1)
    graph.neighbors(2)
    graph.check_integrity()
    assert pin.edge_endpoints(0) == (0, 1)
    assert graph.read_view().edge_endpoints(0) == (2, 1)
