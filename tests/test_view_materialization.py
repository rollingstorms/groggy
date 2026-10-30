"""Explicit selected graph copies contain actual data and independent publication."""
import pytest
from groggy import Graph


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("directed", [False, True])
def test_selected_copy_preserves_sparse_ids_values_and_direction(captured, directed):
    graph = Graph(directed=directed)
    graph.add_nodes([{}, {"score": 0.123456789012345}, {}, {"score": None}, {}])
    graph.add_edge(1, 3, weight=2.5)
    graph.add_edge(1, 3, weight=100)
    selected = graph.subgraph_from_edges([0])
    source = selected.pin_view() if captured else selected
    copy = source.to_graph()
    assert sorted(copy.node_ids) == [1, 3]
    assert copy.edge_ids.to_list() == [0]
    assert copy.is_directed == directed
    assert copy.get_node_attr(1, "score") == 0.123456789012345
    assert copy.get_node_attr(3, "score") is None
    assert copy.get_edge_attr(0, "weight") == 2.5
    assert copy.current_state_id == 1
    copy.set_node_attr(1, "score", 99)
    assert copy.current_state_id == 2
    assert graph.get_node_attr(1, "score") == 0.123456789012345
    assert copy.add_node() > 3


def test_frozen_meta_copy_survives_owner_deletion_and_keeps_nested_source():
    graph = Graph()
    graph.add_nodes(2)
    group = graph.retain_group(graph.read_view().select_nodes([0]), kind="source")
    meta = graph.create_meta_node(group)
    frozen = graph.read_view().select_nodes([meta])
    graph.remove_node(0)
    del graph
    copy = frozen.to_graph()
    assert copy.node_ids.to_list() == [meta]
    assert copy.read_view().meta_subgraph(meta).node_ids == [0]
    copy.check_integrity()


def test_connected_copy_rejects_stale_membership_and_empty_copy_stays_empty():
    graph = Graph(directed=True)
    empty = graph.read_view().to_graph()
    assert empty.node_count() == 0
    assert empty.current_state_id == 0
    assert empty.is_directed
    graph.add_nodes(2)
    selected = graph.connected_view()
    graph.remove_node(1)
    with pytest.raises((ValueError, RuntimeError)):
        selected.to_graph()
