"""Whole-selection column indexing follows canonical row IDs and read ownership."""
import pytest
from groggy import Graph


@pytest.mark.parametrize("captured", [False, True])
def test_view_column_indexing_is_sorted_aligned_and_preserves_absent_rows(captured):
    graph = Graph()
    graph.add_nodes([{"score": 0.123456789012345}, {}, {"score": None}, {"score": 7}])
    subject = graph.read_view() if captured else graph.view()
    assert list(subject.node_ids) == [0, 1, 2, 3]
    assert subject["score"].to_list() == [0.123456789012345, None, None, 7]
    assert subject["unknown"].to_list() == [None] * 4
    accessor = subject.to_nodes()
    ids = accessor.ids if captured else accessor.ids()
    assert ids.to_list() == [0, 1, 2, 3]
    graph.remove_node(0)
    if captured:
        assert subject["score"].to_list() == [0.123456789012345, None, None, 7]
    else:
        with pytest.raises((ValueError, RuntimeError)):
            subject["score"]


def test_connected_column_indexing_rejects_removed_ids_instead_of_silently_substituting_null():
    graph = Graph()
    graph.add_nodes([{"score": 11}, {"score": 12}])
    selection = graph.view()
    graph.remove_node(1)
    with pytest.raises((ValueError, RuntimeError)):
        selection["score"]


@pytest.mark.parametrize("captured", [False, True])
def test_empty_column_and_bad_key_types_have_consistent_read_semantics(captured):
    graph = Graph()
    subject = graph.read_view() if captured else graph.view()
    assert subject["score"].to_list() == []
    for key in [0, True, ["score"], None]:
        with pytest.raises(TypeError):
            subject[key]


@pytest.mark.parametrize("captured", [False, True])
def test_subset_columns_and_node_edge_adapters_use_the_same_ascending_id_order(captured):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": node} for node in range(10)])
    for node in range(9):
        graph.add_edge(node, node + 1)
    subject = graph.read_view().select_edges([7, 2, 5]) if captured else graph.subgraph_from_edges([7, 2, 5])
    assert list(subject.node_ids) == [2, 3, 5, 6, 7, 8]
    assert list(subject.edge_ids) == [2, 5, 7]
    assert subject["score"].to_list() == [2, 3, 5, 6, 7, 8]
    for accessor in [subject.nodes, subject.to_nodes()]:
        ids = accessor.ids if captured else accessor.ids()
        assert ids.to_list() == [2, 3, 5, 6, 7, 8]
    for accessor in [subject.edges, subject.to_edges()]:
        ids = accessor.ids if captured else accessor.ids()
        assert ids.to_list() == [2, 5, 7]


def test_captured_column_is_detached_from_owner_and_preserves_typed_payloads():
    graph = Graph()
    graph.add_nodes([{"value": [0.123456789012345, 2.0]}, {"value": "text"}, {}])
    source = graph.read_view()
    values = source["value"]
    assert values.to_list() == [[0.123456789012345, 2.0], "text", None]
    graph.set_node_attr(0, "value", 99)
    del graph
    assert source["value"].to_list() == values.to_list()
