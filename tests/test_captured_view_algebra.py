"""Exact membership operations share captured values; connected adapters stay explicit."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


def fixture():
    graph = gr.Graph()
    graph.add_nodes(5)
    for source, target in [(0, 1), (0, 1), (1, 2), (0, 0)]:
        graph.add_edge(source, target)
    return graph


@pytest.mark.parametrize("metric,expected", [
    ("jaccard", 0.5), ("dice", 2 / 3),
    ("cosine", 2 / 3), ("overlap", 2 / 3),
])
def test_frozen_exact_union_intersection_difference_and_similarity(metric, expected):
    graph = fixture()
    root = graph.read_view()
    left = root.select_nodes([0, 1, 3])
    right = root.select_nodes([0, 1, 2])
    before = graph.current_state_id
    assert left.merge_with(right).node_ids == [0, 1, 2, 3]
    assert left.intersect_with(right).node_ids == [0, 1]
    assert left.subtract_from(right).node_ids == [3]
    assert left.subtract_from(right).edge_ids == []
    assert left.calculate_similarity(right, metric) == pytest.approx(expected)
    assert graph.current_state_id == before
    a, b = root.select_edges([0]), root.select_edges([1])
    assert a.merge_with(b).edge_ids == [0, 1]  # omit loop 3
    assert a.intersect_with(b).node_ids == [0, 1]
    assert a.intersect_with(b).edge_ids == []


@pytest.mark.parametrize("method", ("merge_with", "intersect_with", "subtract_from", "calculate_similarity"))
def test_independent_captures_reject_even_when_payload_is_reused(method):
    graph = fixture()
    root = graph.read_view()
    for other in [graph.read_view(), fixture().read_view()]:
        with pytest.raises((ValueError, RuntimeError), match="same capture"):
            getattr(root, method)(other)
    with pytest.raises((ValueError, RuntimeError)):
        root.calculate_similarity(root, "unknown")


def test_difference_prunes_incident_edges_and_preserves_frozen_worker_reads():
    graph = fixture()
    root = graph.read_view()
    difference = root.subtract_from(root.select_nodes([1]))
    assert difference.node_ids == [0, 2, 3, 4]
    assert difference.edge_ids == [3]
    graph.remove_node(0)
    del graph
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: difference.merge_with(difference.copy()).edge_ids, range(4)))
    assert results == [[3]] * 4


def test_grouping_preserves_exact_typed_keys_missing_null_and_selected_edges():
    graph = fixture()
    for node, value in [(0, 2**53 + 1), (1, 2**53 + 1), (2, 2**53), (3, None)]:
        graph.set_node_attr(node, "group", value)
    graph.set_edge_attr(0, "group", "a")
    graph.set_edge_attr(1, "group", None)
    graph.set_edge_attr(2, "group", "a")
    root = graph.read_view()
    groups = root.group_by_nodes("group")
    assert [view.node_ids for view in groups] == [[3], [2], [0, 1]]
    assert groups[2].edge_ids == [0, 1, 3]
    edges = root.select_edges([0, 2]).group_by_edges("group")
    assert len(edges) == 1
    assert edges[0].edge_ids == [0, 2]
    assert edges[0].node_ids == [0, 1, 2]
    assert root.group_by_nodes("absent") == []
    graph.set_node_attr(0, "group", "changed")
    graph.remove_node(1)
    assert root.group_by_nodes("group")[2].node_ids == [0, 1]


def test_connected_set_operations_are_implemented_and_use_the_same_owner():
    graph = fixture()
    left, right = graph.induced_subgraph([0, 1, 3]), graph.induced_subgraph([1])
    result = left.subtract_from(right)
    assert sorted(result.node_ids.to_list()) == [0, 3]
    assert sorted(result.edge_ids.to_list()) == [3]
    assert left.merge_with(right).node_count() == 3
    assert left.intersect_with(right).node_count() == 1
    other = fixture().induced_subgraph([1])
    for method in ("merge_with", "intersect_with", "subtract_from"):
        with pytest.raises((ValueError, RuntimeError), match="same capture"):
            getattr(left, method)(other)
