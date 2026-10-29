"""Deterministic query pages and bounded traversals read captured graph state."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pytest
from groggy import AttrValue, EdgeFilter, Graph, GraphBatch, NodeFilter


def all_nodes():
    return NodeFilter.not_filter(NodeFilter.node_set([]))


def fixture():
    graph = Graph(directed=True)
    batch = GraphBatch(effective_at=10)
    nodes = [batch.add_node(score=value) for value in
             [2**53 + 1, 2**53, AttrValue.float32(1), 1, None]]
    nodes.append(batch.add_node())
    for a, b, weight in [(0, 1, 3), (0, 1, 1), (1, 2, 1), (2, 0, None), (2, 2, -1)]:
        batch.add_edge(nodes[a], nodes[b], weight=weight)
    graph.apply_batch(batch, 0, "fixture", "test")
    return graph


def test_ordered_pages_keep_exact_values_null_order_and_deterministic_ties():
    graph = fixture()
    index = graph.read_view().build_index(node_properties=["score"], edge_properties=["weight"])
    assert index.query_nodes(all_nodes(), order_by="score") == [2, 3, 1, 0, 4, 5]
    descending = [0, 1, 2, 3, 4, 5]
    assert index.query_nodes(all_nodes(), order_by="score", descending=True) == descending
    for offset in [0, 1, 4, 6, 2**64 - 1]:
        for limit in [None, 0, 1, 3, 2**64 - 1]:
            expected = descending[offset:] if limit is None else descending[offset:][:limit]
            assert index.query_nodes(all_nodes(), order_by="score", descending=True,
                                     offset=offset, limit=limit) == expected
    assert index.query_nodes(all_nodes(), descending=True, offset=1, limit=2) == [4, 3]
    assert index.query_edges(EdgeFilter.and_filters([]), order_by="weight") == [4, 1, 2, 0, 3]
    assert index.query_edges("weight >= 1", order_by="weight", descending=True, limit=2) == [0, 1]


@pytest.mark.parametrize("values", [["z", "a"], [True, False],
                                   [date(2024, 2, 29), date(2000, 1, 1)],
                                   [datetime(2024, 1, 1, tzinfo=timezone.utc),
                                    datetime(2000, 1, 1, tzinfo=timezone.utc)],
                                   [timedelta(days=2), timedelta(microseconds=-1)]])
def test_ordering_supports_each_scalar_domain_and_missing_cells(values):
    graph = Graph()
    for value in values:
        graph.add_node(x=value)
    graph.add_node(x=None)
    graph.add_node()
    index = graph.read_view().build_index(node_properties=["x"])
    assert index.query_nodes(all_nodes(), order_by="x") == [1, 0, 2, 3]
    assert index.query_nodes(all_nodes(), order_by="x", descending=True) == [0, 1, 2, 3]


@pytest.mark.parametrize("values", [[float("nan")], [1, "x"], [1, True], [[1, 2]], [b"x"], [{"x": 1}]])
def test_unsupported_ordering_rejects_even_when_the_page_would_be_empty(values):
    graph = Graph()
    for value in values:
        graph.add_node(x=value)
    index = graph.read_view().build_index(node_properties=["x"])
    with pytest.raises(ValueError, match="order|scalar|NaN"):
        index.query_nodes(all_nodes(), order_by="x", limit=0)
    assert index.query_nodes(NodeFilter.node_set([]), order_by="x") == []


@pytest.mark.parametrize("bad", [True, -1, 1.5, "2", 2**100])
def test_query_counts_and_traversal_controls_reject_invalid_types_without_mutation(bad):
    graph = fixture()
    head = graph.current_state_id
    view = graph.read_view()
    index = view.build_index(node_properties=["score"])
    for name in ["offset", "limit"]:
        with pytest.raises((TypeError, OverflowError)):
            index.query_nodes(all_nodes(), **{name: bad})
    with pytest.raises((TypeError, OverflowError)):
        view.within_hops([0], bad)
    with pytest.raises((TypeError, OverflowError)):
        view.within_hops([bad], 1)
    assert graph.current_state_id == head


def test_ordering_index_requirements_are_explicit_before_empty_result_shortcuts():
    graph = fixture()
    index = graph.read_view().build_index(node_properties=["score"])
    with pytest.raises(ValueError, match="index"):
        index.query_nodes(NodeFilter.node_set([]), order_by="unindexed")
    assert index.query_nodes(all_nodes(), order_by="unindexed", allow_scan=True) == list(range(6))
    with pytest.raises(ValueError, match="nonempty"):
        index.query_nodes(all_nodes(), order_by="", allow_scan=True)


def test_bounded_direction_and_selected_edge_scope_keep_valid_captured_subgraphs():
    graph = fixture()
    view = graph.read_view()
    assert view.within_hops([0], 1).node_ids == [0, 1]
    assert view.within_hops([0], 1).edge_ids == [0, 1]
    assert view.within_hops([0], 1, direction="incoming").node_ids == [0, 2]
    assert view.within_hops([0], 1, direction="both").node_ids == [0, 1, 2]
    assert view.within_hops([0, 0, 5], 0).node_ids == [0, 5]
    assert view.within_hops([], 2**64 - 1).node_ids == []
    assert view.within_hops([0], 2**64 - 1).node_ids == [0, 1, 2]
    restricted = view.select_edges([0, 1])
    assert restricted.within_hops([0], 50).node_ids == [0, 1]
    with pytest.raises(ValueError, match="seed"):
        restricted.within_hops([0, 2], 0)
    with pytest.raises(ValueError, match="direction"):
        view.within_hops([], 0, direction="sideways")
    undirected = Graph()
    undirected.add_nodes(2)
    undirected.add_edge(0, 1)
    for mode in ["outgoing", "incoming", "both"]:
        assert undirected.read_view().within_hops([1], 1, direction=mode).node_ids == [0, 1]


def test_bounded_queries_keep_effective_clocks_across_owner_mutation_drop_and_threads():
    graph = fixture()
    head = graph.current_state_id
    view = graph.read_view_at(15)
    bounded = view.within_hops([0], 1)
    assert bounded.read_time == view.read_time
    assert bounded.write_state_id is None
    index = bounded.build_index(node_properties=["score"])
    expected = index.query_nodes(all_nodes(), order_by="score")
    assert expected == [1, 0]
    assert graph.current_state_id == head
    graph.remove_node(0)
    del graph
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: index.query_nodes(all_nodes(), order_by="score"), range(10))) == [expected] * 10
    assert index.view.within_hops([0], 1).node_ids == [0, 1]


def test_direct_id_pages_match_captured_scans_and_keep_index_validation():
    graph = fixture()
    view = graph.query().select_edges([0, 1, 3])
    index = view.build_index(node_properties=["score"], edge_properties=["weight"])
    for descending in [False, True]:
        for offset in [0, 1, 10, 2**64 - 1]:
            for limit in [None, 0, 1, 3, 2**64 - 1]:
                args = dict(descending=descending, offset=offset, limit=limit)
                for predicate in ["score == null", "score == 9007199254740993"]:
                    assert index.query_nodes(predicate, **args) == view.query_nodes(predicate, **args)
                for predicate in ["weight == null", "weight == 1"]:
                    assert index.query_edges(predicate, **args) == view.query_edges(predicate, **args)
    graph.remove_node(0)
    assert index.query_nodes("score == 9007199254740993", limit=1) == [0]
    missing = view.build_index()
    with pytest.raises(ValueError, match="index"):
        missing.query_nodes("score == null", limit=0)
    assert missing.query_nodes("score == null", limit=0, allow_scan=True) == []
