"""The query entry point is the existing immutable GraphView, with shared paging."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from groggy import DurableGraph, EdgeFilter, Graph, GraphBatch, GraphView, NodeFilter


def fixture():
    graph = Graph()
    for props in [{"score": 2**53 + 1}, {"score": float(2**53)}, {"score": None}, {}]:
        graph.add_node(**props)
    for a, b, weight in [(0, 1, 3), (0, 1, 1), (1, 1, 2)]:
        graph.add_edge(a, b, weight=weight)
    return graph


def test_query_entry_point_captures_composable_values_topology_and_pages():
    graph = fixture()
    before = graph.current_state_id
    query = graph.query()
    assert isinstance(query, GraphView)
    assert query.query_nodes("NOT score > 0", order_by="score") == [2, 3]
    assert query.query_nodes(NodeFilter.node_set([0, 1, 2, 3]), order_by="score") == [1, 0, 2, 3]
    assert query.query_nodes("score > 9007199254740992.0") == [0]
    assert query.query_edges("weight > 0", order_by="weight", offset=1, limit=1) == [2]
    selected = query.filter_nodes(NodeFilter.node_set([0, 1]))
    assert selected.node_ids == [0, 1] and selected.edge_ids == [0, 1, 2]
    assert query.node_ids == [0, 1, 2, 3]
    assert graph.current_state_id == before
    graph.remove_node(0)
    graph.set_node_attr(1, "score", -1)
    assert query.query_nodes("score > 0") == [0, 1]
    assert selected.query_edges("weight > 0", order_by="weight") == [1, 2, 0]
    assert graph.query().query_nodes("score > 0") == []
    assert query.query_nodes("never_assigned == null") == []


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("offset,limit", [(0, None), (1, 2), (0, 0), (2**63, 2**63)])
def test_captured_scan_pages_match_explicit_indexes(descending, offset, limit):
    query = fixture().query().select_edges([0, 2])
    index = query.build_index(node_properties=["score"], edge_properties=["weight"])
    args = dict(descending=descending, offset=offset, limit=limit)
    assert query.query_nodes("score > 0", order_by="score", **args) == \
        index.query_nodes("score > 0", order_by="score", **args)
    assert query.query_edges("weight > 0", order_by="weight", **args) == \
        index.query_edges("weight > 0", order_by="weight", **args)
    assert query.query_nodes("NOT never_assigned == null", **args) == \
        sorted(query.node_ids, reverse=descending)[offset:][:limit]


def test_connected_query_capture_is_immutable_and_stale_selections_reject():
    graph = fixture()
    connected = graph.filter_nodes("score > 0")
    query = connected.query()
    graph.set_node_attr(0, "score", -5)
    assert query.query_nodes("score > 0") == [0, 1]
    assert connected.query().query_nodes("score > 0") == [1]
    graph.remove_node(1)
    with pytest.raises(ValueError, match="outside|inactive|View"):
        connected.query()
    assert query.query_nodes("score > 0") == [0, 1]


def test_historical_query_pages_preserve_effective_cutoffs_and_source_pins():
    graph = Graph()
    batch = GraphBatch(effective_at=10)
    batch.add_node(score=1)
    first = graph.writer().apply_batch(batch)["state_id"]
    batch = GraphBatch(effective_at=20)
    batch.set_node_attr(0, "score", 2)
    graph.writer().apply_batch(batch)
    cutoff = graph.publication_metadata(first)["recorded_at"]
    historical = graph.read_view_at(25, recorded_by=cutoff)
    ids = historical.query_nodes("score > 0", order_by="score")
    selected = historical.select_nodes(ids)
    assert selected.node_attr(0, "score") == 1
    assert selected.read_time == historical.read_time and selected.write_state_id is None
    assert graph.query().node_attr(0, "score") == 2


def test_durable_query_capture_survives_owner_close_and_reopen_without_writing(tmp_path):
    path = tmp_path / "query.ggj"
    with DurableGraph.create(path) as graph:
        batch = GraphBatch(effective_at=10)
        batch.add_node(score=1)
        graph.writer().apply_batch(batch)
        before, head = path.read_bytes(), graph.current_state_id
        query = graph.query()
        assert path.read_bytes() == before and graph.current_state_id == head
    with DurableGraph.open(path) as graph:
        graph.writer().set_node_attr(0, "score", 2)
        assert graph.query().query_nodes("score == 2") == [0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: query.query_nodes("score == 1"), range(8))) == [[0]] * 8


@pytest.mark.parametrize("bad", [True, -1, 1.5, "2", 2**100])
def test_query_pagination_controls_reject_invalid_types_and_ranges(bad):
    query = fixture().query()
    for getter in [query.query_nodes, query.query_edges]:
        predicate = "score > 0" if getter.__name__ == "query_nodes" else "weight > 0"
        for key in ["offset", "limit"]:
            with pytest.raises((TypeError, OverflowError)):
                getter(predicate, **{key: bad})


@pytest.mark.parametrize("value", [float("nan"), [1, 2], "text"])
def test_query_ordering_checks_every_matching_row_even_for_zero_limit(value):
    graph = fixture()
    graph.set_node_attr(0, "score", value)
    query = graph.query()
    with pytest.raises(ValueError, match="order|scalar|domain"):
        query.query_nodes(NodeFilter.node_set([0, 1]), order_by="score", limit=0)
    assert query.query_nodes(NodeFilter.node_set([1]), order_by="score", limit=0) == []
    with pytest.raises(ValueError, match="nonempty"):
        query.query_edges(EdgeFilter.and_filters([]), order_by="", limit=0)
