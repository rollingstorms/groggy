"""A compound collapse is one accepted mutation, including properties and edges."""
import pytest
from groggy import Graph


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("strategy", ["extract", "collapse"])
def test_collapse_accepts_complete_result_in_one_version(manual, strategy):
    graph = Graph(manual=manual)
    a, b, outside = graph.add_nodes([{"x": 2}, {"x": 4}, {}])
    internal = graph.add_edge(a, b)
    graph.add_edges([(a, outside, {"w": 2}), (b, outside, {"w": 4})])
    graph.commit("before", "test")
    before = graph.current_state_id
    frozen = graph.read_view()
    meta = graph.induced_subgraph([a, b]).collapse(
        node_aggs={"total": ("sum", "x")}, edge_aggs={"w": "sum"},
        node_strategy=strategy, allow_missing_attributes=False,
    )
    assert graph.current_state_id == before + (not manual)
    assert graph.get_node_attr(meta.id, "total") == 6
    if not manual:
        metadata = graph.publication_metadata(graph.current_state_id)
        assert metadata["change_times"]
        assert all(row["effective_at"] == metadata["recorded_at"] for row in metadata["change_times"])
    boundary = [e for e in graph.edge_ids if meta.id in graph.edge_endpoints(e)]
    assert len(boundary) == 1
    assert graph.get_edge_attr(boundary[0], "w") == 6
    assert graph.get_edge_attr(boundary[0], "edge_count") == 2
    assert set(meta.expand().node_ids) == {a, b}
    assert set(meta.expand().edge_ids) == {internal}
    assert frozen.node_attr(a, "x") == 2
    if strategy == "collapse":
        assert set(graph.node_ids) == {meta.id, outside}
    if manual:
        assert graph.has_uncommitted_changes()
        graph.commit("after", "test")
    assert not graph.has_uncommitted_changes()
    snapshot = graph.snapshot_at_commit(before + 1)
    assert snapshot.node_attr(meta.id, "total").value == 6
    assert snapshot.edge_attr(boundary[0], "w").value == 6


@pytest.mark.parametrize("case", ["missing", "invalid_node", "invalid_edge", "numeric_edge", "reserved"])
def test_failed_collapse_preserves_state_and_next_ids(case):
    graph = Graph()
    a, b, outside = graph.add_nodes(3)
    graph.add_edges([(a, outside, {"w": "one"}), (b, outside, {"w": "two"})])
    selection = graph.induced_subgraph([a, b])
    before = graph.current_state_id
    kwargs = {
        "missing": {"node_aggs": {"x": "sum"}, "allow_missing_attributes": False},
        "invalid_node": {"node_aggs": {"x": "unknown"}},
        "invalid_edge": {"edge_aggs": {"w": "unknown"}},
        "numeric_edge": {"edge_aggs": {"w": "sum"}},
        "reserved": {"node_aggs": {"entity_type": "count"}},
    }[case]
    with pytest.raises(RuntimeError):
        selection.collapse(node_strategy="collapse", **kwargs)
    assert graph.current_state_id == before
    assert set(graph.node_ids) == {a, b, outside}
    assert set(graph.edge_ids) == {0, 1}
    assert not graph.has_uncommitted_changes()
    assert graph.add_node() == 3
    assert graph.add_edge(a, b) == 2


@pytest.mark.parametrize("edge_strategy,expected", [("aggregate", 2), ("keep_external", 3)])
def test_directed_boundary_edges_keep_direction_and_parallel_copy(edge_strategy, expected):
    graph = Graph(directed=True)
    a, b, outside = graph.add_nodes(3)
    graph.add_edges([(outside, a, {"w": 1}), (a, outside, {"w": 2}), (b, outside, {"w": 3})])
    before = graph.current_state_id
    meta = graph.induced_subgraph([a, b]).collapse(
        edge_strategy=edge_strategy, edge_aggs={"w": "sum"}, node_strategy="collapse",
    )
    assert graph.current_state_id == before + 1
    assert len(graph.edge_ids) == expected
    incoming = [e for e in graph.edge_ids if graph.edge_endpoints(e) == (outside, meta.id)]
    outgoing = [e for e in graph.edge_ids if graph.edge_endpoints(e) == (meta.id, outside)]
    assert len(incoming) == 1
    assert graph.get_edge_attr(incoming[0], "w") == 1
    assert sum(graph.get_edge_attr(e, "w") for e in outgoing) == 5


def test_lenient_missing_and_compact_text_aggregation():
    graph = Graph()
    a, b, outside = graph.add_nodes(3)
    graph.add_edges([(a, outside, {"tag": "b"}), (b, outside, {"tag": "a"})])
    meta = graph.induced_subgraph([a, b]).collapse(
        node_aggs={"missing": "sum"}, edge_aggs={"tag": "concat"},
    )
    assert graph.get_node_attr(meta.id, "missing") == 0
    edge = next(e for e in graph.edge_ids if meta.id in graph.edge_endpoints(e))
    assert graph.get_edge_attr(edge, "tag") == "b,a"
