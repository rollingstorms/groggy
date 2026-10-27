"""Explicit captured indexes preserve the existing query and lifetime contracts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pytest
from groggy import (
    AttributeFilter, AttrValue, DurableGraph, EdgeFilter, Graph, GraphBatch,
    GraphQueryIndex, GraphView, NodeFilter,
)


def fixture(directed=False):
    graph = Graph(directed=directed)
    batch = GraphBatch(effective_at=10)
    a = batch.add_node(x=2**53 + 1, labels=["Person", "Evidence"])
    b = batch.add_node(x=2**53, labels=["Person"])
    c = batch.add_node(x=None)
    batch.add_node()
    batch.add_edge(a, b, weight=1.5, relationship_type="knows")
    batch.add_edge(a, b, weight=2.5, relationship_type="supports")
    batch.add_edge(b, a, weight=None)
    batch.add_edge(a, a, weight=-1)
    batch.add_edge(b, c)
    graph.apply_batch(batch, 0, "fixture", "test")
    return graph


@pytest.mark.parametrize("directed", [False, True])
def test_indexed_filter_strings_objects_and_selected_graph_closure(directed):
    graph = fixture(directed)
    view = graph.read_view()
    head = graph.current_state_id
    index = view.build_index(node_properties=["x", "labels"],
                             edge_properties=["weight", "relationship_type"])
    assert isinstance(index, GraphQueryIndex)
    assert isinstance(index.view, GraphView)
    assert index.node_properties == ["labels", "x"]
    assert index.edge_properties == ["relationship_type", "weight"]
    assert index.has_topology
    for query in ["x > 9007199254740992.0", "x == null", "x != null", "x >= 9007199254740992"]:
        assert index.find_nodes(query) == view.filter_nodes(query).node_ids
        child = index.filter_nodes(query)
        assert child.node_ids == view.filter_nodes(query).node_ids
        assert child.edge_ids == view.filter_nodes(query).edge_ids
    for query in ["weight >= 1.5", 'relationship_type == "supports"']:
        assert index.find_edges(query) == view.filter_edges(query).edge_ids
    assert index.nodes_with_label("Person").node_ids == [0, 1]
    assert index.nodes_with_label("Person").edge_ids == [0, 1, 2, 3]
    endpoints = EdgeFilter.connects_nodes(0, 1)
    assert index.find_edges(endpoints) == ([0, 1] if directed else [0, 1, 2])
    for node in range(5):
        predicate = NodeFilter.has_neighbor(node)
        assert index.find_nodes(predicate) == view.filter_nodes(predicate).node_ids
    scoped = view.select_edges([1, 3]).build_index(node_properties=["x"])
    assert scoped.find_edges(endpoints) == [1]
    assert scoped.filter_nodes("x >= 0").edge_ids == [1, 3]
    assert graph.current_state_id == head


def test_index_lifecycle_is_explicit_and_missing_indexes_never_silently_scan():
    graph = fixture()
    index = graph.read_view().build_index(node_properties=["x"], topology=False)
    for query in [NodeFilter.has_attribute("unknown"),
                  NodeFilter.and_filters([NodeFilter.node_set([]), NodeFilter.has_attribute("unknown")]),
                  NodeFilter.degree_range(0, 2)]:
        with pytest.raises(ValueError, match="index"):
            index.find_nodes(query)
        assert index.find_nodes(query, allow_scan=True) == index.view.filter_nodes(query).node_ids
    with pytest.raises(ValueError, match="index"):
        index.find_edges("weight > 1")
    assert index.find_edges("weight > 1", allow_scan=True) == [0, 1]
    with pytest.raises(ValueError, match="labels index"):
        index.nodes_with_label("Person")
    with pytest.raises(ValueError):
        graph.read_view().build_index(node_properties=["x", "x"])
    with pytest.raises(ValueError):
        graph.read_view().build_index(edge_properties=[""])
    with pytest.raises(TypeError):
        index.find_nodes(object())
    assert graph.read_view().build_index(node_properties=["missing"]).find_nodes(
        NodeFilter.has_attribute("missing")) == []
    graph.set_node_attr(0, "x", -5)
    assert index.find_nodes("x > 9007199254740992.0") == [0]
    rebuilt = index.rebuild(graph.read_view())
    assert rebuilt.find_nodes("x > 9007199254740992.0") == []
    assert not rebuilt.has_topology
    assert rebuilt.node_properties == ["x"]


@pytest.mark.parametrize("value", [date(2024, 2, 29), datetime(2024, 2, 29, tzinfo=timezone.utc),
                                   timedelta(microseconds=-1), AttrValue.float64(float("nan")),
                                   AttrValue.float32(1.0), 2**63 - 1, b"\x00\xff", [1.0, 2.0]])
def test_indexed_typed_equality_and_range_preserve_existing_predicates(value):
    graph = Graph()
    graph.add_node(x=value)
    graph.add_node(x=None)
    view = graph.read_view()
    index = view.build_index(node_properties=["x"])
    for method in [AttributeFilter.equals, AttributeFilter.not_equals, AttributeFilter.greater_than,
                   AttributeFilter.less_than, AttributeFilter.greater_than_or_equal,
                   AttributeFilter.less_than_or_equal]:
        predicate = NodeFilter.attribute_filter("x", method(value))
        assert index.find_nodes(predicate) == view.filter_nodes(predicate).node_ids


def test_effective_views_indexes_and_results_remain_pinned_after_owner_close(tmp_path):
    path = tmp_path / "indexed.ggj"
    with DurableGraph.create(path) as graph:
        batch = GraphBatch(effective_at=10)
        batch.add_node(x=1)
        graph.apply_batch(batch, 0, "first", "test")
        batch = GraphBatch(effective_at=20)
        batch.set_node_attr(0, "x", 2)
        graph.apply_batch(batch, graph.current_state_id, "second", "test")
        view = graph.read_view_at(15)
        index = view.build_index(node_properties=["x"])
        child = index.filter_nodes("x == 1")
        assert child.read_time == view.read_time
        assert child.write_state_id is None
        clone = index.copy()
    assert clone.find_nodes("x == 1") == [0]
    assert child.node_attr(0, "x") == 1
    with DurableGraph.open(path) as graph:
        assert index.rebuild(graph.read_view()).find_nodes("x == 2") == [0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: clone.find_nodes("x == 1"), range(16))) == [[0]] * 16


def test_rejected_publication_does_not_change_index_graph_ids_or_journal(tmp_path):
    path = tmp_path / "rejection.ggj"
    with DurableGraph.create(path) as graph:
        graph.writer().add_node(x=1)
        index = graph.read_view().build_index(node_properties=["x"])
        head, before = graph.current_state_id, path.read_bytes()
        invalid = GraphBatch()
        created = invalid.add_node(x=2)
        invalid.add_edge(created, 9999)
        with pytest.raises(ValueError):
            graph.apply_batch(invalid, head, "invalid", "test")
        assert graph.current_state_id == head
        assert path.read_bytes() == before
        assert index.find_nodes("x == 1") == [0]
        assert index.rebuild(graph.read_view()).find_nodes("x == 2") == []
        assert graph.writer().add_node(x=3) == 1


def test_branch_restore_recovery_and_deactivation_require_explicit_new_capture(tmp_path):
    path = tmp_path / "lifecycle.ggj"
    with DurableGraph.create(path) as graph:
        graph.writer().add_node(x=1)
        base = graph.current_state_id
        graph.create_branch("original", base)
        index = graph.read_view().build_index(node_properties=["x"])
        graph.writer().set_node_attr(0, "x", 2)
        assert index.find_nodes("x == 1") == [0]
        assert index.rebuild(graph.read_view()).find_nodes("x == 2") == [0]
        graph.checkout_branch("original", graph.current_state_id)
        assert index.rebuild(graph.read_view()).find_nodes("x == 1") == [0]
        graph.writer().remove_node(0)
        assert index.rebuild(graph.read_view()).find_nodes("x == 1") == []
        assert index.find_nodes("x == 1") == [0]
    with DurableGraph.open(path) as graph:
        assert index.rebuild(graph.read_view()).find_nodes("x == 1") == []
        graph.checkout_branch("main", graph.current_state_id)
        rebuilt = index.rebuild(graph.read_view())
        assert rebuilt.find_nodes("x == 2") == [0]
        assert index.find_nodes("x == 1") == [0]


def test_index_build_shares_values_without_taking_ownership_of_mutations():
    graph = Graph(manual=True)
    graph.add_node(x=-9)
    view = graph.read_view()
    assert view.write_state_id is None
    index = view.build_index(node_properties=["x"])
    assert index.find_nodes("x == -9") == [0]
    graph.set_node_attr(0, "x", -10)
    assert index.find_nodes("x == -9") == [0]
    assert index.filter_nodes("x == -9").write_state_id is None
