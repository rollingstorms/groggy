"""Readonly captured entity IDs, aligned columns and selected masks share one pin."""
from concurrent.futures import ThreadPoolExecutor

import groggy as gr
import pytest


def fixture():
    graph = gr.Graph()
    graph.add_nodes(5)
    graph.remove_node(1)
    for source, target in [(0, 2), (0, 2), (2, 3), (0, 0)]:
        graph.add_edge(source, target)
    graph.set_node_attr(0, "score", 2.5)
    graph.set_node_attr(2, "score", None)
    graph.set_node_attr(4, "score", 2**53 + 1)
    graph.set_edge_attr(0, "weight", 2.0)
    return graph, graph.subgraph_from_edges([0, 2]).pin_view()


def test_entity_ids_and_sorted_positions_are_explicit_on_captured_accessors():
    graph, selected = fixture()
    view = graph.read_view()
    nodes = view.nodes
    assert isinstance(nodes, gr.GraphViewEntities)
    assert nodes.ids.to_list() == [0, 2, 3, 4]
    assert nodes[2].id == 2
    assert nodes.at(1).id == 2
    assert nodes.at(-1).id == 4
    assert 2 in nodes and 1 not in nodes
    assert nodes.by_id(4).score == 2**53 + 1
    assert nodes[4].kind == "node"
    assert selected.edges[0].kind == "edge"
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        nodes[1]
    with pytest.raises(IndexError):
        nodes.at(99)
    assert nodes[::-2].node_ids == [2, 4]  # graph membership has stable ID order
    assert nodes[0:0].node_count() == 0
    assert nodes[[4, 0, 4]].node_ids == [0, 4]


def test_columns_and_presence_keep_missing_distinct_from_present_null():
    graph, _ = fixture()
    nodes = graph.read_view().nodes
    assert nodes["score"].to_list() == [2.5, None, None, 2**53 + 1]
    assert nodes.score.to_list() == nodes.column("score").to_list()
    assert nodes.presence("score").to_list() == [True, True, False, True]
    assert nodes["absent"].to_list() == [None] * 4
    assert "score" in nodes.attribute_names()
    assert "score" in nodes[2]
    assert "score" not in nodes[3]
    assert nodes[2].to_dict()["score"] is None
    with pytest.raises(KeyError):
        nodes[3]["score"]
    with pytest.raises(AttributeError):
        _ = nodes[3].score
    with pytest.raises(AttributeError):
        _ = nodes.no_such_column
    values = nodes["score"].to_list()
    values[0] = -1
    assert nodes[0].score == 2.5


def test_node_handles_and_nested_selections_use_exact_parent_edge_scope():
    graph, selected = fixture()
    assert selected.nodes[0].degree == 1
    assert selected.nodes[0].neighbors == [2]
    assert selected.edges[0].endpoints == (0, 2)
    assert selected.edges[0].source == 0
    assert selected.edges[0].target == 2
    assert selected.nodes[[0, 2]].edge_ids == [0]
    assert selected.nodes[:].edge_ids == [0, 2]
    assert selected.edges[:].edge_ids == [0, 2]
    assert selected.nodes.filter("score >= 2").node_ids == [0]
    assert selected.edges.filter("weight == 2.0").edge_ids == [0]
    with pytest.raises((ValueError, RuntimeError)):
        selected.edges[1]
    assert graph.edge_count() == 4


def test_exact_boolean_masks_allow_empty_results_and_reject_bad_shape_or_types():
    graph, _ = fixture()
    nodes = graph.read_view().nodes
    assert nodes[nodes.presence("score")].node_ids == [0, 2, 4]
    assert nodes[gr.BaseArray([False] * 4)].node_count() == 0
    for mask in [gr.BaseArray([True]), gr.BaseArray([True, None, False, True]),
                 gr.BaseArray([1, 0, 0, 1])]:
        with pytest.raises((ValueError, RuntimeError)):
            _ = nodes[mask]


def test_captured_node_all_preserves_isolates_while_edge_all_uses_endpoints():
    graph, _ = fixture()
    view = graph.read_view()
    assert view.nodes.all().node_ids == [0, 2, 3, 4]
    assert view.edges.all().node_ids == [0, 2, 3]
    assert view.select_nodes([4]).edges.all().node_count() == 0


def test_handles_columns_and_iterators_outlive_mutation_owner_and_worker_transfer():
    graph, view = fixture()
    nodes, edges = view.nodes, view.edges
    node, edge = nodes[0], edges[0]
    iterator = iter(nodes)
    assert next(iterator).id == 0
    graph.set_edge_endpoints(0, 4, 3)
    graph.set_node_attr(0, "score", -1)
    graph.remove_node(2)
    del graph, view
    assert [entity.id for entity in iterator] == [2, 3]
    assert node.score == 2.5 and node.neighbors == [2]
    assert edge.endpoints == (0, 2)
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(lambda _: nodes["score"].to_list(), range(4))) == [[2.5, None, None]] * 4


def test_readonly_accessors_cannot_publish_and_explicit_writeback_keeps_the_capture():
    graph, view = fixture()
    node = view.nodes[0]
    before = graph.current_state_id
    with pytest.raises((AttributeError, TypeError)):
        node["score"] = 9
    with pytest.raises((AttributeError, TypeError)):
        node.score = 9
    with pytest.raises(AttributeError):
        view.nodes.set_attrs({"score": 9})
    assert graph.current_state_id == before
    graph.set_node_attr(0, "score", 9)
    assert node.score == 2.5


@pytest.mark.parametrize("value", [True, -1, 1.5, "not_an_id", 2**100])
def test_explicit_entity_ids_reject_invalid_values(value):
    graph, _ = fixture()
    with pytest.raises((ValueError, TypeError, RuntimeError, OverflowError)):
        graph.read_view().nodes.by_id(value)


@pytest.mark.parametrize("value", [True, 1.5, "not_a_position", 2**100])
def test_positions_reject_boolean_fractional_text_and_overflow(value):
    graph, _ = fixture()
    with pytest.raises((ValueError, TypeError, RuntimeError, OverflowError)):
        graph.read_view().nodes.at(value)


def test_effective_time_columns_masks_and_entity_properties_keep_time_coordinates():
    graph = gr.Graph()
    batch = gr.GraphBatch(effective_at=10)
    a = batch.add_node(score=1)
    b = batch.add_node(score=None)
    batch.add_edge(a, b, weight=2.0)
    graph.apply_batch(batch, graph.current_state_id, "initial", "test")
    batch = gr.GraphBatch(effective_at=20)
    batch.set_node_attr(0, "score", 3)
    graph.apply_batch(batch, graph.current_state_id, "later", "test")
    view = graph.read_view_at(15)
    assert view.nodes["score"].to_list() == [1, None]
    assert view.nodes[0].score == 1
    assert view.nodes[view.nodes.presence("score")].read_time == view.read_time
    assert view.edges.all().read_time == view.read_time


def test_meta_node_entity_resolves_retained_source_after_owner_deletion():
    graph = gr.Graph()
    graph.add_nodes(2)
    graph.add_edge(0, 1)
    source = graph.read_view()
    group = graph.retain_group(source, "old")
    meta = graph.create_meta_node(group)
    node = graph.read_view().nodes[meta]
    assert node.meta_subgraph().node_ids == [0, 1]
    graph.remove_node(0)
    del graph
    assert node.meta_subgraph().edge_ids == [0]
