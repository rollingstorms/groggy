"""Every canonical batch operation must preserve pinned scope and refreshed state."""
import math

import pytest
from groggy import DurableGraph, Graph, GraphBatch


def fixture():
    graph = Graph(directed=True)
    batch = GraphBatch(effective_at=0)
    nodes = [batch.add_node(value=index) for index in range(4)]
    batch.add_edge(nodes[0], nodes[1], weight=1)
    batch.add_edge(nodes[0], nodes[1], weight=2)
    batch.add_edge(nodes[2], nodes[3], weight=3)
    graph.apply_batch(batch, 0, "fixture", "test")
    return graph


@pytest.mark.parametrize("kind", ["node", "edge"])
def test_unset_cannot_mutate_entities_outside_the_source_view(kind):
    graph = fixture()
    source = graph.read_view().select_edges([0])
    batch = GraphBatch()
    batch.add_node()
    if kind == "node":
        batch.unset_node_attr(2, "value")
    else:
        batch.unset_edge_attr(1, "weight")
    before = graph.current_state_id
    with pytest.raises(ValueError, match="outside"):
        graph.apply_view_batch(source, batch, "outside unset", "test")
    assert graph.current_state_id == before
    assert graph.get_node_attr(2, "value") == 2
    assert graph.get_edge_attr(1, "weight") == 2
    assert graph.add_node() == 4


def test_unset_refresh_has_absence_and_can_authorize_the_next_scoped_write():
    graph = fixture()
    source = graph.read_view().select_edges([0])
    batch = GraphBatch(effective_at=10)
    batch.unset_node_attr(0, "value")
    batch.unset_edge_attr(0, "weight")
    result = graph.apply_view_batch(source, batch, "unset", "test")
    refreshed = result["view"]
    assert not refreshed.has_node_attribute(0, "value")
    assert not refreshed.has_edge_attribute(0, "weight")
    assert source.node_attr(0, "value") == 0
    assert source.edge_attr(0, "weight") == 1
    update = GraphBatch()
    update.set_node_attr(0, "after", None)
    result = graph.apply_view_batch(refreshed, update, "follow-up", "test")
    assert result["view"].has_node_attribute(0, "after")
    assert result["view"].node_attr(0, "after") is None


@pytest.mark.parametrize("edge,source,target", [(1, 0, 1), (0, 0, 2)])
def test_endpoint_revisions_cannot_escape_selected_edges_or_nodes(edge, source, target):
    graph = fixture()
    view = graph.read_view().select_edges([0])
    batch = GraphBatch(effective_at=10)
    batch.add_node()
    batch.set_edge_endpoints(edge, source, target)
    head = graph.current_state_id
    with pytest.raises(ValueError, match="outside"):
        graph.apply_view_batch(view, batch, "outside revision", "test")
    assert graph.current_state_id == head
    assert graph.edge_endpoints(edge) == (0, 1)
    assert graph.add_node() == 4


def test_revised_endpoints_and_unsets_project_before_cascading_node_removal():
    graph = fixture()
    source = graph.read_view().select_nodes([0, 1])
    batch = GraphBatch(effective_at=10)
    added = batch.add_node(value=None)
    batch.set_edge_endpoints(0, 0, added)
    batch.unset_edge_attr(0, "weight")
    batch.remove_node(1)
    result = graph.apply_view_batch(source, batch, "rewire and remove", "test")
    refreshed = result["view"]
    assert refreshed.node_ids == [0, 4]
    assert refreshed.edge_ids == [0]
    assert refreshed.edge_endpoints(0) == (0, 4) == graph.edge_endpoints(0)
    assert not refreshed.has_edge_attribute(0, "weight")
    assert source.edge_endpoints(0) == (0, 1)
    assert source.edge_ids == [0, 1]
    followup = GraphBatch()
    followup.set_edge_attr(0, "weight", 8)
    assert graph.apply_view_batch(refreshed, followup, "again", "test")["view"].edge_attr(0, "weight") == 8


def test_nan_properties_do_not_invalidate_an_unchanged_view_source():
    graph = Graph()
    graph.add_node(value=float("nan"), vector=[float("nan"), -0.0])
    view = graph.read_view()
    batch = GraphBatch()
    batch.set_node_attr(0, "new", 1)
    refreshed = graph.apply_view_batch(view, batch, "nan source", "test")["view"]
    assert math.isnan(refreshed.node_attr(0, "value"))
    assert math.isnan(view.node_attr(0, "value"))
    assert refreshed.node_attr(0, "new") == 1


@pytest.mark.parametrize("endpoints", [(1, 0), (1, 1)])
def test_endpoint_reversal_and_self_loop_match_the_accepted_graph(endpoints):
    graph = fixture()
    source = graph.read_view().select_edges([0])
    batch = GraphBatch(effective_at=10)
    batch.set_edge_endpoints(0, *endpoints)
    result = graph.apply_view_batch(source, batch, "revised", "test")
    assert result["view"].edge_endpoints(0) == endpoints
    assert graph.edge_endpoints(0) == endpoints
    assert result["view"].edge_ids == [0]
    assert source.edge_endpoints(0) == (0, 1)
    assert graph.edge_endpoints(1) == (0, 1)


def test_durable_scope_rejection_preserves_bytes_and_complete_projection_survives_reopen(tmp_path):
    path = tmp_path / "view-complete.ggj"
    with DurableGraph.create_from_graph(path, fixture()) as graph:
        source = graph.read_view().select_nodes([0, 1])
        head = graph.current_state_id
        before = path.read_bytes()
        bad = GraphBatch()
        bad.unset_edge_attr(2, "weight")
        with pytest.raises(ValueError, match="outside"):
            graph.apply_view_batch(source, bad, "outside unset", "test")
        assert path.read_bytes() == before
        assert graph.current_state_id == head
        batch = GraphBatch(effective_at=10)
        added = batch.add_node()
        batch.set_edge_endpoints(0, 0, added)
        batch.unset_edge_attr(0, "weight")
        batch.remove_node(1)
        result = graph.apply_view_batch(source, batch, "mixed", "test")
        refreshed = result["view"]
        assert result["added_nodes"] == [4]
        assert refreshed.edge_ids == [0]
        assert refreshed.edge_endpoints(0) == (0, 4)
        assert not refreshed.has_edge_attribute(0, "weight")
        followup = GraphBatch()
        followup.set_node_attr(4, "after", 9)
        assert graph.apply_view_batch(refreshed, followup, "again", "test")["view"].node_attr(4, "after") == 9
        graph.verify_storage()
    assert source.edge_endpoints(0) == (0, 1)
    assert refreshed.edge_endpoints(0) == (0, 4)
    with DurableGraph.open(path) as graph:
        assert graph.read_view().edge_endpoints(0) == (0, 4)
        assert graph.get_edge_attr(0, "weight") is None
        assert graph.get_node_attr(4, "after") == 9
        assert graph.read_view().edge_endpoints(2) == (2, 3)
        graph.verify_storage()


def test_refreshed_view_keeps_meta_source_catalog_after_owner_drop():
    graph = fixture()
    group = graph.retain_group(graph.read_view(), "source")
    meta = graph.create_meta_node(group)
    view = graph.read_view().select_nodes([meta])
    batch = GraphBatch()
    batch.set_node_attr(meta, "caption", "changed")
    refreshed = graph.apply_view_batch(view, batch, "meta property", "test")["view"]
    assert refreshed.meta_subgraph(meta).node_ids == [0, 1, 2, 3]
    del graph
    assert view.meta_subgraph(meta).edge_ids == [0, 1, 2]
    assert refreshed.nodes[meta].meta_subgraph().edge_endpoints(0) == (0, 1)
