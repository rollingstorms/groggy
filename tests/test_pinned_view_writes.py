"""Immutable source pins authorize one scoped checked batch and a refreshed view."""
import pytest
from groggy import DurableGraph, Graph, GraphBatch


def fixture(directed=False, manual=False):
    graph = Graph(directed=directed, manual=manual)
    graph.add_nodes([{"value": 1}, {}, {}, {}])
    graph.add_edges([(0, 1), (0, 1), (1, 1), (2, 3)])
    graph.commit("fixture", "test")
    return graph


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("manual", [False, True])
def test_mixed_view_batch_returns_new_snapshot_and_preserves_exact_edge_scope(directed, manual):
    # Explicit topology times need explicit endpoint births, in either write mode.
    graph = Graph(directed=directed, manual=manual)
    initial = GraphBatch(effective_at=0)
    nodes = [initial.add_node(value=1), initial.add_node(), initial.add_node(), initial.add_node()]
    for a, b in [(0, 1), (0, 1), (1, 1), (2, 3)]:
        initial.add_edge(nodes[a], nodes[b])
    graph.apply_batch(initial, 0, "timed fixture", "test")
    source = graph.read_view().select_edges([0, 2]).copy()
    before = source.write_state_id
    batch = GraphBatch(effective_at=20)
    added = batch.add_node(value=None, labels=["z", "a"])
    batch.set_node_external_id(added, "new")
    new_edge = batch.add_edge(1, added, weight=2.5)
    batch.set_node_attr(0, "value", 1.0000000000000002, effective_at=-10)
    batch.set_edge_attr(0, "weight", 7.5)
    batch.remove_edge(2)
    result = graph.apply_view_batch(source, batch, "mixed", "test")
    view = result["view"]
    assert result["state_id"] == before + 1 == view.write_state_id
    assert result["added_nodes"] == [4]
    assert result["added_edges"] == [4]
    assert view.node_ids == [0, 1, 4]
    assert view.edge_ids == [0, 4]
    assert graph.contains_edge(1)
    assert source.node_attr(0, "value") == 1
    assert source.contains_edge(2)
    assert view.node_attr(4, "labels") == ["a", "z"]
    assert view.has_node_attribute(4, "value")
    assert view.node_attr(4, "value") is None
    assert graph.node_by_external_id("new") == 4
    assert not graph.has_uncommitted_changes()
    assert graph.automatic_publication == (not manual)
    clocks = graph.publication_metadata(result["state_id"])["change_times"]
    assert any(row["effective_at"] == -10 for row in clocks)
    update = GraphBatch()
    update.set_node_attr(4, "value", 9)
    next_result = graph.apply_view_batch(view, update, "next", "test")
    assert next_result["view"].node_attr(4, "value") == 9
    assert view.node_attr(4, "value") is None
    with pytest.raises(ValueError, match="Stale"):
        graph.apply_view_batch(source, update, "stale", "test")


@pytest.mark.parametrize("operation", ["node", "edge", "remove_node", "remove_edge", "endpoint", "identity", "cascade"])
def test_scope_rejection_preserves_ids_history_and_outside_entities(operation):
    graph = fixture()
    view = graph.read_view().select_edges([0])
    head = graph.current_state_id
    batch = GraphBatch()
    batch.add_node()
    if operation == "node":
        batch.set_node_attr(2, "value", 7)
    elif operation == "edge":
        batch.set_edge_attr(1, "value", 7)
    elif operation == "remove_node":
        batch.remove_node(2)
    elif operation == "remove_edge":
        batch.remove_edge(1)
    elif operation == "endpoint":
        batch.add_edge(0, 2)
    elif operation == "identity":
        batch.set_node_external_id(2, "outside")
    else:
        batch.remove_node(0)  # excluded parallel edge 1 would also be removed
    with pytest.raises(ValueError, match="outside"):
        graph.apply_view_batch(view, batch, "bad", "test")
    assert graph.current_state_id == head
    assert not graph.has_uncommitted_changes()
    assert len(graph.edge_ids) == 4
    assert graph.add_node() == 4


def test_source_graph_branch_dirty_and_standalone_provenance_is_enforced():
    graph = fixture()
    source = graph.read_view()
    batch = GraphBatch()
    batch.set_node_attr(0, "value", 3)
    foreign = fixture()
    with pytest.raises(ValueError, match="different graph"):
        foreign.apply_view_batch(source, batch, "foreign", "test")
    graph.create_branch("other")
    graph.checkout_branch("other")
    with pytest.raises(ValueError, match="different branch"):
        graph.apply_view_batch(source, batch, "branch", "test")
    graph.checkout_branch("main")
    manual = Graph(manual=True)
    manual.add_node(value=1)
    dirty = manual.read_view()
    assert dirty.write_state_id is None
    manual.commit("publish", "test")
    with pytest.raises(ValueError, match="read-only"):
        manual.apply_view_batch(dirty, batch, "dirty", "test")


def test_complete_cascade_and_filtered_pin_derived_from_current_state():
    graph = fixture()
    source = graph.view().pin_view().select_nodes([0, 1])
    batch = GraphBatch()
    batch.remove_node(0)
    result = graph.apply_view_batch(source, batch, "delete", "test")
    assert result["view"].node_ids == [1]
    assert result["view"].edge_ids == [2]
    assert source.node_ids == [0, 1]
    assert set(graph.edge_ids) == {2, 3}


def test_schema_and_conversion_failures_do_not_publish_or_consume_ids():
    graph = fixture()
    source = graph.read_view()
    schema = {"node_labels": {"Person": {"name": {"type": "text"}}}}
    batch = GraphBatch()
    batch.add_node(labels=["Person"])
    with pytest.raises(ValueError, match="name"):
        graph.apply_view_batch(source, batch, "schema", "test", schema=schema)
    invalid = GraphBatch()
    invalid.add_node(payload=object())
    with pytest.raises((ValueError, TypeError)):
        graph.apply_view_batch(source, invalid, "conversion", "test")
    assert graph.current_state_id == source.write_state_id
    assert graph.add_node() == 4


def test_durable_view_publication_recovery_and_old_owner_rejection(tmp_path):
    path = tmp_path / "views.ggj"
    with DurableGraph.create(path) as graph:
        initial = GraphBatch()
        initial.add_node(value=1)
        initial.add_node()
        graph.apply_batch(initial, 0, "init", "test")
        source = graph.read_view().select_nodes([0])
        bad = GraphBatch()
        bad.set_node_attr(1, "value", 7)
        before = path.read_bytes()
        with pytest.raises(ValueError, match="outside"):
            graph.apply_view_batch(source, bad, "bad", "test")
        assert path.read_bytes() == before
        update = GraphBatch()
        update.set_node_attr(0, "value", 7, effective_at=5)
        result = graph.apply_view_batch(source, update, "accepted", "test")
        new_view = result["view"]
        assert new_view.node_attr(0, "value") == 7
        assert source.node_attr(0, "value") == 1
    assert new_view.node_attr(0, "value") == 7
    with DurableGraph.open(path) as graph:
        assert graph.get_node_attr(0, "value") == 7
        before = path.read_bytes()
        with pytest.raises(ValueError, match="different graph"):
            graph.apply_view_batch(new_view, update, "reopened", "test")
        assert path.read_bytes() == before
        result = graph.apply_view_batch(graph.read_view(), update, "fresh", "test")
        assert result["state_id"] == 3


def test_durable_installed_schema_cannot_be_bypassed_by_a_view_batch(tmp_path):
    path = tmp_path / "schema.ggj"
    schema = {"node_labels": {"Person": {"name": {"type": "text"}}}}
    with DurableGraph.create(path, schema=schema) as graph:
        source = graph.read_view()
        batch = GraphBatch()
        batch.add_node(labels=["Person"])
        before = path.read_bytes()
        for extra in [None, {}]:
            with pytest.raises(ValueError, match="name"):
                graph.apply_view_batch(source, batch, "bad", "test", schema=extra)
            assert graph.current_state_id == 0
            assert path.read_bytes() == before
