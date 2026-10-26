"""Absence, null, coalesced history and durable recovery share one mutation contract."""
from datetime import date, datetime, timedelta, timezone

import pytest
from groggy import AttrValue, DurableGraph, Graph, GraphBatch


def initial(graph):
    batch = GraphBatch(effective_at=10)
    a = batch.add_node(v=None, labels=["Entity"], name="Ada")
    b = batch.add_node()
    batch.add_edge(a, b, v=[1.25, 2.5])
    return graph.apply_batch(batch, 0, "initial", "test")["state_id"]


def test_mixed_unset_null_timestamps_and_pinned_tables():
    graph = Graph()
    first = initial(graph)
    old = graph.read_view_at(10)
    table = old.nodes_table().base_table()
    batch = GraphBatch(effective_at=20)
    batch.unset_node_attr(0, "v")
    batch.unset_edge_attr(0, "v", effective_at=30)
    batch.set_node_attr(1, "v", None)
    second = graph.apply_batch(batch, first, "unset", "test")["state_id"]
    assert old.has_node_attribute(0, "v") and old.node_attr(0, "v") is None
    assert not graph.read_view_at(20).has_node_attribute(0, "v")
    assert graph.read_view_at(20).has_node_attribute(1, "v")
    assert graph.read_view_at(29).edge_attr(0, "v") == [1.25, 2.5]
    assert not graph.read_view_at(30).has_edge_attribute(0, "v")
    assert table.source_views()[0].has_node_attribute(0, "v")
    history = graph.node_attr_history(0, "v", first, second)
    assert history[0][1].value is None  # Explicit Null is an AttrValue object.
    assert history[1] == (second, None)  # An unset is an absent value.
    graph.writer().set_node_attr(0, "v", 99, effective_at=40)
    assert not graph.read_view_at(39).has_node_attribute(0, "v")
    assert graph.read_view_at(40).node_attr(0, "v") == 99


def test_manual_bulk_scalar_and_unset_preserve_final_value_and_history():
    graph = Graph(manual=True)
    node = graph.add_node(v=1)
    first = graph.commit("initial", "test")
    graph.set_node_attrs({"v": {node: 2}})
    graph.set_node_attrs({"v": {node: 3}})
    graph.unset_node_attr(node, "v")
    graph.set_node_attr(node, "v", None)
    final = graph.commit("coalesced", "test")
    assert graph.get_node_attr(node, "v") is None
    assert graph.has_node_attribute(node, "v")
    history = graph.node_attr_history(node, "v", first, final)
    assert len(history) == 2 and history[-1][1].value is None
    graph.unset_node_attrs({"v": [node]})
    last = graph.commit("unset", "test")
    assert not graph.has_node_attribute(node, "v")
    assert graph.node_attr_history(node, "v", last, last) == [(last, None)]


@pytest.mark.parametrize("field", ["entity_type", "external_id"])
def test_system_managed_properties_cannot_be_unset(field):
    graph = Graph()
    node = graph.add_node()
    head = graph.current_state_id
    with pytest.raises(ValueError, match="system-managed"):
        graph.unset_node_attr(node, field)
    assert graph.current_state_id == head


def test_meta_source_cannot_be_unset_but_user_reference_can():
    graph = Graph()
    initial(graph)
    group = graph.retain_group(graph.read_view())
    meta = graph.create_meta_node(group)
    old = graph.read_view()
    with pytest.raises(ValueError, match="requires its source"):
        graph.unset_node_attr(meta, "contains_subgraph")
    graph.writer().set_node_attr(0, "group", AttrValue.subgraph_ref(group))
    saved = graph.read_view()
    graph.unset_node_attr(0, "group")
    assert not graph.read_view().has_node_attribute(0, "group")
    assert saved.has_node_attribute(0, "group")
    assert old.meta_subgraph(meta).node_attr(0, "name") == "Ada"


@pytest.mark.parametrize("description", [
    {"unset_node_attrs": {"v": [0, 999]}},
    {"unset_node_attrs": {"v": [0, 0]}},
    {"unset_node_attrs": {"v": [0]}, "node_attrs": {"v": [(0, 2)]}},
    {"unset_edge_attrs": {"v": [0, 999]}},
])
def test_failed_mixed_batch_preserves_all_state_and_id_allocation(description):
    graph = Graph()
    head = initial(graph)
    description["add_nodes"] = 1
    with pytest.raises(ValueError):
        graph.apply_batch(description, head, "bad", "test")
    assert graph.current_state_id == head
    assert graph.has_node_attribute(0, "v")
    assert not graph.has_uncommitted_changes()
    assert graph.add_node() == 2


def test_durable_schema_rejects_unset_before_journaling_and_valid_removal_recovers(tmp_path):
    path = tmp_path / "history.ggj"
    backup = tmp_path / "backup.ggj"
    schema = {"node_labels": {"Entity": {"name": {"type": "text"}}}}
    with DurableGraph.create(path, schema=schema) as graph:
        initial(graph)
        before = path.read_bytes()
        writer = graph.writer()
        with pytest.raises(ValueError, match="required property is absent"):
            writer.unset_node_attr(0, "name")
        assert path.read_bytes() == before
        old = graph.read_view_at(10)
        writer.unset_node_attrs({"v": [0]}, effective_at=20)
        writer.unset_edge_attr(0, "v", effective_at=30)
        head = writer.state_id
        metadata = graph.publication_metadata(head)
        graph.backup(backup)
    for location in [path, backup]:
        with DurableGraph.open(location) as graph:
            assert graph.current_state_id == head
            assert graph.publication_metadata(head) == metadata
            assert graph.read_view_at(10).has_node_attribute(0, "v")
            assert not graph.read_view_at(20).has_node_attribute(0, "v")
            assert not graph.read_view_at(30).has_edge_attribute(0, "v")
    assert old.has_node_attribute(0, "v")


def test_unset_all_inline_type_families_recover_without_resurrecting_values(tmp_path):
    values = [None, 2**60, 1.25, True, "index_0", b"\x00\xff", [1, 2],
              [1.25, 2.5], [True, False], ["a", "b"], {"nested": [1, None]},
              date(2026, 10, 25), datetime(2026, 10, 25, tzinfo=timezone.utc),
              timedelta(days=-1, microseconds=7)]
    path = tmp_path / "types.ggj"
    with DurableGraph.create(path) as graph:
        batch = GraphBatch(effective_at=10)
        a = batch.add_node(**{f"p{i}": value for i, value in enumerate(values)})
        b = batch.add_node()
        batch.add_edge(a, b, **{f"p{i}": value for i, value in enumerate(values)})
        graph.apply_batch(batch, 0, "typed", "test")
        batch = GraphBatch(effective_at=20)
        for i in range(len(values)):
            batch.unset_node_attr(0, f"p{i}")
            batch.unset_edge_attr(0, f"p{i}")
        graph.apply_batch(batch, graph.current_state_id, "unset", "test")
    with DurableGraph.open(path) as graph:
        old, current = graph.read_view_at(10), graph.read_view_at(20)
        for i, value in enumerate(values):
            assert old.node_attr(0, f"p{i}") == value
            assert old.edge_attr(0, f"p{i}") == value
            assert not current.has_node_attribute(0, f"p{i}")
            assert not current.has_edge_attribute(0, f"p{i}")


def test_durable_unset_group_reference_retains_historical_source(tmp_path):
    source = Graph()
    initial(source)
    path = tmp_path / "retained.ggj"
    with DurableGraph.create(path) as graph:
        group = graph.retain_group(source.read_view_at(10))
        meta = graph.create_meta_node(group, effective_at=10)
        writer = graph.writer()
        writer.set_node_attr(meta, "other_source", AttrValue.subgraph_ref(group), effective_at=15)
        first = writer.state_id
        writer.unset_node_attr(meta, "other_source", effective_at=20)
        second = writer.state_id
    del source
    with DurableGraph.open(path) as graph:
        assert graph.read_view_at(15).has_node_attribute(meta, "other_source")
        assert not graph.read_view_at(20).has_node_attribute(meta, "other_source")
        assert graph.read_view_at(20).meta_subgraph(meta).node_attr(0, "name") == "Ada"
        assert graph.snapshot_at_commit(first).node_attr(meta, "other_source") is not None
        assert graph.snapshot_at_commit(second).node_attr(meta, "other_source") is None
