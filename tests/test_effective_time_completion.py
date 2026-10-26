"""Effective reads include frozen groups, explicit legacy clocks and retained pins."""
import pytest
from groggy import AttrValue, DurableGraph, Graph, GraphBatch


def initial(graph):
    batch = GraphBatch(effective_at=10)
    a = batch.add_node(name="original", embedding=[1.25, 2.5], payload=b"\x00\xff")
    b = batch.add_node()
    batch.add_edge(a, b)
    batch.add_edge(a, b)
    graph.apply_batch(batch, 0, "initial", "test")


def test_frozen_group_revisions_and_nested_views_survive_later_mutations_and_owner_drop():
    graph = Graph()
    initial(graph)
    group = graph.retain_group(graph.read_view(), kind="inner")
    meta = graph.create_meta_node(group, effective_at=20)
    old = graph.read_view_at(25)
    assert old.meta_subgraph(meta).node_attr(0, "name") == "original"
    graph.writer().set_node_attr(0, "name", "changed", effective_at=30)
    group2 = graph.retain_group(graph.read_view().select_nodes([0, 1]), kind="inner")
    assert group2 != group
    graph.writer().set_node_attr(meta, "contains_subgraph", AttrValue.subgraph_ref(group2), effective_at=40)
    assert graph.read_view_at(39).meta_subgraph(meta).node_attr(0, "name") == "original"
    revised = graph.read_view_at(40)
    assert revised.meta_subgraph(meta).node_attr(0, "name") == "changed"
    outer_group = graph.retain_group(revised.select_nodes([meta]), kind="outer")
    outer = graph.create_meta_node(outer_group, effective_at=50)
    frozen = graph.read_view_at(50)
    assert frozen.meta_subgraph(outer).meta_subgraph(meta).edge_ids == [0, 1]
    table = frozen.nodes_table().base_table()
    graph.writer().remove_node(0, effective_at=60)
    del graph
    assert frozen.meta_subgraph(outer).meta_subgraph(meta).node_attr(0, "payload") == b"\x00\xff"
    assert table.source_views()[0].meta_subgraph(outer).meta_subgraph(meta).edge_ids == [0, 1]
    assert old.meta_subgraph(meta).node_attr(0, "name") == "original"


def test_durable_group_transfer_backup_reopen_and_original_source_destruction(tmp_path):
    source = Graph(directed=True)
    initial(source)
    group = source.retain_group(source.read_view())
    meta = source.create_meta_node(group, effective_at=20)
    child = source.read_view_at(25).select_nodes([meta])
    path = tmp_path / "groups.ggj"
    backup = tmp_path / "backup.ggj"
    with DurableGraph.create(path) as graph:
        group = graph.retain_group(child, kind="outer")
        outer = graph.create_meta_node(group, effective_at=50)
        state = graph.current_state_id
        frozen = graph.read_view_at(50)
        graph.backup(backup)
    del source, child
    for location in [path, backup]:
        with DurableGraph.open(location) as graph:
            assert graph.current_state_id == state
            assert graph.read_view_at(49).node_ids == []
            view = graph.read_view_at(50).meta_subgraph(outer).meta_subgraph(meta)
            assert view.is_directed
            assert view.node_attr(0, "embedding") == [1.25, 2.5]
            assert view.node_attr(0, "payload") == b"\x00\xff"
    assert frozen.meta_subgraph(outer).meta_subgraph(meta).edge_ids == [0, 1]


def test_legacy_clocks_are_required_explicit_and_read_local():
    graph = Graph(manual=True)
    graph.add_node(v=1)
    first = graph.commit("old", "test")
    graph.set_node_attr(0, "v", None)
    second = graph.commit("old2", "test")
    clocks = {first: (10, 100), second: (20, 200)}
    with pytest.raises(ValueError, match="legacy clocks"):
        graph.read_view_at(15)
    assert graph.read_view_at(15, legacy_times=clocks).node_attr(0, "v") == 1
    view = graph.read_view_at(25, legacy_times=clocks)
    assert view.has_node_attribute(0, "v") and view.node_attr(0, "v") is None
    assert view.read_time["legacy_times"] == clocks
    assert graph.read_view_at(25, recorded_by=150, legacy_times=clocks).node_attr(0, "v") == 1
    assert graph.publication_metadata(first) is None
    assert graph.current_state_id == second
    with pytest.raises(ValueError, match="outside"):
        graph.read_view_at(15, legacy_times={999: (10, 100)})
    precise = Graph()
    initial(precise)
    with pytest.raises(ValueError, match="override precise"):
        precise.read_view_at(15, legacy_times={1: (0, 0)})


@pytest.mark.parametrize("clocks", [{1: (True, 1)}, {True: (1, 1)}, {1: (1,)}, {1: [1, 1]}])
def test_invalid_legacy_clock_descriptions_reject(clocks):
    graph = Graph(manual=True)
    graph.add_node()
    graph.commit("legacy", "test")
    with pytest.raises((TypeError, ValueError)):
        graph.read_view_at(100, legacy_times=clocks)


def test_repeated_reads_and_branch_changes_never_reuse_another_heads_values():
    graph = Graph()
    initial(graph)
    graph.create_branch("other")
    for _ in range(3):
        assert graph.read_view_at(15).node_attr(0, "name") == "original"
    graph.writer().set_node_attr(0, "name", "main", effective_at=15)
    assert graph.read_view_at(15).node_attr(0, "name") == "main"
    graph.checkout_branch("other")
    assert graph.read_view_at(15).node_attr(0, "name") == "original"


def test_known_frozen_group_sources_cannot_be_overridden():
    graph = Graph()
    initial(graph)
    source = graph.read_view()
    group = graph.retain_group(source)
    graph.create_meta_node(group, effective_at=20)
    with pytest.raises(ValueError, match="override a frozen"):
        graph.read_view_at(25, group_sources={group: source})


def test_duplicate_durable_retention_is_idempotent_and_does_not_publish_state(tmp_path):
    source = Graph()
    initial(source)
    frozen = source.read_view_at(15)
    path = tmp_path / "idempotent.ggj"
    with DurableGraph.create(path) as graph:
        group = graph.retain_group(frozen)
        before = path.read_bytes()
        assert graph.current_state_id == 0
        assert graph.retain_group(frozen) == group
        assert graph.current_state_id == 0
        assert path.read_bytes() == before
