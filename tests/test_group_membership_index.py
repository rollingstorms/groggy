"""Captured group membership preserves selection and source-local identity."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from groggy import AttrValue, DurableGraph, Graph, GroupMembershipIndex


def fixture(root=None):
    source = Graph()
    source.add_nodes(2)
    source.set_node_attr(0, "value", 7)
    for a, b in [(0, 1), (0, 1), (0, 0)]:
        source.add_edge(a, b)
    parent = Graph()
    child = parent.retain_group(source.read_view(), kind="child")
    parent.create_meta_node(child)
    root = Graph() if root is None else root
    outer = root.retain_group(parent.read_view(), kind="parent")
    assert root.retain_group(source.read_view(), kind="child") == child
    root.create_meta_node(outer)
    root.create_meta_node(child)
    root.writer().set_node_attr(0, "also", AttrValue.subgraph_ref(child))
    if isinstance(root, Graph):
        edge = root.add_edge(0, 1)
    else:
        edge = root.writer().add_edges([(0, 1)])[0]
    root.writer().set_edge_attr(edge, "evidence", AttrValue.subgraph_ref(child))
    return root, source, outer, child


def test_group_index_tracks_direct_recursive_references_and_qualified_membership():
    root, source, parent, child = fixture()
    head = root.current_state_id
    index = root.build_group_index()
    assert isinstance(index, GroupMembershipIndex)
    assert index.group_ids == sorted([parent, child])
    assert index.root_groups == sorted([parent, child])
    assert index.children(parent) == [child] and index.parents(child) == [parent]
    assert index.children(child) == [] and index.parents(parent) == []
    assert index.node_references(child) == [(0, "also"), (1, "contains_subgraph")]
    assert index.node_references(child, recursive=True) == [
        (0, "also"), (0, "contains_subgraph"), (1, "contains_subgraph")]
    assert index.edge_references(child, recursive=True) == [(0, "evidence")]
    assert all(index.contains_edge(child, i) for i in range(3))
    assert not index.contains_edge(parent, 0)
    assert index.contains_node(child, 1) and not index.contains_node(parent, 1)
    assert not index.contains_node(child, 999)
    assert index.group_view(child).node_attr(0, "value") == 7
    assert index.group_view(child).write_state_id is None
    source.set_node_attr(0, "value", 9)
    assert index.group_view(child).node_attr(0, "value") == 7
    assert root.current_state_id == head
    rows = index.node_references(child)
    rows.clear()
    assert len(index.node_references(child)) == 2
    assert index.copy().group_ids == index.group_ids


def test_selected_indexes_limit_root_references_and_reachable_sources():
    root, _, parent, child = fixture()
    root.retain_group(Graph().read_view(), kind="unused")
    selected = root.read_view().select_nodes([1])
    index = selected.build_group_index()
    assert index.root_groups == index.group_ids == [child]
    assert index.node_references(child, recursive=True) == [(1, "contains_subgraph")]
    assert index.edge_references(child, recursive=True) == []
    with pytest.raises(ValueError, match="outside"):
        index.group_view(parent)
    assert selected.select_nodes([]).build_group_index().group_ids == []
    parent_only = root.read_view().select_nodes([0]).build_group_index()
    assert parent_only.node_references(parent) == [(0, "contains_subgraph")]
    assert parent_only.group_view(child).node_ids == [0, 1]
    root.set_node_attr(1, "chosen", True)
    connected = root.filter_nodes("chosen == true")
    assert connected.build_group_index().root_groups == [child]


def test_rebuilds_branch_changes_and_owner_drop_preserve_old_group_indexes():
    root, source, parent, child = fixture()
    index = root.build_group_index()
    root.create_branch("old")
    root.remove_node(1)
    current = index.rebuild(root.read_view())
    assert current.node_references(child) == [(0, "also")]
    assert current.edge_references(child) == []
    root.checkout_branch("old")
    assert len(root.build_group_index().node_references(child)) == 2
    del root, source
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: index.node_references(child, recursive=True), range(12)))
    assert all(len(rows) == 3 for rows in results)
    assert current.children(parent) == [child]


def test_effective_time_group_indexes_retain_read_and_nested_source_pins():
    source, graph = Graph(), Graph()
    source.add_node(value=7)
    group = graph.retain_group(source.read_view())
    meta = graph.create_meta_node(group, effective_at=10)
    graph.writer().remove_nodes([meta], effective_at=20)
    past = graph.read_view_at(15)
    index = past.build_group_index()
    assert index.view.read_time == past.read_time
    assert index.view.write_state_id is None
    assert index.group_view(group).read_time is None
    assert graph.build_group_index().group_ids == []
    assert index.node_references(group) == [(meta, "contains_subgraph")]


def test_durable_group_indexes_leave_bytes_unchanged_and_survive_reopen(tmp_path):
    path = tmp_path / "groups.ggj"
    with DurableGraph.create(path) as graph:
        _, _, parent, child = fixture(graph)
        before, head = path.read_bytes(), graph.current_state_id
        index = graph.build_group_index()
        assert path.read_bytes() == before and graph.current_state_id == head
    assert index.group_view(child).node_attr(0, "value") == 7
    with DurableGraph.open(path) as graph:
        reopened = graph.build_group_index()
        assert reopened.group_ids == index.group_ids
        assert reopened.node_references(child, recursive=True) == index.node_references(child, recursive=True)
        assert reopened.children(parent) == [child]


@pytest.mark.parametrize("bad", [True, -1, 1.5, "0", 2**100])
def test_group_index_controls_reject_invalid_ids(bad):
    graph, _, _, child = fixture()
    index, head = graph.build_group_index(), graph.current_state_id
    for call in [lambda: index.group_view(bad), lambda: index.children(bad),
                 lambda: index.parents(bad), lambda: index.node_references(bad),
                 lambda: index.edge_references(bad), lambda: index.contains_node(child, bad),
                 lambda: index.contains_edge(child, bad)]:
        with pytest.raises((TypeError, OverflowError)):
            call()
    assert graph.current_state_id == head


def test_unknown_revisions_fail_without_implicit_global_or_live_lookup():
    index = Graph().build_group_index()
    assert index.group_ids == [] and index.root_groups == []
    for call in [lambda: index.group_view(0), lambda: index.children(0),
                 lambda: index.parents(0), lambda: index.node_references(0),
                 lambda: index.edge_references(0), lambda: index.contains_node(0, 0),
                 lambda: index.contains_edge(0, 0)]:
        with pytest.raises(ValueError, match="outside"):
            call()
