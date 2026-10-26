"""Branch-selected state reads, immutable referents and durable branch lifecycle."""
import pytest
from groggy import Graph, DurableGraph, GraphBatch

LATEST = 2**64 - 1


def initial(graph):
    batch = GraphBatch(effective_at=10)
    a = batch.add_node(v=None)
    b = batch.add_node()
    batch.add_edge(a, b, w=1.25)
    return graph.apply_batch(batch, 0, "initial", "test")["state_id"]


def branch(graph, name):
    if isinstance(graph, DurableGraph):
        return graph.create_branch(name, graph.current_state_id)
    return graph.create_branch(name)


def checkout(graph, name):
    if isinstance(graph, DurableGraph):
        return graph.checkout_branch(name, graph.current_state_id)
    return graph.checkout_branch(name)


@pytest.mark.parametrize("durable", [False, True])
def test_recorded_timestamp_reads_never_select_sibling_or_unselected_future_states(tmp_path, durable):
    graph = DurableGraph.create(tmp_path / "graph.ggj") if durable else Graph()
    try:
        branch(graph, "empty")
        first = initial(graph)
        branch(graph, "other")
        graph.writer().set_node_attr(0, "v", 2, effective_at=20)
        main = graph.current_state_id
        old = graph.read_view_at(30)
        checkout(graph, "other")
        assert graph.snapshot_at_timestamp(LATEST).commit_id == first
        graph.writer().unset_node_attr(0, "v", effective_at=30)
        side = graph.current_state_id
        reserved = graph.writer().add_node()
        checkout(graph, "main")
        assert graph.current_state_id == main
        selected = graph.snapshot_at_timestamp(LATEST)
        assert selected.commit_id == main and selected.node_attr(0, "v").value == 2
        assert not selected.node_exists(reserved)
        assert graph.snapshot_at_commit(side).node_attr(0, "v") is None
        assert graph.read_view_at(40).node_attr(0, "v") == 2
        assert old.node_attr(0, "v") == 2 and old.read_time["branch"] == "main"
        copied = graph.snapshot_at_commit(first, owned=True).to_graph()
        assert copied.snapshot_at_timestamp(LATEST).commit_id == first
        assert copied.snapshot_at_commit(side).node_attr(0, "v") is None
        checkout(graph, "empty")
        with pytest.raises(ValueError, match="selected lineage"):
            graph.snapshot_at_timestamp(LATEST)
        assert graph.writer().add_node() == reserved + 1
    finally:
        if durable:
            graph.close()


def test_frozen_group_sources_and_branch_selection_recover_in_backup_and_original(tmp_path):
    path = tmp_path / "graph.ggj"
    backup = tmp_path / "backup.ggj"
    with DurableGraph.create(path, directed=True) as graph:
        initial(graph)
        group = graph.retain_group(graph.read_view_at(10))
        meta = graph.create_meta_node(group, effective_at=20)
        branch(graph, "other")
        graph.writer().set_node_attr(0, "v", 2, effective_at=30)
        main = graph.current_state_id
        frozen = graph.read_view_at(40)
        checkout(graph, "other")
        graph.writer().remove_node(0, effective_at=40)
        side = graph.current_state_id
        pinned = graph.read_view_at(40)
        assert pinned.meta_subgraph(meta).has_node_attribute(0, "v")
        graph.backup(backup)
    for location in [path, backup]:
        with DurableGraph.open(location) as graph:
            assert graph.snapshot_at_timestamp(LATEST).commit_id == side
            assert not graph.read_view_at(40).contains_node(0)
            assert graph.read_view_at(40).meta_subgraph(meta).has_node_attribute(0, "v")
            checkout(graph, "main")
            assert graph.snapshot_at_timestamp(LATEST).commit_id == main
            assert graph.read_view_at(40).node_attr(0, "v") == 2
            assert graph.read_view_at(40).edge_ids == [0]
    assert pinned.meta_subgraph(meta).edge_ids == [0]
    assert frozen.node_attr(0, "v") == 2


def test_rejected_durable_branch_operations_preserve_journal_and_installed_policy(tmp_path):
    path = tmp_path / "branches.ggj"
    schema = {"node_labels": {"Entity": {"name": {"type": "text"}}}}
    with DurableGraph.create(path) as graph:
        node = graph.writer().add_node(labels=["Entity"])
        branch(graph, "invalid")
        graph.writer().set_node_attr(node, "name", "Ada")
        graph.set_schema(schema, graph.current_state_id, graph.schema_revision)
        head = graph.current_state_id
        for operation in [lambda: graph.create_branch("invalid", head),
                          lambda: graph.create_branch("stale", 0),
                          lambda: graph.checkout_branch("unknown", head),
                          lambda: graph.checkout_branch("invalid", head)]:
            before = path.read_bytes()
            with pytest.raises(ValueError):
                operation()
            assert path.read_bytes() == before and graph.current_state_id == head
        assert graph.get_node_attr(node, "name") == "Ada"
    with DurableGraph.open(path) as graph:
        assert graph.get_node_attr(node, "name") == "Ada"
        assert graph.snapshot_at_timestamp(LATEST).commit_id == head


def test_manual_branch_metadata_points_at_published_state_and_dirty_checkout_rejects():
    graph = Graph(manual=True)
    graph.add_node(v=1)
    first = graph.commit("first", "test")
    graph.set_node_attr(0, "pending", 3)
    branch(graph, "published-only")
    assert graph.current_state_id == first and graph.has_uncommitted_changes()
    with pytest.raises(ValueError, match="uncommitted"):
        checkout(graph, "published-only")
    graph.reset_hard()
    checkout(graph, "published-only")
    assert graph.current_state_id == first and not graph.has_node_attribute(0, "pending")
    assert graph.snapshot_at_timestamp(LATEST).node_attr(0, "v").value == 1
