"""Complete retained graph archives, distinct from active JSON exports."""
import math
import struct

import pytest
from groggy import Graph, GraphBatch


def seed(graph):
    batch = GraphBatch(effective_at=10)
    first = batch.add_node(value=[1.25, -0.0])
    second = batch.add_node()
    batch.set_node_external_id(first, "original")
    batch.add_edge(first, second, weight=2.5)
    graph.apply_batch(batch, graph.current_state_id, "seed", "test")
    return graph.current_state_id


def test_complete_archive_restores_history_policy_config_and_reservations(tmp_path):
    graph = Graph(directed=True, config={"query_timeout_ms": 123})
    first = seed(graph)
    graph.checkpoint("label", "test")
    graph.create_branch("old")
    graph.writer().set_node_external_id(0, "renamed")
    graph.writer().unset_node_attr(0, "value", effective_at=20)
    unused = graph.add_node()
    graph.remove_node(unused)
    latest = graph.current_state_id
    clocks = graph.publication_metadata(first)
    path = tmp_path / "graph.gga"
    graph.save_archive(path)
    del graph
    copy = Graph.load_archive(path)
    assert copy.is_directed and copy.configuration()["query_timeout_ms"] == 123
    assert copy.current_state_id == latest and not copy.has_uncommitted_changes()
    assert copy.publication_metadata(first) == clocks
    assert len(copy.checkpoints()) == 1
    assert copy.allocated_entity_for_external_id("original") == ("node", 0)
    assert copy.allocated_entity_for_external_id("renamed") == ("node", 0)
    assert copy.get_node_attr(0, "value") is None
    copy.checkout_branch("old")
    assert copy.get_node_attr(0, "value") == [1.25, -0.0]
    assert math.copysign(1, copy.get_node_attr(0, "value")[1]) == -1
    assert copy.read_view_at(10).edge_ids == [0]
    assert copy.add_node() == unused + 1
    assert copy.current_state_id > latest


def test_archive_preserves_nonfinite_bits_null_absence_and_structured_types(tmp_path):
    graph = Graph()
    node = graph.add_node()
    nan = struct.unpack("d", struct.pack("Q", 0x7FF800000000007B))[0]
    graph.set_node_attr(node, "bits", [nan, float("inf"), -0.0])
    graph.set_node_attr(node, "null", None)
    graph.set_node_attr(node, "payload", {"wide": 10**30, "items": [1, "two"]})
    path = tmp_path / "values.gga"
    graph.save_archive(path)
    copy = Graph.load_archive(path)
    values = copy.get_node_attr(node, "bits")
    assert [struct.unpack("Q", struct.pack("d", value))[0] for value in values] == [
        0x7FF800000000007B, 0x7FF0000000000000, 0x8000000000000000
    ]
    assert copy.get_node_attr(node, "payload") == {"wide": 10**30, "items": [1, "two"]}
    assert copy.snapshot_at_commit(copy.current_state_id).node_attr(node, "null").value is None
    assert copy.snapshot_at_commit(copy.current_state_id).node_attr(node, "absent") is None


def test_frozen_nested_sources_survive_save_owner_drop_and_restore(tmp_path):
    graph = Graph()
    seed(graph)
    first = graph.retain_group(graph.read_view_at(10))
    meta = graph.create_meta_node(first)
    second = graph.retain_group(graph.read_view_at(10**18))
    outer = graph.create_meta_node(second)
    graph.set_node_attr(0, "value", "new")
    path = tmp_path / "groups.gga"
    graph.save_archive(path)
    del graph
    copy = Graph.load_archive(path)
    view = copy.read_view_at(10**18)
    assert view.meta_subgraph(outer).meta_subgraph(meta).node_attr(0, "value") == [1.25, -0.0]
    assert copy.get_node_attr(0, "value") == "new"


def test_manual_policy_dirty_rejection_and_independent_prepared_capture(tmp_path):
    graph = Graph(manual=True)
    graph.add_node()
    path = tmp_path / "manual.gga"
    with pytest.raises(ValueError, match="uncommitted"):
        graph.save_archive(path)
    assert not path.exists()
    graph.commit("accept", "test")
    graph.save_archive(path)
    copy = Graph.load_archive(path)
    copy.add_node()
    assert copy.has_uncommitted_changes()
    assert len(graph.nodes) == 1


@pytest.mark.parametrize("damage", ["truncate", "checksum", "append", "header"])
def test_archive_corruption_and_existing_destination_reject_without_clobber(tmp_path, damage):
    graph = Graph()
    seed(graph)
    path = tmp_path / "source.gga"
    graph.save_archive(path)
    original = path.read_bytes()
    with pytest.raises(Exception):
        graph.save_archive(path)
    assert path.read_bytes() == original
    data = bytearray(original)
    if damage == "truncate":
        data = data[:-1]
    elif damage == "checksum":
        data[40] ^= 1
    elif damage == "append":
        data.append(0)
    else:
        data[0] ^= 1
    bad = tmp_path / "bad.gga"
    bad.write_bytes(data)
    with pytest.raises(Exception):
        Graph.load_archive(bad)
    assert path.read_bytes() == original


def test_durable_import_keeps_retained_source_and_continues_after_reopen_and_backup(tmp_path):
    from groggy import DurableGraph
    graph = Graph()
    root = seed(graph)
    graph.create_branch("root")
    group = graph.retain_group(graph.read_view_at(10))
    meta = graph.create_meta_node(group)
    graph.writer().set_node_external_id(0, "new")
    graph.writer().unset_node_attr(0, "value", effective_at=20)
    unused = graph.add_node()
    graph.remove_node(unused)
    source_state = graph.current_state_id
    path = tmp_path / "journal.ggj"
    backup = tmp_path / "backup.ggj"
    with DurableGraph.create_from_graph(path, graph) as store:
        assert store.current_state_id == source_state
        assert graph.current_state_id == source_state
        batch = GraphBatch(effective_at=30)
        batch.add_node(v=9)
        result = store.apply_batch(batch, source_state, "after", "test")
        assert result["added_nodes"] == [unused + 1]
        latest = store.current_state_id
        store.backup(backup)
    del graph
    for path in (path, backup):
        with DurableGraph.open(path) as store:
            assert store.current_state_id == latest
            assert store.allocated_entity_for_external_id("original") == ("node", 0)
            assert store.allocated_entity_for_external_id("new") == ("node", 0)
            view = store.read_view_at(10**18)
            assert view.meta_subgraph(meta).node_attr(0, "value") == [1.25, -0.0]
            store.checkout_branch("root", store.current_state_id)
            assert store.current_state_id == root
            assert store.snapshot_at_commit(root).node_attr(0, "value").value == [1.25, -0.0]


def test_durable_import_rejects_dirty_source_and_existing_destination(tmp_path):
    from groggy import DurableGraph
    graph = Graph(manual=True)
    graph.add_node()
    path = tmp_path / "journal.ggj"
    with pytest.raises(ValueError, match="uncommitted"):
        DurableGraph.create_from_graph(path, graph)
    assert not path.exists()
    graph.commit("accept", "test")
    path.write_bytes(b"existing")
    with pytest.raises(Exception):
        DurableGraph.create_from_graph(path, graph)
    assert path.read_bytes() == b"existing"
