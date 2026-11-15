"""File content sync is distinct from host-specific namespace/maintenance support."""
import os
import pytest
from groggy import Graph, DurableGraph, storage_capabilities


def test_platform_capabilities_report_host_mechanisms_without_shared_mutable_state():
    capabilities = storage_capabilities()
    assert capabilities == {"file_sync": True, "directory_sync": os.name == "posix", "checked_replacement": os.name == "posix"}
    capabilities["file_sync"] = False
    assert storage_capabilities()["file_sync"] is True


def test_archive_and_journal_backup_reopen_and_preserve_historical_values(tmp_path):
    graph = Graph()
    graph.add_node(score=1.25)
    graph.set_node_attr(0, "score", 2.5)
    archive = tmp_path / "graph.archive"
    graph.save_archive(archive)
    assert Graph.load_archive(archive).get_node_attr(0, "score") == 2.5
    journal = tmp_path / "graph.journal"
    backup = tmp_path / "backup.journal"
    with DurableGraph.create(journal) as durable:
        durable.writer().add_node(score=1.25)
        state = durable.current_state_id
        durable.backup(backup)
        durable.writer().set_node_attr(0, "score", 2.5)
        if not storage_capabilities()["checked_replacement"]:
            head = durable.current_state_id
            with pytest.raises((ValueError, RuntimeError), match="Unix"):
                durable.checkpoint_storage()
            assert durable.current_state_id == head
    with DurableGraph.open(journal) as reopened:
        assert reopened.get_node_attr(0, "score") == 2.5
    with DurableGraph.open(backup) as reopened:
        assert reopened.current_state_id == state
        assert reopened.get_node_attr(0, "score") == 1.25
