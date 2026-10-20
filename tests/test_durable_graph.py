import os
import subprocess
import sys

import pytest

from groggy import DurableGraph, GraphBatch


def test_durable_writer_restores_values_history_times_configuration_and_ids(tmp_path):
    path = tmp_path / "graph.ggj"
    with DurableGraph.create(path, directed=True, config={"query_timeout_ms": 123}) as graph:
        writer = graph.writer(author="test")
        node = writer.upsert_nodes([{"external_id": "ada", "labels": ["Entity"], "bytes": b"\x00\xff", "nested": {"a": [1, None]}}])[0]
        first = writer.state_id
        before = graph.snapshot_at_commit(first)
        writer.set_node_attr(node, "name", "Ada", effective_at=10)
        state = writer.state_id
        metadata = graph.publication_metadata(state)
        assert graph.commit_history()[-1].timestamp == metadata["recorded_at"] // 1_000_000
        assert before.node_attr(node, "name") is None
    assert graph.is_closed
    with DurableGraph.open(path) as recovered:
        assert recovered.is_directed
        assert recovered.current_state_id == state
        assert recovered.node_by_external_id("ada") == node
        assert recovered.get_node_attr(node, "bytes") == b"\x00\xff"
        assert recovered.get_node_attr(node, "nested") == {"a": [1, None]}
        assert recovered.get_node_attr(node, "name") == "Ada"
        assert recovered.publication_metadata(state) == metadata
        assert recovered.configuration()["query_timeout_ms"] == 123
        assert len(recovered.commit_history()) == 2
        assert recovered.writer().add_node() == 1
    assert before.node_attr(node, "bytes").value == b"\x00\xff"


def test_installed_policy_cannot_be_bypassed_by_no_schema_or_weaker_policy(tmp_path):
    path = tmp_path / "graph.ggj"
    schema = {"node_labels": {"Entity": {"name": {"type": "text"}}}}
    with DurableGraph.create(path, schema=schema) as graph:
        before = path.read_bytes()
        batch = GraphBatch()
        node = batch.add_node(labels=["Entity"])
        batch.set_node_external_id(node, "unreserved")
        with pytest.raises(ValueError, match="name"):
            graph.apply_batch(batch, 0, "bad", "test", schema={})
        assert path.read_bytes() == before
        assert graph.allocated_entity_for_external_id("unreserved") is None
        assert graph.writer().add_node(labels=["Entity"], name="Ada") == 0
    with DurableGraph.open(path) as graph:
        assert graph.schema()["node_labels"]["Entity"]["name"] == {
            "type": "text", "required": True, "nullable": False
        }
        with pytest.raises(ValueError, match="name"):
            graph.writer(schema={}).set_node_attr(0, "name", 1)
        assert graph.set_schema(None, graph.current_state_id, graph.schema_revision) == 1
        graph.writer().set_node_attr(0, "name", 1)
    with DurableGraph.open(path) as graph:
        assert graph.schema() is None
        assert graph.schema_revision == 1
        assert graph.get_node_attr(0, "name") == 1


def test_branches_reservations_and_high_water_counters_survive_restart(tmp_path):
    path = tmp_path / "graph.ggj"
    with DurableGraph.create(path) as graph:
        writer = graph.writer()
        assert writer.upsert_nodes([{"external_id": "a"}]) == [0]
        first = writer.state_id
        graph.create_branch("other", first)
        writer.upsert_nodes([{"external_id": "retained"}])
        graph.checkout_branch("other", writer.state_id)
    with DurableGraph.open(path) as graph:
        assert graph.current_state_id == first
        assert graph.node_by_external_id("retained") is None
        assert graph.allocated_entity_for_external_id("retained") == ("node", 1)
        assert graph.writer().add_node() == 2
        assert {row["name"] for row in graph.list_branches()} == {"main", "other"}


def test_file_lock_and_closed_store_checks(tmp_path):
    path = tmp_path / "graph.ggj"
    graph = DurableGraph.create(path)
    with pytest.raises(RuntimeError, match="lock"):
        DurableGraph.open(path)
    writer = graph.writer()
    graph.close()
    graph.close()
    with pytest.raises(ValueError, match="closed"):
        writer.add_node()
    with DurableGraph.open(path) as recovered:
        assert recovered.node_count() == 0


def test_partial_tail_is_truncated_and_full_corruption_is_an_error(tmp_path):
    path = tmp_path / "graph.ggj"
    with DurableGraph.create(path) as graph:
        graph.writer().add_node()
    valid = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b"GFRAME01\x01")
    with DurableGraph.open(path) as graph:
        assert graph.recovered_tail_bytes == 9
        assert graph.node_count() == 1
    assert path.read_bytes() == valid
    corrupt = bytearray(valid)
    corrupt[-1] ^= 1
    path.write_bytes(corrupt)
    with pytest.raises(ValueError, match="checksum"):
        DurableGraph.open(path)
    assert path.read_bytes() == corrupt


def test_crash_exit_without_python_cleanup_recovers_synced_transaction(tmp_path):
    path = tmp_path / "crash.ggj"
    program = '''
import os, sys
from groggy import DurableGraph
store = DurableGraph.create(sys.argv[1])
writer = store.writer()
writer.upsert_nodes([{"external_id": "crash-safe", "name": "Ada"}])
os._exit(0)
'''
    result = subprocess.run([sys.executable, "-c", program, str(path)], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with DurableGraph.open(path) as graph:
        assert graph.node_by_external_id("crash-safe") == 0
        assert graph.get_node_attr(0, "name") == "Ada"


def test_identical_upsert_does_not_append_journal_record(tmp_path):
    path = tmp_path / "graph.ggj"
    with DurableGraph.create(path) as graph:
        writer = graph.writer()
        row = {"external_id": "a", "name": "Ada"}
        assert writer.upsert_nodes([row]) == [0]
        before = path.read_bytes()
        state = writer.state_id
        assert writer.upsert_nodes([row]) == [0]
        assert writer.state_id == state
        assert path.read_bytes() == before
