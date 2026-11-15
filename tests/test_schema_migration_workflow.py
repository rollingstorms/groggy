"""Explicit compatible policy stages preserve atomic data and retained reads."""

import pytest

from groggy import DurableGraph, GraphBatch


def policy(kind):
    return {"node_labels": {"Entity": {"code": {"type": kind}}}}


def test_compatible_schema_migration_reopens_with_retained_original_values(tmp_path):
    path = tmp_path / "migration.ggj"
    backup = tmp_path / "migration-backup.ggj"
    with DurableGraph.create(path, schema=policy("int")) as graph:
        writer = graph.writer()
        nodes = [writer.add_node(labels=["Entity"], code=value) for value in (10, 20)]
        original = graph.current_state_id
        retained = graph.snapshot_at_commit(original)
        graph.create_branch("before-migration", original)
        before = path.read_bytes()
        with pytest.raises(ValueError, match="code"):
            graph.set_schema(policy("text"), original, 0)
        assert path.read_bytes() == before
        assert graph.schema_revision == 0

        # The compatible intermediate still requires the property and rejects Null.
        assert graph.set_schema(policy("any"), original, 0) == 1
        bad = GraphBatch()
        bad.set_node_attr(nodes[0], "code", "10")
        bad.set_node_attr(nodes[1], "code", None)
        before = path.read_bytes()
        with pytest.raises(ValueError, match="code"):
            graph.apply_batch(bad, original, "invalid migration", "test")
        assert graph.current_state_id == original
        assert path.read_bytes() == before
        assert [graph.get_node_attr(node, "code") for node in nodes] == [10, 20]

        batch = GraphBatch()
        for node, value in zip(nodes, ("10", "20")):
            batch.set_node_attr(node, "code", value)
        migrated = graph.apply_batch(batch, original, "convert codes", "test")["state_id"]
        assert graph.set_schema(policy("text"), migrated, 1) == 2
        assert retained.node_attr(nodes[0], "code").value == 10
        before = path.read_bytes()
        with pytest.raises(ValueError, match="code"):
            graph.checkout_branch("before-migration", migrated)
        assert graph.current_state_id == migrated
        assert path.read_bytes() == before
        graph.backup(backup)

    for candidate in (path, backup):
        with DurableGraph.open(candidate) as recovered:
            assert recovered.schema_revision == 2
            assert recovered.schema()["node_labels"]["Entity"]["code"]["type"] == "text"
            assert [recovered.get_node_attr(node, "code") for node in nodes] == ["10", "20"]
            assert recovered.snapshot_at_commit(original).node_attr(nodes[1], "code").value == 20
    assert retained.node_attr(nodes[0], "code").value == 10
