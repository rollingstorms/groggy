"""Explicit tracked restoration and reference switching share one API."""
import math

import pytest

from groggy import DurableGraph, Graph, GraphBatch


def fixture():
    g = Graph()
    batch = GraphBatch(effective_at=10)
    a = batch.add_node(value=-0.0)
    b = batch.add_node(value=None)
    batch.add_edge(a, b, weight=7)
    g.writer().apply_batch(batch)
    old = g.current_state_id
    g.writer().set_node_attr(0, "value", "changed", effective_at=20)
    g.writer().remove_nodes([1], effective_at=30)
    return g, old, g.current_state_id


@pytest.mark.parametrize("durable", [False, True])
def test_tracked_restore_has_one_new_parent_delta_clocks_pinned_views_and_recovery(tmp_path, durable):
    source, target, current = fixture()
    path = tmp_path / "graph"
    g = DurableGraph.create_from_graph(path, source) if durable else source
    try:
        pin = g.read_view()
        versions = len(g.commit_history())
        writer = g.writer(author="owner")
        restored = writer.restore_state(
            target, mode="tracked", effective_at=50,
            change_times=[{"kind": "node_added", "entity": 1, "effective_at": 40}],
            message="restore original"
        )
        assert restored > current
        assert writer.state_id == restored
        assert len(g.commit_history()) == versions + 1
        record = next(r for r in g.commit_history() if r.id == restored)
        assert record.parents == [current]
        assert record.author == "owner"
        assert record.message == "restore original"
        assert g.read_view().edge_endpoints(0) == (0, 1)
        assert math.copysign(1, g.get_node_attr(0, "value")) == -1
        assert g.get_node_attr(1, "value") is None
        assert 1 not in pin.node_ids
        assert 1 not in g.read_view_at(39).node_ids
        assert 1 in g.read_view_at(40).node_ids
        assert g.snapshot_at_commit(current).node_attr(0, "value").value == "changed"
        # No implicit source-history pruning or checkpoint creation.
        assert g.snapshot_at_commit(target).edge_exists(0)
        g.check_integrity()
        if durable:
            g.verify_storage()
            backup = tmp_path / "backup"
            g.backup(backup)
    finally:
        if durable:
            g.close()
    if durable:
        for candidate in (path, backup):
            with DurableGraph.open(candidate) as reopened:
                assert reopened.current_state_id == restored
                assert math.copysign(1, reopened.get_node_attr(0, "value")) == -1
                assert reopened.snapshot_at_commit(current).node_attr(0, "value").value == "changed"
                reopened.verify_storage()


def test_mode_is_required_and_unknown_modes_and_switch_timestamps_reject():
    g, target, current = fixture()
    for caller in (g, g.writer()):
        with pytest.raises(TypeError):
            caller.restore_state(target)
        with pytest.raises(ValueError, match="mode"):
            caller.restore_state(target, mode="guess")
        with pytest.raises(ValueError, match="timestamps"):
            caller.restore_state(target, mode="switch", effective_at=50)
        assert g.current_state_id == current
    assert g.restore_state(target, mode="switch") == target
    assert g.current_state_id == target
    assert len(g.commit_history()) == current


@pytest.mark.parametrize("durable", [False, True])
def test_tracked_restore_rejects_stale_dirty_bad_timeline_and_extraneous_overrides(tmp_path, durable):
    source, target, current = fixture()
    g = DurableGraph.create_from_graph(tmp_path / "graph", source) if durable else source
    try:
        versions = len(g.commit_history())
        for kwargs in (
            {"expected_state": target}, {"effective_at": 5},
            {"change_times": [{"kind": "node_added", "entity": 999, "effective_at": 50}]},
        ):
            with pytest.raises(ValueError):
                g.restore_state(target, mode="tracked", **kwargs)
            assert g.current_state_id == current
            assert len(g.commit_history()) == versions
            assert g.get_node_attr(0, "value") == "changed"
    finally:
        if durable:
            g.close()
    manual = Graph(manual=True)
    manual.add_node()
    old = manual.commit("initial", "test")
    manual.set_node_attr(0, "pending", 1)
    with pytest.raises(ValueError, match="uncommitted"):
        manual.restore_state(old, mode="tracked")
    assert manual.has_uncommitted_changes()
    manual.reset_hard()
    action = manual.restore_state(old, mode="tracked")
    assert action > old
    assert not manual.has_uncommitted_changes()


def test_data_identical_tracked_restore_records_explicit_action_and_switch_does_not():
    g = Graph()
    g.add_node(value=3)
    old = g.current_state_id
    assert g.restore_state(old, mode="switch") == old
    new = g.restore_state(old, mode="tracked")
    assert new > old
    assert len(g.commit_history()) == 2
    assert g.get_node_attr(0, "value") == 3
    assert g.restore_state(old, mode="switch") == old
    assert len(g.commit_history()) == 2
