"""Explicit full retained-state checks do not publish or repair data."""
import pytest
from groggy import DurableGraph, Graph, GraphBatch


def test_integrity_reports_all_retained_branches_without_changing_selected_state():
    graph = Graph()
    graph.add_node(value=-0.0)
    graph.create_branch("other")
    graph.set_node_attr(0, "value", 2.0)
    graph.checkout_branch("other")
    graph.set_node_attr(0, "value", 3.0)
    before = graph.current_state_id
    frozen = graph.read_view()
    report = graph.check_integrity()
    assert report["selected_state"] == before
    assert report["selected_branch"] == "other"
    assert report["retained_versions"] == 3 and report["verified_states"] == 4
    assert report["branches"] == 2 and report["active_nodes"] == 1
    assert report["maximum_active_nodes"] == 1 and report["legacy_versions"] == 0
    assert report["schema_revision"] is None and not report["current_schema_checked"]
    assert graph.current_state_id == before and frozen.node_attr(0, "value") == 3.0
    assert not graph.has_uncommitted_changes()


def test_manual_pending_changes_reject_and_accepted_legacy_history_remains_inspectable():
    graph = Graph(manual=True)
    graph.add_node(value=1)
    with pytest.raises(ValueError, match="uncommitted"):
        graph.check_integrity()
    assert graph.has_uncommitted_changes()
    graph.commit("base", "test")
    report = graph.check_integrity()
    assert report["legacy_versions"] == 1 and report["verified_states"] == 2
    assert report["selected_state"] == 1


def test_integrity_accepts_create_remove_in_one_batch_and_preserves_allocation():
    graph = Graph()
    batch = GraphBatch()
    node = batch.add_node()
    batch.remove_node(node)
    graph.apply_batch(batch, 0, "transient", "test")
    report = graph.check_integrity()
    assert report["active_nodes"] == 0
    assert report["allocated_nodes"] == 1
    assert graph.add_node() == 1


def test_durable_integrity_checks_captured_policy_without_rewriting_the_journal(tmp_path):
    path = tmp_path / "graph.ggj"
    with DurableGraph.create(path) as graph:
        batch = GraphBatch()
        batch.add_node(value=1)
        graph.apply_batch(batch, 0, "initial", "test")
        graph.set_schema({}, graph.current_state_id, 0)
        before = path.read_bytes()
        report = graph.check_integrity()
        assert report["current_schema_checked"] and report["schema_revision"] == 1
        assert report["retained_versions"] == 1
        assert path.read_bytes() == before
        with pytest.raises(Exception):
            DurableGraph.open(path)
    with DurableGraph.open(path) as graph:
        assert graph.check_integrity() == report
