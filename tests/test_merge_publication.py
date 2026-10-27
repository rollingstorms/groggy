"""Checked merge plans are explicit, one-shot, typed, and owner-bound."""
import math
import pytest
from groggy import DurableGraph, Graph, GraphBatch, MergePlan, MergeOutcome

def fixture(conflict=False):
    g = Graph()
    g.add_node(value=0)
    base = g.current_state_id
    g.create_branch("right")
    g.set_node_attr(0, "value" if conflict else "left", 1)
    left = g.current_state_id
    g.checkout_branch("right")
    g.set_node_attr(0, "value" if conflict else "right", 2)
    right = g.current_state_id
    g.checkout_branch("main")
    return g, base, left, right

def test_conflict_free_plan_finish_publishes_one_version_and_preserves_views_and_parent_states():
    g, base, left, right = fixture()
    pin = g.read_view()
    count = len(g.commit_history())
    plan = g.prepare_merge(right, expected_state=left)
    assert isinstance(plan, MergePlan)
    assert list(plan.parents) == [left, right]
    assert plan.base_state == base
    assert plan.conflicts == []
    outcome = plan.resolve()
    assert isinstance(outcome, MergeOutcome)
    assert plan.is_consumed and not outcome.is_consumed
    merged = g.finish_merge(outcome)
    assert outcome.is_consumed
    assert len(g.commit_history()) == count + 1
    assert merged > right
    assert g.get_node_attr(0, "left") == 1
    assert g.get_node_attr(0, "right") == 2
    assert pin.node_attr(0, "right") is None
    assert g.snapshot_at_commit(left).node_attr(0, "right") is None
    with pytest.raises(ValueError, match="consumed"):
        plan.resolve()
    with pytest.raises(ValueError, match="consumed"):
        g.finish_merge(outcome)
    g.check_integrity()

def test_exact_side_resolution_and_custom_null_or_absence_are_distinct():
    g, _, left, right = fixture(True)
    plan = g.prepare_merge(right)
    conflict = plan.conflicts[0]
    assert conflict["kind"] == "node_attr"
    assert conflict["left"]["present"] and conflict["left"]["value"] == 1
    outcome = plan.resolve(lambda c: "right")
    g.finish_merge(outcome)
    assert g.get_node_attr(0, "value") == 2
    g.restore_state(left, mode="switch")
    outcome = g.prepare_merge(right).resolve(lambda c: {"kind": "property", "present": True, "value": None})
    g.finish_merge(outcome)
    assert g.read_view().node_attr(0, "value") is None  # explicit null has a property history event
    assert g.snapshot_at_commit(g.current_state_id).node_attr(0, "value") is not None
    g.restore_state(left, mode="switch")
    outcome = g.prepare_merge(right).resolve(lambda c: {"kind": "property", "present": False})
    g.finish_merge(outcome)
    assert g.snapshot_at_commit(g.current_state_id).node_attr(0, "value") is None

def test_callback_errors_keep_original_exception_and_do_not_mutate_graph():
    g, _, left, right = fixture(True)
    def resolver(conflict):
        raise RuntimeError("caller stopped")
    plan = g.prepare_merge(right)
    with pytest.raises(RuntimeError, match="caller stopped"):
        plan.resolve(resolver)
    assert plan.is_consumed
    assert g.current_state_id == left
    assert g.get_node_attr(0, "value") == 1

def test_callbacks_can_advance_owner_but_finish_rejects_stale_outcome():
    g, _, _, right = fixture(True)
    def resolver(conflict):
        g.set_node_attr(0, "intervening", True)
        return "right"
    outcome = g.prepare_merge(right).resolve(resolver)
    state = g.current_state_id
    with pytest.raises(ValueError, match="Stale"):
        g.finish_merge(outcome)
    assert g.current_state_id == state
    assert g.get_node_attr(0, "value") == 1

def test_foreign_branch_boolean_and_wrong_resolution_shapes_reject():
    g, _, left, right = fixture(True)
    for value in [True, -1, "wrong"]:
        with pytest.raises((TypeError, ValueError, OverflowError)):
            g.prepare_merge(value)
    with pytest.raises(TypeError):
        g.prepare_merge(right, base=True)
    with pytest.raises(ValueError):
        g.prepare_merge(right).resolve(lambda c: {"kind": "drop_edge"})
    outcome = g.prepare_merge(right).resolve(lambda c: "right")
    other, _, _, _ = fixture()
    with pytest.raises(ValueError, match="foreign"):
        other.finish_merge(outcome)
    assert g.current_state_id == left

def test_durable_writer_merge_reopens_and_keeps_schema_and_retained_history(tmp_path):
    g, _, left, right = fixture(True)
    path = tmp_path / "graph"
    with DurableGraph.create_from_graph(path, g) as store:
        writer = store.writer()
        merged = writer.merge(right, resolver=lambda c: "right")
        assert writer.state_id == merged
        store.verify_storage()
    with DurableGraph.open(path) as store:
        assert store.current_state_id == merged
        assert store.snapshot_at_commit(merged).node_attr(0, "value").value == 2
        assert store.snapshot_at_commit(left).node_attr(0, "value").value == 1
        store.check_integrity()

def test_topology_conflicts_require_explicit_drop_or_missing_endpoint_nodes():
    g = Graph()
    g.add_nodes(2)
    g.create_branch("right")
    g.remove_node(1)
    left = g.current_state_id
    g.checkout_branch("right")
    g.add_edge(0, 1)
    right = g.current_state_id
    g.checkout_branch("main")
    plan = g.prepare_merge(right)
    assert plan.conflicts[0]["kind"] == "topology"
    with pytest.raises(ValueError, match="Topology"):
        plan.resolve(lambda c: "right")
    outcome = g.prepare_merge(right).resolve(lambda c: {"kind": "endpoint_nodes", "nodes": [{"id": 1, "properties": {"restored": True}}]})
    g.finish_merge(outcome)
    assert g.get_node_attr(1, "restored") is True
    assert g.edge_endpoints(0) == (0, 1)
    g.restore_state(left, mode="switch")
    outcome = g.prepare_merge(right).resolve(lambda c: {"kind": "drop_edge"})
    g.finish_merge(outcome)
    assert not g.has_node(1) and not g.has_edge(0)

def test_effective_merge_times_and_strict_override_parsing():
    g = Graph()
    batch = GraphBatch(effective_at=10)
    batch.add_node(value=0)
    g.apply_batch(batch, 0, "root", "test")
    g.create_branch("right")
    g.set_node_attr(0, "left", True)
    left = g.current_state_id
    g.checkout_branch("right")
    g.set_node_attr(0, "right", 7)
    right = g.current_state_id
    g.checkout_branch("main")
    outcome = g.prepare_merge(right).resolve()
    with pytest.raises(TypeError):
        g.finish_merge(outcome, effective_at=True)
    assert not outcome.is_consumed
    with pytest.raises(ValueError, match="Duplicate"):
        g.finish_merge(outcome, change_times=[{"kind":"node_attr","entity":0,"name":"right","effective_at":50}]*2)
    assert not outcome.is_consumed
    state = g.finish_merge(outcome, change_times=[{"kind":"node_attr","entity":0,"name":"right","effective_at":50}])
    assert state > right and left < state
    assert g.read_view_at(50).node_attr(0, "right") == 7


def test_exact_float_side_choice_preserves_signed_zero_in_history_and_merge(tmp_path):
    g = Graph()
    g.add_node(value=1.0)
    g.create_branch("minus")
    g.set_node_attr(0, "value", 0.0)
    plus = g.current_state_id
    g.checkout_branch("minus")
    g.set_node_attr(0, "value", -0.0)
    minus = g.current_state_id
    g.checkout_branch("main")
    plan = g.prepare_merge(minus)
    assert plan.conflicts[0]["right"]["dtype"] == "Float64"
    merged = g.finish_merge(plan.resolve(lambda c: "right"))
    assert math.copysign(1.0, g.get_node_attr(0, "value")) == -1.0
    assert math.copysign(1.0, g.snapshot_at_commit(plus).node_attr(0, "value").value) == 1.0
    assert merged > minus
    g.check_integrity()

    path = tmp_path / "exact-bits"
    with DurableGraph.create_from_graph(path, g) as store:
        store.verify_storage()
    with DurableGraph.open(path) as store:
        for state, sign in [(plus, 1.0), (minus, -1.0), (merged, -1.0)]:
            value = store.snapshot_at_commit(state).node_attr(0, "value").value
            assert math.copysign(1.0, value) == sign
