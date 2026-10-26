"""Explicit maintenance retains graph semantics while deduplicating raw slots."""
import math

import pytest
from groggy import Graph


def test_pool_compaction_keeps_pinned_historical_and_archived_values(tmp_path):
    graph = Graph()
    graph.add_node(value=0)
    graph.create_branch("original")
    for i in range(20):
        graph.set_node_attr(0, "value", i % 3)
    frozen = graph.read_view()
    state = graph.current_state_id
    before = graph.check_integrity()
    report = graph.compact_pool()
    assert report["previous_node_values"] > report["current_node_values"]
    assert report["selected_state"] == graph.current_state_id == state
    assert graph.check_integrity()["retained_versions"] == before["retained_versions"]
    assert graph.get_node_attr(0, "value") == frozen.node_attr(0, "value") == 1
    assert graph.optimize() is None
    path = tmp_path / "archive.gga"
    graph.save_archive(path)
    copy = Graph.load_archive(path)
    copy.checkout_branch("original")
    assert copy.get_node_attr(0, "value") == 0
    assert frozen.node_attr(0, "value") == 1


def test_pool_compaction_preserves_signed_zero_nan_null_and_absence():
    graph = Graph()
    graph.add_node(negative=-0.0, positive=0.0, nan=float("nan"), explicit=None)
    for _ in range(3):
        graph.set_node_attr(0, "negative", -0.0)
        graph.set_node_attr(0, "positive", 0.0)
        graph.set_node_attr(0, "nan", float("nan"))
    graph.compact_pool()
    assert math.copysign(1, graph.get_node_attr(0, "negative")) == -1
    assert math.copysign(1, graph.get_node_attr(0, "positive")) == 1
    assert math.isnan(graph.get_node_attr(0, "nan"))
    assert graph.get_node_attr(0, "explicit") is None
    assert not graph.has_node_attribute(0, "missing")
    assert graph.has_node_attribute(0, "explicit")
    graph.check_integrity()


def test_pool_compaction_requires_accepted_manual_changes():
    graph = Graph(manual=True)
    graph.add_node(value=1)
    with pytest.raises(ValueError, match="uncommitted"):
        graph.compact_pool()
    graph.commit("accept", "test")
    graph.compact_pool()
