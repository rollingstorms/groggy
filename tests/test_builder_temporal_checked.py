"""Checked temporal math and retained snapshots share the atomic builder boundary."""
import json

import pytest
import groggy as gr
from groggy.algorithms.base import AlgorithmHandle


class Steps(AlgorithmHandle):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        return {"id": self.id, "params": {"steps": gr.AttrValue(json.dumps({"atomic": True, "steps": self.steps}))}}


def init(edges=False, value=8):
    return {"id": "core.init_edges" if edges else "core.init_nodes", "params": {"target": "values", "value": value}}


def attach(source="result", attr="answer", edges=False):
    return {"id": "core.attach_edge_attr" if edges else "core.attach_node_attr", "params": {"source": source, "attr": attr}}


def decay(**params):
    return {"id": "temporal.decay", "params": {"attr": "values", "half_life": 2.0, "time_delta": 4.0, "output": "result", **params}}


def fixture():
    graph = gr.Graph()
    graph.add_nodes([{}, {}, {}])
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    return graph


@pytest.mark.parametrize("edges", [False, True])
@pytest.mark.parametrize("persist", [False, True])
def test_decay_preserves_entity_kind_and_one_publication(edges, persist):
    graph = fixture()
    before = graph.current_state_id
    _, profile = graph.view().apply(Steps([init(edges), decay(), attach(edges=edges)]), persist=persist, return_profile=True)
    ids = range(2 if edges else 3)
    if persist:
        getter = graph.get_edge_attr if edges else graph.get_node_attr
        assert [getter(i, "answer") for i in ids] == [2.0 for _ in ids]
    else:
        key = "edge_attributes.answer" if edges else "node_attributes.answer"
        assert sorted(profile["outputs"][key]) == [(i, 2.0) for i in ids]
    assert graph.current_state_id == before + int(persist)


def test_decay_reads_loaded_staged_properties_and_allows_in_place_variable():
    graph = fixture()
    before = graph.current_state_id
    graph.view().apply(Steps([
        init(value=12), attach("values", "score"),
        {"id": "core.load_node_attr", "params": {"attr": "score", "target": "values", "default": 0}},
        decay(output="values"), attach("values"),
    ]), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == [3.0] * 3
    assert graph.current_state_id == before + 1


@pytest.mark.parametrize("invalid", [True, None, "text"])
def test_decay_rejects_non_numeric_values_and_discards_prior_attachments(invalid):
    graph = fixture()
    before = graph.current_state_id
    with pytest.raises(RuntimeError, match="finite numeric"):
        graph.view().apply(Steps([init(), attach("values", "prefix"), init(value=invalid), decay(), attach()]), persist=True)
    assert graph.current_state_id == before
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("params", [{"half_life": 0}, {"half_life": -1}, {"half_life": True}, {"half_life": "bad"}, {"time_delta": -1}, {"time_delta": True}, {"time_delta": "bad"}, {"time_delta": None}])
def test_decay_rejects_invalid_control_values_before_publication(params):
    graph = fixture()
    before = graph.current_state_id
    with pytest.raises(RuntimeError):
        graph.view().apply(Steps([init(), attach("values", "prefix"), decay(**params), attach()]), persist=True)
    assert graph.current_state_id == before
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("elapsed,expected", [(0.0, 8.0), (5e-324, 4.0), (1e308, 0.0)])
def test_decay_handles_subnormal_half_life_without_nan(elapsed, expected):
    graph = fixture()
    graph.view().apply(Steps([init(), decay(half_life=5e-324, time_delta=elapsed), attach()]), persist=True)
    assert graph.get_node_attr(0, "answer") == expected


def test_snapshot_builder_accepts_a_retained_state_with_staged_writes():
    graph = fixture()
    before = graph.current_state_id
    graph.view().apply(Steps([init(), attach("values", "prefix"),
        {"id": "temporal.snapshot_at", "params": {"state_id": before, "output": "saved"}},
        decay(), attach(),
    ]), persist=True)
    assert graph.current_state_id == before + 1
    assert graph.get_node_attr(0, "answer") == 2.0
    assert graph.snapshot_at_commit(before).node_attr(0, "prefix") is None


@pytest.mark.parametrize("params", [{}, {"state_id": -1}, {"timestamp": -1}, {"state_id": True}, {"state_id": 1.5}, {"timestamp": 1.0}, {"timestamp": "bad"}, {"state_id": None}, {"state_id": 1, "timestamp": 0}, {"state_id": 2**60}, {"timestamp": 0}])
def test_snapshot_builder_invalid_references_do_not_publish_prefix(params):
    graph = fixture()
    before = graph.current_state_id
    with pytest.raises(RuntimeError):
        graph.view().apply(Steps([init(), attach("values", "prefix"),
            {"id": "temporal.snapshot_at", "params": {"output": "saved", **params}}, attach("values")]), persist=True)
    assert graph.current_state_id == before
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("step,params,match", [
    ("temporal.diff_nodes", {"before": 1}, "before"),
    ("temporal.diff_edges", {"after": False}, "after"),
    ("temporal.diff_nodes", {"output_prefix": None}, "output_prefix"),
    ("temporal.window_aggregate", {"attr": "score", "function": "sum", "output": "result", "index_var": 1}, "index_var"),
    ("temporal.filter", {"predicate": "created_after", "commit": -1, "output": "result"}, "non-negative"),
    ("temporal.filter", {"predicate": "modified_in_range", "start": 3, "end": 2, "output": "result"}, "start"),
    ("temporal.mark_changed_nodes", {"output": "result", "change_type": 1}, "change_type"),
    ("temporal.mark_changed_nodes", {"output": "result", "change_type": "typo"}, "change_type"),
])
def test_existing_temporal_factories_reject_malformed_selectors(step, params, match):
    graph = fixture()
    before = graph.current_state_id
    with pytest.raises(RuntimeError, match=match):
        graph.view().apply(Steps([init(), attach("values", "prefix"), {"id": step, "params": params}, attach("values")]), persist=True)
    assert graph.current_state_id == before
    assert graph.get_node_attr(0, "prefix") is None
