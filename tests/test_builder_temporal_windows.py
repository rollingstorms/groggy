"""The remaining retained-state temporal kernels share checked publication."""
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


def attach(source="result", attr="answer"):
    return {"id": "core.attach_node_attr", "params": {"source": source, "attr": attr}}


def fixture():
    graph = gr.Graph(manual=True)
    graph.add_nodes([{"score": 1, "label": "index_0", "nullable": None}, {}])
    graph.add_edge(0, 1)
    graph.commit("first", "test")
    first = graph.current_state_id
    graph.set_node_attr(0, "score", 3)
    graph.add_node()
    graph.commit("second", "test")
    second = graph.current_state_id
    graph.set_edge_attr(0, "unrelated", 10)
    graph.commit("third", "test")
    return graph, first, second, graph.current_state_id


def aggregate(start, end, function="sum", attr="score", **params):
    return {"id": "temporal.window_aggregate", "params": {"attr": attr, "function": function, "start": start, "end": end, "output": "result", **params}}


@pytest.mark.parametrize("function,expected,missing", [("count", 2, 0), ("sum", 4.0, None), ("avg", 2.0, None), ("min", 1.0, None), ("max", 3.0, None), ("first", 1, None), ("last", 3, None)])
@pytest.mark.parametrize("persist", [False, True])
def test_window_aggregates_changes_once_without_snapshot_fallback(function, expected, missing, persist):
    graph, first, _, head = fixture()
    prefix = [{"id": "core.init_nodes", "params": {"target": "prefix", "value": 100}}, attach("prefix", "score")]
    _, profile = graph.view().apply(Steps(prefix + [aggregate(first, head, function), attach()]), persist=persist, return_profile=True)
    actual = [graph.get_node_attr(n, "answer") for n in range(3)] if persist else [value for _, value in profile["outputs"]["node_attributes.answer"]]
    assert actual == [expected, missing, missing]
    assert graph.current_state_id == head + int(persist)


@pytest.mark.parametrize("function,attr,expected", [("first", "label", "index_0"), ("last", "nullable", None), ("count", "nullable", 1)])
def test_text_prefixes_are_literal_and_null_is_an_explicit_history_value(function, attr, expected):
    graph, first, _, head = fixture()
    graph.view().apply(Steps([aggregate(first, head, function, attr), attach()]), persist=True)
    assert graph.get_node_attr(0, "answer") == expected


@pytest.mark.parametrize("invalid", [True, None, "bad", float("inf"), float("nan")])
def test_numeric_history_values_reject_without_silently_skipping_or_publishing(invalid):
    graph, first, _, _ = fixture()
    graph.set_node_attr(0, "score", invalid)
    graph.commit("invalid", "test")
    head = graph.current_state_id
    prefix = [{"id": "core.init_nodes", "params": {"target": "prefix", "value": 4}}, attach("prefix", "prefix")]
    with pytest.raises(RuntimeError, match="finite numeric"):
        graph.view().apply(Steps(prefix + [aggregate(first, head), attach()]), persist=True)
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("kind,expected", [("created", [0, 0, 1]), ("deleted", [0, 0, 0]), ("modified", [1, 0, 1]), ("any", [1, 0, 1])])
def test_change_type_masks_use_actual_events_in_the_requested_window(kind, expected):
    graph, _, second, head = fixture()
    step = {"id": "temporal.mark_changed_nodes", "params": {"start": second, "end": second, "change_type": kind, "output": "result"}}
    graph.view().apply(Steps([step, attach()]), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == expected
    assert graph.current_state_id == head + 1


@pytest.mark.parametrize("predicate,coordinate,expected", [("created_after", "first", [0, 0, 1]), ("created_before", "second", [1, 1, 0]), ("existed_at", "first", [1, 1, 0]), ("modified_in_range", "second", [1, 0, 1])])
def test_lifecycle_filters_use_retained_states(predicate, coordinate, expected):
    graph, first, second, head = fixture()
    state = first if coordinate == "first" else second
    params = {"predicate": predicate, "output": "result"}
    params.update({"start": state, "end": state} if predicate == "modified_in_range" else {"commit": state})
    graph.view().apply(Steps([{"id": "temporal.filter", "params": params}, attach()]), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == expected
    assert graph.current_state_id == head + 1


def test_window_marks_historical_existence_and_can_chain_all_four_kernels():
    graph, first, second, head = fixture()
    steps = [aggregate(first, head), attach(attr="sum"),
        {"id": "temporal.filter", "params": {"predicate": "existed_at", "commit": first, "output": "result"}}, attach(attr="existed"),
        {"id": "temporal.mark_changed_nodes", "params": {"start": second, "end": second, "output": "result"}}, attach(attr="changed"),
        {"id": "temporal.window", "params": {"start": first, "end": first, "output": "result"}}, attach(attr="window"),
    ]
    graph.view().apply(Steps(steps), persist=True)
    assert graph.current_state_id == head + 1
    assert [graph.get_node_attr(n, "window") for n in range(3)] == [1, 1, 0]
    assert graph.get_node_attr(0, "sum") == 4.0
    assert graph.get_node_attr(2, "changed") == 1


@pytest.mark.parametrize("kind", ["window_aggregate", "filter", "mark_changed_nodes", "window"])
def test_invalid_retained_bounds_discard_prior_assignments(kind):
    graph, first, _, head = fixture()
    params = {"output": "result", "start": first, "end": 2**60}
    if kind == "filter": params["predicate"] = "modified_in_range"
    if kind == "window_aggregate": params.update(attr="score", function="sum")
    steps = [{"id": "core.init_nodes", "params": {"target": "prefix", "value": 9}}, attach("prefix", "prefix"), {"id": "temporal." + kind, "params": params}]
    with pytest.raises(RuntimeError, match="lineage"):
        graph.view().apply(Steps(steps), persist=True)
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("params", [{"start": -1, "end": 0}, {"start": 1.5, "end": 2}, {"start": 1}, {"end": 2}, {"start": 2, "end": 1}])
@pytest.mark.parametrize("kind", ["window_aggregate", "mark_changed_nodes", "window"])
def test_window_factories_reject_partial_invalid_or_truncated_coordinates(kind, params):
    graph, _, _, head = fixture()
    if kind == "window_aggregate": params = {"attr": "score", "function": "sum", **params}
    with pytest.raises(RuntimeError):
        graph.view().apply(Steps([{"id": "temporal." + kind, "params": {"output": "result", **params}}, attach()]), persist=True)
    assert graph.current_state_id == head
