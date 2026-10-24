"""Builder attachment steps honor nonpersistent execution and return owned columns."""
import json

import pytest

from groggy import AttrValue, Graph
from groggy.algorithms.base import AlgorithmHandle
from groggy.builder import AlgorithmBuilder


class StepsHandle(AlgorithmHandle):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps(self.steps))}}


def steps_handle(steps):
    return StepsHandle(steps)


@pytest.mark.parametrize("value", [1.0000000001, 2**60 + 1, True, "text", [1.0000000001, 2.0], None])
def test_builder_nonpersistent_typed_output_does_not_mutate_graph(value):
    graph = Graph()
    nodes = graph.add_nodes(2)
    state = graph.current_state_id
    builder = AlgorithmBuilder("transient")
    values = builder.init_nodes(default=value)
    builder.attach_as("result", values)
    _, profile = graph.view().apply(builder.build(), persist=False, return_profile=True)
    rows = profile["outputs"]["node_attributes.result"]
    assert rows == [(node, value) for node in nodes]
    assert graph.current_state_id == state
    assert not graph.has_uncommitted_changes()
    assert all(graph.get_node_attr(node, "result") is None for node in nodes)
    rows.clear()
    assert graph.current_state_id == state


def test_mixed_node_edge_outputs_are_distinct_and_owned():
    graph = Graph()
    graph.add_nodes(2)
    graph.add_edge(0, 1)
    state = graph.current_state_id
    handle = steps_handle([
        {"id": "core.init_nodes", "params": {"target": "nodes", "value": 2}},
        {"id": "core.init_edges", "params": {"target": "edges", "value": "edge"}},
        {"id": "core.attach_node_attr", "params": {"source": "nodes", "attr": "result"}},
        {"id": "core.attach_edge_attr", "params": {"source": "edges", "attr": "result"}},
    ])
    _, profile = graph.view().apply(handle, persist=False, return_profile=True)
    assert profile["outputs"]["node_attributes.result"] == [(0, 2), (1, 2)]
    assert profile["outputs"]["edge_attributes.result"] == [(0, "edge")]
    assert graph.current_state_id == state
    assert graph.get_edge_attr(0, "result") is None


def test_nonpersistent_attachment_leaves_later_graph_reads_unchanged():
    graph = Graph()
    graph.add_node(score=7)
    state = graph.current_state_id
    builder = AlgorithmBuilder("unchanged reads")
    builder.attach_as("score", builder.init_nodes(default=99))
    builder.attach_as("read_back", builder.load_attr("score"))
    _, profile = graph.view().apply(builder.build(), persist=False, return_profile=True)
    assert profile["outputs"]["node_attributes.score"] == [(0, 99)]
    assert profile["outputs"]["node_attributes.read_back"] == [(0, 7)]
    assert graph.get_node_attr(0, "score") == 7
    assert graph.current_state_id == state


def test_later_validation_failure_cannot_leave_nonpersistent_graph_changes():
    graph = Graph()
    graph.add_node()
    builder = AlgorithmBuilder("failure")
    values = builder.init_nodes(default=4)
    builder.attach_as("valid", values)
    builder.attach_as("entity_type", values)
    with pytest.raises(RuntimeError, match="entity_type"):
        graph.view().apply(builder.build(), persist=False)
    assert graph.get_node_attr(0, "valid") is None
    assert graph.current_state_id == 1
    assert not graph.has_uncommitted_changes()


def test_manual_dirty_graph_retains_pending_changes_without_publication():
    graph = Graph(manual=True)
    graph.add_node(existing=3)
    builder = AlgorithmBuilder("manual")
    builder.attach_as("result", builder.init_nodes(default=4))
    _, profile = graph.view().apply(builder.build(), persist=False, return_profile=True)
    assert profile["outputs"]["node_attributes.result"] == [(0, 4)]
    assert graph.current_state_id == 0
    assert graph.has_uncommitted_changes()
    assert graph.get_node_attr(0, "existing") == 3
    assert graph.get_node_attr(0, "result") is None


def test_empty_attachment_has_a_typed_empty_output_and_no_version():
    graph = Graph()
    builder = AlgorithmBuilder("empty")
    builder.attach_as("result", builder.init_nodes(default=3))
    _, profile = graph.view().apply(builder.build(), persist=False, return_profile=True)
    assert profile["outputs"]["node_attributes.result"] == []
    assert graph.current_state_id == 0


def test_smoothing_returns_two_transient_columns_without_overwriting_sources():
    graph = Graph()
    graph.add_nodes([{"x": x, "y": x * 2, "t": t, "stroke_id": 1}
                     for t, x in enumerate([0.0, 9.0, 0.0])])
    state = graph.current_state_id
    handle = steps_handle([{"id": "drot.smooth_strokes", "params": {
        "target_x": "x", "target_y": "y", "window": 3,
    }}])
    _, profile = graph.view().apply(handle, persist=False, return_profile=True)
    assert len(profile["outputs"]["node_attributes.x"]) == 3
    assert len(profile["outputs"]["node_attributes.y"]) == 3
    assert profile["outputs"]["node_attributes.x"] != [(0, 0.0), (1, 9.0), (2, 0.0)]
    assert graph.get_node_attr(1, "x") == 9.0
    assert graph.current_state_id == state


def test_explicit_persistence_still_writes_and_repeated_output_names_keep_last_column():
    graph = Graph()
    graph.add_node()
    builder = AlgorithmBuilder("last output")
    builder.attach_as("result", builder.init_nodes(default=1))
    builder.attach_as("result", builder.init_nodes(default=2))
    handle = builder.build()
    _, profile = graph.view().apply(handle, persist=False, return_profile=True)
    assert profile["outputs"]["node_attributes.result"] == [(0, 2)]
    assert graph.current_state_id == 1
    graph.view().apply(handle, persist=True)
    assert graph.get_node_attr(0, "result") == 2
