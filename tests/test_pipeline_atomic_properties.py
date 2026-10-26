"""An outer pipeline of certified builder algorithms shares one property acceptance."""
import json
import pytest
from groggy import Graph, Pipeline, apply, AttrValue
from groggy.algorithms.base import AlgorithmHandle
from groggy.builder import AlgorithmBuilder


class Steps(AlgorithmHandle):
    def __init__(self, steps, atomic=False):
        self.steps, self.atomic = steps, atomic
    @property
    def id(self):
        return "builder.step_pipeline"
    def to_spec(self):
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({
            "atomic": self.atomic, "steps": self.steps}))}}


def assign(name, value, atomic=False):
    builder = AlgorithmBuilder("assign " + name)
    builder.attach_as(name, builder.init_nodes(default=value))
    return builder.build(atomic=atomic)


def copy_attr(source, target, atomic=False):
    builder = AlgorithmBuilder("copy " + source)
    builder.attach_as(target, builder.load_attr(source))
    return builder.build(atomic=atomic)


@pytest.mark.parametrize("child_atomic", [False, True])
@pytest.mark.parametrize("entry", ["pipe", "subgraph", "apply", "graph"])
def test_outer_pipeline_reads_prior_algorithm_updates_and_accepts_once(child_atomic, entry):
    graph = Graph()
    graph.add_nodes([{"v": 1}, {"v": 2}])
    head = graph.current_state_id
    prior = graph.read_view()
    pipe = Pipeline([assign("v", 1.0000000001, child_atomic),
                     copy_attr("v", "copied", child_atomic), assign("v", 9, child_atomic)])
    if entry == "pipe":
        _, profile = pipe(graph.view(), atomic=True, return_profile=True)
    elif entry == "subgraph":
        _, profile = graph.view().apply(pipe, atomic=True, return_profile=True)
    elif entry == "apply":
        _, profile = apply(graph.view(), pipe, atomic=True, return_profile=True)
    else:
        _, profile = graph.apply(pipe, atomic=True, return_profile=True)
    assert graph.current_state_id == head + 1
    assert not graph.has_uncommitted_changes() and profile["atomic"]
    for node in [0, 1]:
        assert graph.get_node_attr(node, "v") == 9
        assert graph.get_node_attr(node, "copied") == 1.0000000001
    assert prior.node_attr(0, "v") == 1
    metadata = graph.publication_metadata(head + 1)
    assert len({row["effective_at"] for row in metadata["change_times"]}) == 1


def test_failure_in_later_algorithm_discards_prior_staged_properties_and_reuses_handle():
    graph = Graph()
    graph.add_node(v=1, denom=0)
    second = AlgorithmBuilder("division")
    value = second.init_nodes(default=1) / second.load_attr("denom")
    second.attach_as("invalid", value)
    pipe = Pipeline([assign("a", 7), second.build()])
    with pytest.raises(RuntimeError):
        pipe(graph.view(), atomic=True)
    assert graph.current_state_id == 1 and graph.get_node_attr(0, "a") is None
    assert not graph.has_uncommitted_changes()
    graph.set_node_attr(0, "denom", 1)
    head = graph.current_state_id
    pipe(graph.view(), atomic=True)
    assert graph.current_state_id == head + 1 and graph.get_node_attr(0, "invalid") == 1


def test_unsupported_later_algorithm_fails_before_prefix_attachment():
    graph = Graph()
    graph.add_node()
    streaming = Steps([{"id": "core.execution_block", "params": {
        "mode": "streaming", "target": "unsupported", "options": {}, "body": {"nodes": []}}}])
    with pytest.raises(RuntimeError, match="step 1.*does not support atomic"):
        Pipeline([assign("a", 7), streaming])(graph.view(), atomic=True)
    assert graph.current_state_id == 1 and graph.get_node_attr(0, "a") is None


def test_mixed_node_edge_updates_share_the_outer_overlay_and_selected_scope():
    graph = Graph()
    graph.add_nodes(3)
    graph.add_edge(0, 1, w=1)
    graph.add_edge(1, 2, w=2)
    selected = graph.induced_subgraph([0, 1])
    head = graph.current_state_id
    first = Steps([
        {"id": "core.init_edges", "params": {"target": "v", "value": 2.0000000001}},
        {"id": "core.attach_edge_attr", "params": {"source": "v", "attr": "w"}},
    ], atomic=True)
    second = Steps([
        {"id": "core.load_edge_attr", "params": {"target": "v", "attr": "w"}},
        {"id": "core.attach_edge_attr", "params": {"source": "v", "attr": "copied"}},
        {"id": "core.init_nodes", "params": {"target": "n", "value": 4}},
        {"id": "core.attach_node_attr", "params": {"source": "n", "attr": "selected"}},
    ])
    selected.apply([first, second], atomic=True)
    assert graph.current_state_id == head + 1
    assert graph.get_edge_attr(0, "copied") == 2.0000000001
    assert graph.get_edge_attr(1, "w") == 2 and graph.get_edge_attr(1, "copied") is None
    assert graph.get_node_attr(2, "selected") is None


def test_nonpersistent_outer_pipeline_returns_outputs_and_uses_original_graph_reads():
    graph = Graph()
    graph.add_node(v=7)
    pipe = Pipeline([assign("v", 9, True), copy_attr("v", "copy", True)])
    _, profile = pipe(graph.view(), atomic=True, persist=False, return_profile=True)
    assert graph.current_state_id == 1 and graph.get_node_attr(0, "v") == 7
    assert profile["outputs"]["node_attributes.v"] == [(0, 9)]
    assert profile["outputs"]["node_attributes.copy"] == [(0, 7)]


def test_dirty_manual_state_rejects_outer_run_and_clean_manual_state_accepts_once():
    graph = Graph(manual=True)
    graph.add_node(v=1)
    pipe = Pipeline([assign("a", 2), copy_attr("a", "b")])
    with pytest.raises(RuntimeError, match="uncommitted"):
        pipe(graph.view(), atomic=True)
    graph.commit("initial", "test")
    pipe(graph.view(), atomic=True)
    assert graph.current_state_id == 2 and graph.get_node_attr(0, "b") == 2
    assert not graph.automatic_publication and not graph.has_uncommitted_changes()


def test_default_pipeline_keeps_existing_per_algorithm_publication():
    graph = Graph()
    graph.add_node()
    Pipeline([assign("a", 1, True), assign("b", 2, True)])(graph.view())
    assert graph.current_state_id == 3


def test_atomic_outer_read_only_pipeline_adds_no_version():
    graph = Graph()
    graph.add_node()
    builder = AlgorithmBuilder("read")
    builder.init_nodes(default=1)
    Pipeline([builder.build(), builder.build(atomic=True)])(graph.view(), atomic=True)
    assert graph.current_state_id == 1
