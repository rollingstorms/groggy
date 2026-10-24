"""Explicit property-atomic builder runs stage, read their writes, then accept once."""
import json

import pytest

from groggy import AttrValue, Graph
from groggy.algorithms.base import AlgorithmHandle
from groggy.builder import AlgorithmBuilder


class AtomicSteps(AlgorithmHandle):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({
            "atomic": True, "steps": self.steps,
        }))}}


def test_atomic_builder_reads_staged_values_and_accepts_all_columns_once():
    graph = Graph()
    nodes = graph.add_nodes([{"score": 7}, {"score": 8}])
    before = graph.current_state_id
    prior = graph.read_view()
    builder = AlgorithmBuilder("atomic properties")
    builder.attach_as("score", builder.init_nodes(default=1.0000000001))
    builder.attach_as("copied", builder.load_attr("score"))
    builder.attach_as("score", builder.init_nodes(default=2.0000000001))
    graph.view().apply(builder.build(atomic=True), persist=True)
    assert graph.current_state_id == before + 1
    assert not graph.has_uncommitted_changes()
    for node in nodes:
        assert graph.get_node_attr(node, "score") == 2.0000000001
        assert graph.get_node_attr(node, "copied") == 1.0000000001
    assert prior.node_attr(0, "score") == 7
    snapshot = graph.snapshot_at_commit(before + 1)
    assert snapshot.node_attr(0, "copied").value == 1.0000000001
    metadata = graph.publication_metadata(before + 1)
    assert metadata["change_times"]
    assert all(row["effective_at"] == metadata["recorded_at"] for row in metadata["change_times"])
    assert graph.commit("checkpoint", "test") == before + 1


def test_late_runtime_failure_discards_earlier_attachments():
    graph = Graph()
    graph.add_node(existing=3)
    builder = AlgorithmBuilder("failure")
    values = builder.init_nodes(default=4)
    builder.attach_as("valid", values)
    builder.attach_as("entity_type", values)
    with pytest.raises(RuntimeError, match="entity_type"):
        graph.view().apply(builder.build(atomic=True), persist=True)
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "valid") is None
    assert graph.get_node_attr(0, "existing") == 3
    assert not graph.has_uncommitted_changes()
    assert graph.add_node() == 1


def test_atomic_mixed_edge_node_columns_read_prior_edge_attachments():
    graph = Graph()
    graph.add_nodes(2)
    graph.add_edge(0, 1, w=7)
    before = graph.current_state_id
    handle = AtomicSteps([
        {"id": "core.init_nodes", "params": {"target": "n", "value": 2}},
        {"id": "core.init_edges", "params": {"target": "e", "value": 3.0000000001}},
        {"id": "core.attach_node_attr", "params": {"source": "n", "attr": "result"}},
        {"id": "core.attach_edge_attr", "params": {"source": "e", "attr": "w"}},
        {"id": "core.load_edge_attr", "params": {"target": "copy", "attr": "w"}},
        {"id": "core.attach_edge_attr", "params": {"source": "copy", "attr": "copied"}},
    ])
    graph.view().apply(handle, persist=True)
    assert graph.current_state_id == before + 1
    assert graph.get_edge_attr(0, "w") == 3.0000000001
    assert graph.get_edge_attr(0, "copied") == 3.0000000001
    assert graph.get_node_attr(0, "result") == 2


def test_unsupported_step_rejects_before_any_attachment_executes():
    graph = Graph()
    graph.add_node()
    builder = AlgorithmBuilder("unsupported")
    values = builder.init_nodes(default=2)
    builder.attach_as("prefix", values)
    builder.node_degrees(values)
    with pytest.raises(RuntimeError, match="does not yet support atomic"):
        graph.view().apply(builder.build(atomic=True), persist=True)
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("empty", [False, True])
def test_no_attachments_and_empty_columns_create_no_version(empty):
    graph = Graph()
    if not empty:
        graph.add_node()
    before = graph.current_state_id
    builder = AlgorithmBuilder("no changes")
    values = builder.init_nodes(default=2)
    if empty:
        builder.attach_as("result", values)
    graph.view().apply(builder.build(atomic=True), persist=True)
    assert graph.current_state_id == before
    assert not graph.has_uncommitted_changes()


def test_explicit_atomic_mode_on_clean_manual_graph_publishes_once():
    graph = Graph(manual=True)
    graph.add_node()
    builder = AlgorithmBuilder("manual")
    builder.attach_as("result", builder.init_nodes(default=9))
    handle = builder.build(atomic=True)
    with pytest.raises(RuntimeError, match="uncommitted"):
        graph.view().apply(handle, persist=True)
    graph.commit("initial", "test")
    graph.view().apply(handle, persist=True)
    assert graph.current_state_id == 2
    assert graph.get_node_attr(0, "result") == 9
    assert not graph.automatic_publication
    assert not graph.has_uncommitted_changes()


def test_nonpersistent_atomic_spec_keeps_existing_transient_read_semantics():
    graph = Graph()
    graph.add_node(score=7)
    builder = AlgorithmBuilder("transient")
    builder.attach_as("score", builder.init_nodes(default=9))
    builder.attach_as("copy", builder.load_attr("score"))
    _, profile = graph.view().apply(builder.build(atomic=True), persist=False, return_profile=True)
    assert profile["outputs"]["node_attributes.score"] == [(0, 9)]
    assert profile["outputs"]["node_attributes.copy"] == [(0, 7)]
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "score") == 7


def loop_step(body, iterations=2, **params):
    return {"id": "iter.loop", "params": {
        "iterations": iterations, "body": body, **params,
    }}


def increment_body():
    return [
        {"id": "core.load_node_attr", "params": {"attr": "score", "target": "old"}},
        {"id": "core.init_nodes", "params": {"value": 1, "target": "one"}},
        {"id": "core.add", "params": {"left": "old", "right": "one", "target": "next"}},
        {"id": "core.attach_node_attr", "params": {"attr": "score", "source": "next"}},
    ]


def test_nested_atomic_loops_read_prior_iterations_and_publish_once():
    graph = Graph()
    graph.add_node(score=0)
    prior = graph.read_view()
    graph.view().apply(AtomicSteps([loop_step([loop_step(increment_body(), 3)], 2)]), persist=True)
    assert graph.get_node_attr(0, "score") == 6
    assert prior.node_attr(0, "score") == 0
    assert graph.current_state_id == 2
    assert not graph.has_uncommitted_changes()


@pytest.mark.parametrize("body", [
    [{"id": "core.node_degree", "params": {}}],
    [{"params": {}}],
])
def test_zero_iteration_nested_loop_still_rejects_unsupported_or_malformed_body(body):
    graph = Graph()
    graph.add_node(score=0)
    with pytest.raises(RuntimeError, match="does not yet support atomic"):
        graph.view().apply(AtomicSteps(increment_body() + [loop_step([loop_step(body, 0)], 0)]), persist=True)
    assert graph.get_node_attr(0, "score") == 0
    assert graph.current_state_id == 1


def test_atomic_loop_rejects_alternate_batch_plan_before_prefix_writes():
    graph = Graph()
    graph.add_node(score=0)
    with pytest.raises(RuntimeError, match="does not yet support atomic"):
        graph.view().apply(AtomicSteps(increment_body() + [loop_step([], batch_plan={})]), persist=True)
    assert graph.get_node_attr(0, "score") == 0
    assert graph.current_state_id == 1


def test_empty_atomic_loops_publish_nothing_and_transient_loops_read_owner():
    graph = Graph()
    graph.add_node(score=0)
    graph.view().apply(AtomicSteps([loop_step(increment_body(), 0)]), persist=True)
    assert graph.current_state_id == 1
    _, profile = graph.view().apply(AtomicSteps([loop_step(increment_body(), 3)]), persist=False, return_profile=True)
    assert profile["outputs"]["node_attributes.score"] == [(0, 1)]
    assert graph.get_node_attr(0, "score") == 0
    assert graph.current_state_id == 1


def test_later_loop_iteration_failure_discards_earlier_iterations():
    graph = Graph()
    graph.add_node(score=2)
    body = [
        {"id": "core.load_node_attr", "params": {"attr": "score", "target": "old"}},
        {"id": "core.init_nodes", "params": {"value": 1, "target": "one"}},
        {"id": "core.sub", "params": {"left": "old", "right": "one", "target": "next"}},
        {"id": "core.attach_node_attr", "params": {"attr": "score", "source": "next"}},
        {"id": "core.div", "params": {"left": "one", "right": "next", "target": "ratio"}},
    ]
    with pytest.raises(RuntimeError, match="Loop iteration 2/3.*division by zero"):
        graph.view().apply(AtomicSteps([loop_step(body, 3)]), persist=True)
    assert graph.get_node_attr(0, "score") == 2
    assert graph.current_state_id == 1
    assert not graph.has_uncommitted_changes()
    graph.view().apply(AtomicSteps([loop_step(increment_body())]), persist=True)
    assert graph.get_node_attr(0, "score") == 4
    assert graph.current_state_id == 2


def test_public_builder_iteration_uses_atomic_loop_boundary():
    graph = Graph()
    graph.add_node(score=0)
    builder = AlgorithmBuilder("loop properties")
    with builder.iterate(3):
        builder.attach_as("score", builder.load_attr("score") + builder.init_nodes(default=1))
    graph.view().apply(builder.build(atomic=True), persist=True)
    assert graph.get_node_attr(0, "score") == 3
    assert graph.current_state_id == 2


def block_step(nodes, *, direction="in", mode="message_pass", **options):
    return {"id": "core.execution_block", "params": {
        "mode": mode, "target": "values",
        "options": {"include_self": False, "direction": direction, **options},
        "body": {"nodes": nodes},
    }}


def body_node(op, inputs, output="result", **metadata):
    return {"id": output, "domain": "core", "op_type": op,
            "inputs": inputs, "output": output, "metadata": metadata}


@pytest.mark.parametrize("kernel,params,expected", [
    ("core.fused_axpy", {"a": "two", "x": "values", "b": "four", "y": "two"}, 14),
    ("core.fused_madd", {"a": "values", "b": "two", "c": "four"}, 10),
])
def test_atomic_fused_arithmetic_consumes_staged_property_loads(kernel, params, expected):
    graph = Graph()
    graph.add_nodes([{"score": 99}, {"score": 88}])
    steps = [
        {"id": "core.init_nodes", "params": {"target": "staged", "value": 3}},
        {"id": "core.attach_node_attr", "params": {"source": "staged", "attr": "score"}},
        {"id": "core.load_node_attr", "params": {"target": "values", "attr": "score"}},
        {"id": "core.init_nodes", "params": {"target": "two", "value": 2}},
        {"id": "core.init_nodes", "params": {"target": "four", "value": 4}},
        {"id": kernel, "params": {**params, "target": "result"}},
        {"id": "core.attach_node_attr", "params": {"source": "result", "attr": "answer"}},
    ]
    graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == 2
    assert [graph.get_node_attr(n, "answer") for n in [0, 1]] == [expected, expected]


@pytest.mark.parametrize("direction,expected", [
    ("in", [8, 8, 0]), ("out", [12, 0, 4]), ("undirected", [16, 8, 4]),
])
def test_atomic_fused_neighbors_respect_direction(direction, expected):
    graph = Graph(directed=True)
    graph.add_nodes(3)
    graph.add_edges([(0, 1), (0, 1), (0, 0), (2, 0)])
    before = graph.current_state_id
    steps = [
        {"id": "core.init_nodes", "params": {"target": "values", "value": 2}},
        {"id": "graph.fused_neighbor_mul_agg", "params": {
            "values": "values", "scalars": "values", "target": "result", "direction": direction}},
        {"id": "core.attach_node_attr", "params": {"source": "result", "attr": "answer"}},
    ]
    graph.view().apply(AtomicSteps([loop_step(steps, 1)]), persist=True)
    assert graph.current_state_id == before + 1
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == expected


@pytest.mark.parametrize("direction,expected", [
    ("in", [0, 0, 0]), ("out", [5, 0, 0]), ("undirected", [5, 5, 5]),
])
def test_atomic_message_pass_respects_direction_and_order(direction, expected):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": 1}, {"score": 2}, {"score": 3}])
    graph.add_edges([(0, 1), (0, 2)])
    before = graph.current_state_id
    steps = [
        {"id": "core.load_node_attr", "params": {"target": "values", "attr": "score"}},
        block_step([body_node("neighbor_agg", ["values"], agg="sum")], direction=direction),
        {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "answer"}},
    ]
    graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == before + 1
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == expected


@pytest.mark.parametrize("block", [
    block_step([body_node("where", ["a", "b"])]),
    block_step([], mode="streaming"),
    block_step([], ordered=False),
    block_step([], direction="typo"),
    block_step([body_node("attach_node_attr", [])]),
])
def test_atomic_block_rejects_uncertified_behavior_before_prefix_writes(block):
    graph = Graph()
    graph.add_node(score=0)
    with pytest.raises(RuntimeError, match="does not yet support atomic"):
        graph.view().apply(AtomicSteps(increment_body() + [block]), persist=True)
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "score") == 0


def test_atomic_block_overflow_discards_prefix_attachments():
    graph = Graph()
    graph.add_node(score=2**63 - 1)
    steps = [
        {"id": "core.load_node_attr", "params": {"target": "values", "attr": "score"}},
        {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "prefix"}},
        block_step([body_node("constant", [], "one", value=1), body_node("add", ["values", "one"])]),
    ]
    with pytest.raises(RuntimeError, match="Integer overflow"):
        graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "prefix") is None
    assert graph.get_node_attr(0, "score") == 2**63 - 1


@pytest.mark.parametrize("persist", [False, True])
def test_nested_block_reads_staged_loads_and_preserves_transient_owner_reads(persist):
    graph = Graph()
    graph.add_node(score=2)
    steps = [
        {"id": "core.load_node_attr", "params": {"target": "values", "attr": "score"}},
        block_step([body_node("constant", [], "one", value=1), body_node("add", ["values", "one"])]),
        {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "score"}},
    ]
    _, profile = graph.view().apply(AtomicSteps([loop_step(steps, 3)]), persist=persist, return_profile=True)
    assert graph.current_state_id == (2 if persist else 1)
    assert graph.get_node_attr(0, "score") == (5 if persist else 2)
    if not persist:
        assert profile["outputs"]["node_attributes.score"] == [(0, 3)]


def test_invalid_fused_direction_rejects_before_prefix_writes():
    graph = Graph()
    graph.add_node(score=0)
    step = {"id": "graph.fused_neighbor_mul_agg", "params": {
        "values": "values", "scalars": "values", "target": "result", "direction": "typo"}}
    with pytest.raises(RuntimeError, match="Invalid fused neighbor direction"):
        graph.view().apply(AtomicSteps(increment_body() + [step]), persist=True)
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "score") == 0


@pytest.mark.parametrize("direction,expected", [
    ("in", [4, 8, 0]), ("out", [5, 0, 5]), ("undirected", [8, 16, 8]),
])
def test_message_pass_counts_parallel_edges_and_self_loop_once(direction, expected):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": 1}, {"score": 2}, {"score": 3}])
    graph.add_edges([(0, 1), (0, 1), (0, 0), (2, 0)])
    steps = [
        {"id": "core.load_node_attr", "params": {"target": "values", "attr": "score"}},
        block_step([body_node("neighbor_agg", ["values"], agg="sum")], direction=direction),
        {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "answer"}},
    ]
    graph.view().apply(AtomicSteps(steps), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == expected


def test_public_message_pass_builder_supports_atomic_attachment():
    graph = Graph(directed=True)
    graph.add_nodes([{"label": 9}, {"label": 2}, {"label": 2}])
    graph.add_edges([(0, 1), (0, 2)])
    before = graph.current_state_id
    builder = AlgorithmBuilder("atomic messages")
    labels = builder.load_attr("label")
    with builder.message_pass(target=labels, include_self=True, direction="out") as messages:
        update = builder.core.mode(messages.pull(labels), tie_break="lowest")
        messages.apply(update)
    builder.attach_as("label", labels)
    graph.view().apply(builder.build(atomic=True), persist=True)
    assert graph.current_state_id == before + 1
    assert [graph.get_node_attr(n, "label") for n in range(3)] == [2, 2, 2]
