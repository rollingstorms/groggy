"""Explicit property-atomic builder runs stage, read their writes, then accept once."""
import json

import pytest

from groggy import AttrValue, Graph
from groggy.algorithms.base import AlgorithmHandle
from groggy.builder import AlgorithmBuilder


class AtomicSteps(AlgorithmHandle):
    def __init__(self, steps, atomic=True):
        self.steps = steps
        self.atomic = atomic

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({
            "atomic": self.atomic, "steps": self.steps,
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
    steps = [
        {"id": "core.init_nodes", "params": {"target": "values", "value": 2}},
        {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "prefix"}},
        {"id": "core.sample_nodes", "params": {"target": "sample", "count": 1}},
    ]
    with pytest.raises(RuntimeError, match="does not yet support atomic"):
        graph.view().apply(AtomicSteps(steps), persist=True)
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
    [{"id": "core.sample_nodes", "params": {"target": "sample", "count": 1}}],
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


def body_node(operation, inputs, output="result", **metadata):
    return {"id": output, "domain": "core", "op_type": operation,
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


@pytest.mark.parametrize("operator,expected", [
    ("eq", [-10, 10, -10]), ("ne", [10, -10, 10]),
    ("lt", [10, -10, -10]), ("le", [10, 10, -10]),
    ("gt", [-10, -10, 10]), ("ge", [-10, 10, 10]),
])
def test_public_conditional_message_pass_matches_normal_builder(operator, expected):
    graph = Graph()
    graph.add_nodes([{"score": -1}, {"score": 0}, {"score": 2}])
    before = graph.current_state_id
    for blocked in [False, True]:
        builder = AlgorithmBuilder("conditional block" if blocked else "conditional map")
        values = builder.load_attr("score")
        if blocked:
            with builder.message_pass(target=values) as messages:
                mask = builder.core.compare(values, operator, 0)
                chosen = builder.core.where(mask, 10, -10)
                messages.apply(chosen)
            chosen = values
        else:
            mask = builder.core.compare(values, operator, 0)
            chosen = builder.core.where(mask, 10, -10)
        builder.attach_as("blocked" if blocked else "mapped", chosen)
        graph.view().apply(builder.build(atomic=True), persist=True)
    assert graph.current_state_id == before + 2
    assert [graph.get_node_attr(n, "mapped") for n in range(3)] == expected
    assert [graph.get_node_attr(n, "blocked") for n in range(3)] == expected


def test_conditional_block_masks_match_existing_epsilon_rule():
    graph = Graph()
    epsilon = 2.220446049250313e-16
    graph.add_nodes([{"mask": mask} for mask in [0, epsilon / 2, epsilon, epsilon * 2, -1]])
    for blocked in [False, True]:
        builder = AlgorithmBuilder("mask parity")
        mask = builder.load_attr("mask")
        if blocked:
            with builder.message_pass(target=mask) as messages:
                selected = builder.core.where(mask, 7, 9)
                messages.apply(selected)
            selected = mask
        else:
            selected = builder.core.where(mask, 7, 9)
        builder.attach_as("blocked" if blocked else "mapped", selected)
        graph.view().apply(builder.build(atomic=True), persist=True)
    assert [graph.get_node_attr(n, "blocked") for n in range(5)] == [9, 9, 9, 7, 7]
    assert [graph.get_node_attr(n, "mapped") for n in range(5)] == [9, 9, 9, 7, 7]


@pytest.mark.parametrize("node", [
    body_node("compare", ["values", "values"], op="unknown"),
    body_node("compare", ["values", "values"], op=123),
    body_node("compare", ["values"]),
    body_node("where", ["values", "values", "values", "extra"]),
])
def test_malformed_block_conditionals_reject_before_prefix_writes(node):
    graph = Graph()
    graph.add_node(score=0)
    with pytest.raises(RuntimeError, match="does not yet support atomic"):
        graph.view().apply(AtomicSteps(increment_body() + [block_step([node])]), persist=True)
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "score") == 0


@pytest.mark.parametrize("mask,expected", [(1, "chosen"), (0, None), ("invalid", None)])
def test_block_where_resolves_only_selected_value_and_rolls_back_errors(mask, expected):
    graph = Graph()
    graph.add_node(mask=mask)
    steps = [
        {"id": "core.load_node_attr", "params": {"target": "values", "attr": "mask"}},
        {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "prefix"}},
        block_step([body_node("constant", [], "yes", value="chosen"),
                    body_node("where", ["values", "yes", "missing"])]),
        {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "answer"}},
    ]
    if expected is None:
        with pytest.raises(RuntimeError, match="missing|expected numeric"):
            graph.view().apply(AtomicSteps(steps), persist=True)
        assert graph.current_state_id == 1
        assert graph.get_node_attr(0, "prefix") is None
        assert graph.get_node_attr(0, "answer") is None
    else:
        graph.view().apply(AtomicSteps(steps), persist=True)
        assert graph.current_state_id == 2
        assert graph.get_node_attr(0, "answer") == expected
    assert graph.get_node_attr(0, "mask") == mask


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("ordered,expected", [(True, [0, 0, 0]), (False, [0, 1, 2])])
def test_public_message_pass_distinguishes_ordered_and_synchronous_updates(atomic, ordered, expected):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": 1}, {"score": 2}, {"score": 3}])
    graph.add_edges([(0, 1), (1, 2)])
    before = graph.current_state_id
    builder = AlgorithmBuilder("message visibility")
    values = builder.load_attr("score")
    with builder.message_pass(target=values, ordered=ordered, include_self=False, direction="in") as messages:
        messages.apply(builder.graph_ops.neighbor_agg(values, "sum"))
    builder.attach_as("answer", values)
    graph.view().apply(builder.build(atomic=atomic, optimize=False), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == expected
    assert [graph.get_node_attr(n, "score") for n in range(3)] == [1, 2, 3]
    assert graph.current_state_id == before + 1


@pytest.mark.parametrize("direction,expected", [
    ("in", [4, 2, 0]), ("out", [5, 0, 1]), ("undirected", [8, 2, 1]),
])
def test_synchronous_message_pass_counts_parallel_edges_and_self_loop_once(direction, expected):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": 1}, {"score": 2}, {"score": 3}])
    graph.add_edges([(0, 1), (0, 1), (0, 0), (2, 0)])
    steps = [
        {"id": "core.load_node_attr", "params": {"target": "values", "attr": "score"}},
        block_step([body_node("neighbor_agg", ["values"], agg="sum")], direction=direction, ordered=False),
        {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "answer"}},
    ]
    graph.view().apply(AtomicSteps(steps), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == expected


def test_nested_synchronous_iterations_use_prior_staged_state_once_per_iteration():
    graph = Graph(directed=True)
    graph.add_nodes([{"score": 1}, {"score": 2}, {"score": 3}])
    graph.add_edges([(0, 1), (1, 2)])
    before = graph.current_state_id
    body = [
        {"id": "core.load_node_attr", "params": {"target": "values", "attr": "score"}},
        block_step([body_node("neighbor_agg", ["values"], agg="sum")], ordered=False),
        {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "score"}},
    ]
    graph.view().apply(AtomicSteps([loop_step(body, 2)]), persist=True)
    assert [graph.get_node_attr(n, "score") for n in range(3)] == [0, 0, 1]
    assert graph.current_state_id == before + 1


def test_synchronous_block_late_node_failure_discards_prefix_and_partial_results():
    graph = Graph()
    graph.add_nodes([{"score": 1}, {"score": 2**63 - 1}])
    steps = [
        {"id": "core.load_node_attr", "params": {"target": "values", "attr": "score"}},
        {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "prefix"}},
        block_step([body_node("constant", [], "one", value=1),
                    body_node("add", ["values", "one"])], ordered=False),
        {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "answer"}},
    ]
    with pytest.raises(RuntimeError, match="Integer overflow"):
        graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == 1
    assert [graph.get_node_attr(n, "score") for n in range(2)] == [1, 2**63 - 1]
    assert all(graph.get_node_attr(n, "prefix") is None for n in range(2))
    assert all(graph.get_node_attr(n, "answer") is None for n in range(2))


@pytest.mark.parametrize("empty", [False, True])
def test_empty_synchronous_body_preserves_values_and_empty_graph_creates_no_state(empty):
    graph = Graph()
    if not empty:
        graph.add_node()
    before = graph.current_state_id
    steps = [
        {"id": "core.init_nodes", "params": {"target": "values", "value": "retained"}},
        block_step([], ordered=False),
        {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "answer"}},
    ]
    graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == before + (0 if empty else 1)
    if not empty:
        assert graph.get_node_attr(0, "answer") == "retained"


@pytest.mark.parametrize("directed,expected", [(True, [3, 0, 1]), (False, [4, 2, 1])])
def test_atomic_node_degree_preserves_parallel_loop_and_direction_conventions(directed, expected):
    graph = Graph(directed=directed)
    graph.add_nodes(3)
    graph.add_edges([(0, 1), (0, 1), (0, 0), (2, 0)])
    before = graph.current_state_id
    builder = AlgorithmBuilder("atomic degree")
    builder.attach_as("degree", builder.node_degrees(builder.init_nodes(default=0)))
    graph.view().apply(builder.build(atomic=True), persist=True)
    assert [graph.get_node_attr(n, "degree") for n in range(3)] == expected
    assert graph.current_state_id == before + 1


@pytest.mark.parametrize("kernel", ["core.weighted_degree", "core.edge_weight_sum"])
@pytest.mark.parametrize("persist", [False, True])
def test_weighted_builder_steps_read_staged_edge_properties(kernel, persist):
    graph = Graph(directed=True)
    graph.add_nodes(3)
    graph.add_edge(0, 1, weight=7)
    graph.add_edge(0, 0, weight=7)
    before = graph.current_state_id
    weight = 2.5000000001
    steps = [
        {"id": "core.init_edges", "params": {"target": "weights", "value": weight}},
        {"id": "core.attach_edge_attr", "params": {"source": "weights", "attr": "weight"}},
        {"id": kernel, "params": {"target": "sums", "weight_attr": "weight"}},
        {"id": "core.attach_node_attr", "params": {"source": "sums", "attr": "answer"}},
    ]
    _, profile = graph.view().apply(AtomicSteps([loop_step(steps, 2)]), persist=persist, return_profile=True)
    expected = [3 * weight, weight, 0] if persist else [21, 7, 0]
    if persist:
        assert [graph.get_node_attr(n, "answer") for n in range(3)] == pytest.approx(expected)
        assert graph.get_edge_attr(0, "weight") == weight
    else:
        assert dict(profile["outputs"]["node_attributes.answer"]) == dict(enumerate(expected))
        assert graph.get_edge_attr(0, "weight") == 7
    assert graph.current_state_id == before + (1 if persist else 0)


@pytest.mark.parametrize("kernel", ["core.weighted_degree", "core.edge_weight_sum"])
@pytest.mark.parametrize("invalid", [None, True, "bad", [1]])
def test_invalid_staged_weight_discards_edge_and_prefix_node_writes(kernel, invalid):
    graph = Graph()
    graph.add_nodes(2)
    graph.add_edge(0, 1, weight=7)
    before = graph.current_state_id
    steps = [
        {"id": "core.init_nodes", "params": {"target": "prefix", "value": 1}},
        {"id": "core.attach_node_attr", "params": {"source": "prefix", "attr": "prefix"}},
        {"id": "core.init_edges", "params": {"target": "weights", "value": invalid}},
        {"id": "core.attach_edge_attr", "params": {"source": "weights", "attr": "weight"}},
        {"id": kernel, "params": {"target": "result", "weight_attr": "weight"}},
    ]
    with pytest.raises(RuntimeError, match="weight.*must be numeric"):
        graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == before
    assert graph.get_edge_attr(0, "weight") == 7
    assert graph.get_node_attr(0, "prefix") is None
    assert not graph.has_uncommitted_changes()


@pytest.mark.parametrize("kernel", ["core.weighted_degree", "core.edge_weight_sum"])
@pytest.mark.parametrize("weight", [float("nan"), float("inf"), 1e308])
def test_nonfinite_weights_or_sum_overflow_fail_atomically(kernel, weight):
    graph = Graph()
    graph.add_node()
    graph.add_edge(0, 0, weight=weight)
    before = graph.current_state_id
    steps = [
        {"id": "core.init_nodes", "params": {"target": "n", "value": 1}},
        {"id": "core.attach_node_attr", "params": {"source": "n", "attr": "prefix"}},
        {"id": kernel, "params": {"target": "result", "weight_attr": "weight"}},
    ]
    with pytest.raises(RuntimeError, match="finite"):
        graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == before
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("kernel", ["core.weighted_degree", "core.edge_weight_sum"])
def test_incident_weight_defaults_and_signed_values(kernel):
    graph = Graph()
    graph.add_nodes(3)
    graph.add_edge(0, 1)
    graph.add_edge(0, 0, weight=-2)
    steps = [
        {"id": kernel, "params": {"target": "result", "weight_attr": "weight"}},
        {"id": "core.attach_node_attr", "params": {"source": "result", "attr": "answer"}},
    ]
    graph.view().apply(AtomicSteps(steps), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == [-3, 1, 0]


def test_missing_degree_source_is_an_error_and_discards_prefix():
    graph = Graph()
    graph.add_node()
    steps = [
        {"id": "core.init_nodes", "params": {"target": "n", "value": 1}},
        {"id": "core.attach_node_attr", "params": {"source": "n", "attr": "prefix"}},
        {"id": "core.node_degree", "params": {"source": "missing", "target": "result"}},
    ]
    with pytest.raises(RuntimeError, match="missing"):
        graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("persist", [False, True])
def test_triangle_step_counts_distinct_nodes_despite_parallel_edges_and_loops(atomic, persist):
    graph = Graph()
    graph.add_nodes(5)
    graph.add_edges([(0, 1), (1, 2), (2, 0), (0, 1), (0, 0), (2, 3)])
    before = graph.current_state_id
    steps = [
        {"id": "core.triangle_count", "params": {"target": "counts"}},
        {"id": "core.attach_node_attr", "params": {"source": "counts", "attr": "triangles"}},
    ]
    _, profile = graph.view().apply(AtomicSteps(steps, atomic=atomic), persist=persist, return_profile=True)
    expected = [1, 1, 1, 0, 0]
    if persist:
        assert [graph.get_node_attr(n, "triangles") for n in range(5)] == expected
    else:
        assert dict(profile["outputs"]["node_attributes.triangles"]) == dict(enumerate(expected))
    assert graph.current_state_id == before + int(persist)


@pytest.mark.parametrize("k,expected", [
    (0, [1, 1, 1, 1, 1]), (1, [1, 1, 1, 1, 0]),
    (2, [1, 1, 1, 0, 0]), (3, [0, 0, 0, 0, 0]),
])
def test_k_core_peels_cascading_nodes_with_multiedges_and_self_loops(k, expected):
    graph = Graph()
    graph.add_nodes(5)
    graph.add_edges([(0, 1), (1, 2), (2, 0), (0, 1), (0, 0), (2, 3)])
    before = graph.current_state_id
    steps = [
        {"id": "core.k_core_mark", "params": {"k": k, "target": "core"}},
        {"id": "core.attach_node_attr", "params": {"source": "core", "attr": "core"}},
    ]
    graph.view().apply(AtomicSteps([loop_step(steps, 2)]), persist=True)
    assert [graph.get_node_attr(n, "core") for n in range(5)] == expected
    assert graph.current_state_id == before + 1


@pytest.mark.parametrize("kernel,params", [
    ("core.triangle_count", {}), ("core.k_core_mark", {"k": 2}),
])
def test_structural_steps_reject_directed_graph_and_discard_prefix(kernel, params):
    graph = Graph(directed=True)
    graph.add_nodes(3)
    graph.add_edges([(0, 1), (1, 2), (2, 0)])
    before = graph.current_state_id
    steps = [
        {"id": "core.init_nodes", "params": {"target": "n", "value": 1}},
        {"id": "core.attach_node_attr", "params": {"source": "n", "attr": "prefix"}},
        {"id": kernel, "params": {**params, "target": "result"}},
    ]
    with pytest.raises(RuntimeError, match="require an undirected graph"):
        graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == before
    assert graph.get_node_attr(0, "prefix") is None


def test_negative_k_rejects_before_prefix_execution():
    graph = Graph()
    graph.add_node(score=0)
    with pytest.raises(RuntimeError, match="nonnegative k"):
        graph.view().apply(AtomicSteps(increment_body() + [
            {"id": "core.k_core_mark", "params": {"k": -1, "target": "core"}},
        ]), persist=True)
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "score") == 0


@pytest.mark.parametrize("kernel,params", [
    ("core.triangle_count", {}), ("core.k_core_mark", {"k": 0}),
])
def test_empty_structural_step_creates_no_state(kernel, params):
    graph = Graph()
    steps = [
        {"id": kernel, "params": {**params, "target": "result"}},
        {"id": "core.attach_node_attr", "params": {"source": "result", "attr": "answer"}},
    ]
    graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == 0


def predicate(operation, value, kind="Int"):
    return {"type": operation, "value": {"type": kind, "value": value}}


@pytest.mark.parametrize("persist", [False, True])
def test_property_filters_and_sort_read_staged_values_without_changing_selection(persist):
    graph = Graph()
    graph.add_nodes([{"score": 3}, {"score": 2}, {"score": 1}])
    graph.add_edges([(0, 1), (1, 2)])
    before = graph.current_state_id
    steps = [
        {"id": "core.init_nodes", "params": {"target": "n", "value": 10}},
        {"id": "core.attach_node_attr", "params": {"source": "n", "attr": "score"}},
        {"id": "core.init_edges", "params": {"target": "e", "value": 10}},
        {"id": "core.attach_edge_attr", "params": {"source": "e", "attr": "score"}},
        {"id": "core.filter_nodes_by_attr", "params": {"attr": "score", "predicate": predicate("gt", 5), "target": "nodes"}},
        {"id": "core.filter_edges_by_attr", "params": {"attr": "score", "predicate": predicate("gt", 5), "target": "edges"}},
        {"id": "core.sort_nodes_by_attr", "params": {"attr": "score", "order": "asc", "target": "ranks"}},
        {"id": "core.attach_node_attr", "params": {"source": "nodes", "attr": "selected"}},
        {"id": "core.attach_edge_attr", "params": {"source": "edges", "attr": "selected"}},
        {"id": "core.attach_node_attr", "params": {"source": "ranks", "attr": "rank"}},
    ]
    _, profile = graph.view().apply(AtomicSteps([loop_step(steps, 2)]), persist=persist, return_profile=True)
    if persist:
        assert [graph.get_node_attr(n, "selected") for n in range(3)] == [1, 1, 1]
        assert [graph.get_edge_attr(e, "selected") for e in range(2)] == [1, 1]
        assert [graph.get_node_attr(n, "rank") for n in range(3)] == [0, 1, 2]
    else:
        assert profile["outputs"]["node_attributes.selected"] == []
        assert profile["outputs"]["edge_attributes.selected"] == []
        assert dict(profile["outputs"]["node_attributes.rank"]) == {0: 2, 1: 1, 2: 0}
    assert graph.current_state_id == before + int(persist)
    assert graph.node_count() == 3
    assert graph.edge_count() == 2


@pytest.mark.parametrize("order,expected", [("asc", [1, 2, 0, 3]), ("desc", [0, 2, 1, 3])])
def test_sort_ranks_put_absent_and_null_last_with_id_ties(order, expected):
    graph = Graph()
    graph.add_nodes([{"score": 2}, {"score": None}, {"score": 1}, {}])
    steps = [
        {"id": "core.sort_nodes_by_attr", "params": {"attr": "score", "order": order, "target": "ranks"}},
        {"id": "core.attach_node_attr", "params": {"source": "ranks", "attr": "rank"}},
    ]
    graph.view().apply(AtomicSteps(steps), persist=True)
    assert [graph.get_node_attr(n, "rank") for n in range(4)] == expected


def test_exact_top_k_and_range_preserve_large_integer_float_order():
    graph = Graph()
    base = 2**53
    graph.add_nodes([{"score": base + 1}, {"score": float(base)}, {"score": base + 2}, {"score": float(base)}])
    steps = [
        {"id": "core.load_node_attr", "params": {"attr": "score", "target": "values"}},
        {"id": "core.top_k", "params": {"source": "values", "k": 3, "order": "asc", "target": "top"}},
        {"id": "core.filter_nodes_by_attr", "params": {"attr": "score", "predicate": {"type": "range", "min": float(base), "max": float(base)}, "target": "equal"}},
        {"id": "core.attach_node_attr", "params": {"source": "top", "attr": "rank"}},
        {"id": "core.attach_node_attr", "params": {"source": "equal", "attr": "equal"}},
    ]
    graph.view().apply(AtomicSteps(steps), persist=True)
    assert [graph.get_node_attr(n, "rank") for n in range(4)] == [2, 0, None, 1]
    assert [graph.get_node_attr(n, "equal") for n in range(4)] == [None, 1, None, 1]


@pytest.mark.parametrize("kernel", ["core.sort_nodes_by_attr", "core.top_k"])
@pytest.mark.parametrize("values", [[1, "two"], [1, True], [[1], [2]], [float("nan"), 1]])
def test_invalid_ordering_discards_prefix_without_inventing_equal_values(kernel, values):
    graph = Graph()
    graph.add_nodes([{"score": value} for value in values])
    steps = [
        {"id": "core.init_nodes", "params": {"target": "n", "value": 1}},
        {"id": "core.attach_node_attr", "params": {"source": "n", "attr": "prefix"}},
        {"id": "core.load_node_attr", "params": {"attr": "score", "target": "values"}},
        {"id": kernel, "params": {"attr": "score", "source": "values", "k": 2, "target": "result"}},
    ]
    with pytest.raises(RuntimeError, match="Ordering"):
        graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "prefix") is None


def test_null_filter_distinguishes_absent_and_present_null_and_returns_sparse_mask():
    graph = Graph()
    graph.add_nodes([{}, {"score": None}, {"score": 1}])
    steps = [
        {"id": "core.filter_nodes_by_attr", "params": {"attr": "score", "predicate": {"type": "eq", "value": {"type": "None"}}, "target": "mask"}},
        {"id": "core.attach_node_attr", "params": {"source": "mask", "attr": "selected"}},
    ]
    graph.view().apply(AtomicSteps(steps), persist=True)
    assert [graph.get_node_attr(n, "selected") for n in range(3)] == [None, 1, None]


def test_reversed_range_rejects_before_prefix_execution():
    graph = Graph()
    graph.add_node(score=0)
    with pytest.raises(RuntimeError, match="range requires ordered"):
        graph.view().apply(AtomicSteps(increment_body() + [
            {"id": "core.filter_nodes_by_attr", "params": {"attr": "score", "target": "mask", "predicate": {"type": "range", "min": 2, "max": 1}}},
        ]), persist=True)
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "score") == 0
