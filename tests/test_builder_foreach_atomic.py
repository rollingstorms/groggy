"""Nested sampler output cardinality and property/owned-result boundaries."""
import json
import warnings

import pytest
import groggy as gr
from groggy.algorithms.base import AlgorithmHandle
from groggy.builder import AlgorithmBuilder


class Steps(AlgorithmHandle):
    def __init__(self, steps, sampler=False):
        self.steps = steps
        self.sampler = sampler

    @property
    def id(self):
        return "builder.sample_pipeline" if self.sampler else "builder.step_pipeline"

    def to_spec(self):
        return {"id": self.id, "params": {"steps": gr.AttrValue(json.dumps({"atomic": not self.sampler, "steps": self.steps}))}}


def fixture():
    graph = gr.Graph(directed=True)
    graph.add_nodes([{"score": 0} for _ in range(4)])
    for node in range(3):
        graph.add_edge(node, node + 1)
    return graph


def inputs(kind="edges"):
    return {"id": "sample.iterate_" + kind, "params": {"target": "seeds"}}


def each(body, source="seeds", target="mapped"):
    return {"id": "sample.for_each", "params": {"source": source, "target": target, "body": body}}


def emit(source, mode="per_item", target="__sample_output__"):
    return {"id": "sample.emit_subgraphs", "params": {"source": source, "target": target, "mode": mode, "induced": False}}


def increment_body():
    return [
        {"id": "core.load_node_attr", "params": {"attr": "score", "target": "values"}},
        {"id": "core.map_nodes", "params": {"source": "values", "target": "next", "expr": {"type": "binary_op", "op": "add", "left": {"type": "var", "name": "value"}, "right": {"type": "const", "value": 1}}}},
        {"id": "core.attach_node_attr", "params": {"source": "next", "attr": "score"}},
    ]


@pytest.mark.parametrize("kind,expected", [("nodes", [[0], [1], [2], [3]]), ("edges", [[0], [1], [1], [2], [2], [3]])])
def test_foreach_keeps_all_explicit_child_results_in_item_order(kind, expected):
    graph = fixture()
    body = [inputs("nodes"), emit("seeds")]
    samples = graph.view().sample(Steps([inputs(kind), each(body), emit("mapped")], sampler=True))
    assert [sorted(sample.node_ids) for sample in samples] == expected


@pytest.mark.parametrize("explicit", [False, True])
def test_explicit_empty_child_outputs_are_not_input_fallback(explicit):
    graph = fixture()
    body = [{"id": "core.sample_nodes", "params": {"target": "empty", "count": 0}}, emit("empty")] if explicit else []
    samples = graph.view().sample(Steps([inputs(), each(body), emit("mapped")], sampler=True))
    assert len(samples) == (0 if explicit else 3)


@pytest.mark.parametrize("persist", [False, True])
def test_overlapping_items_share_atomic_properties_or_merge_owned_columns(persist):
    graph = fixture()
    before = graph.current_state_id
    _, profile = graph.view().apply(Steps([inputs(), each(increment_body())]), persist=persist, return_profile=True)
    if persist:
        assert [graph.get_node_attr(node, "score") for node in range(4)] == [1, 2, 2, 1]
    else:
        # Transient property reads still use the owner, without simulating staged writes.
        assert profile["outputs"]["node_attributes.score"] == [(0, 1), (1, 1), (2, 1), (3, 1)]
        assert [graph.get_node_attr(node, "score") for node in range(4)] == [0] * 4
    assert graph.current_state_id == before + int(persist)


def test_owned_columns_retain_prior_rows_and_later_item_assignments():
    graph = fixture()
    prefix = [{"id": "core.init_nodes", "params": {"target": "prefix", "value": 9}},
              {"id": "core.attach_node_attr", "params": {"source": "prefix", "attr": "score"}}]
    # Only nodes 1 and 2 participate, preserving prior rows 0 and 3 in owned output.
    selected = {"id": "sample.emit_subgraphs", "params": {"source": "mask", "target": "seeds", "mode": "per_item", "induced": False}}
    # Use direct graph property filter to select those middle nodes.
    for node in range(4):
        graph.set_node_attr(node, "middle", node in [1, 2])
    mask = {"id": "core.filter_nodes_by_attr", "params": {"attr": "middle", "target": "mask", "predicate": {"type": "eq", "value": {"type": "Bool", "value": True}}}}
    _, profile = graph.view().apply(Steps(prefix + [mask, selected, each(increment_body())]), persist=False, return_profile=True)
    assert profile["outputs"]["node_attributes.score"] == [(0, 9), (1, 1), (2, 1), (3, 9)]


def test_histogram_outputs_are_retained_for_every_item_with_stable_keys():
    graph = fixture()
    body = [{"id": "core.load_node_attr", "params": {"attr": "score", "target": "values"}},
            {"id": "core.histogram", "params": {"source": "values", "target": "hist", "bins": 2}}]
    before = graph.current_state_id
    _, profile = graph.view().apply(Steps([inputs(), each(body)]), persist=False, return_profile=True)
    assert profile["outputs"] == {f"for_each.mapped.{index}.histograms.hist": [2, 0] for index in range(3)}
    assert graph.current_state_id == before


@pytest.mark.parametrize("empty", [False, True])
def test_recursive_preflight_rejects_temporal_descendants_even_without_items(empty):
    graph = gr.Graph() if empty else fixture()
    before = graph.current_state_id
    nested = each([{"id": "temporal.diff_nodes", "params": {}}], source="inner")
    body = [inputs("nodes"), nested]
    with pytest.raises(RuntimeError, match="does not yet support atomic"):
        graph.view().apply(Steps([inputs(), each(body)]), persist=True)
    assert graph.current_state_id == before


def test_late_item_failure_discards_all_staged_item_properties():
    graph = fixture()
    graph.set_node_attr(2, "score", "bad")
    before = graph.current_state_id
    with pytest.raises(RuntimeError):
        graph.view().apply(Steps([inputs("nodes"), each(increment_body())]), persist=True)
    assert graph.current_state_id == before
    assert [graph.get_node_attr(node, "score") for node in range(4)] == [0, 0, "bad", 0]


@pytest.mark.parametrize("use_ir", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_public_each_preserves_child_array_cardinality(use_ir, optimize):
    graph = fixture()
    builder = AlgorithmBuilder("flat_each", use_ir=use_ir)
    seeds = builder.iterate_edges()
    mapped = builder.each(seeds, lambda child: child.emit_subgraphs(child.iterate_nodes(), induced=False))
    builder.emit_subgraphs(mapped, induced=False)
    samples = graph.view().sample(builder.build_sampler(optimize=optimize))
    assert [sorted(sample.node_ids) for sample in samples] == [[0], [1], [1], [2], [2], [3]]


def test_public_each_properties_compose_with_atomic_loops_and_fresh_child_variables():
    graph = fixture()
    builder = AlgorithmBuilder("property_each")
    def write(child):
        child.attach_as("answer", child.init_nodes(default=7))
    with builder.iterate(2):
        builder.each(builder.iterate_edges(), write)
    before = graph.current_state_id
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph.view().apply(builder.build(atomic=True), persist=True)
    assert not any("doesn't attach" in str(warning.message) for warning in caught)
    assert graph.current_state_id == before + 1
    assert [graph.get_node_attr(node, "answer") for node in range(4)] == [7] * 4
