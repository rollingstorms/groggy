"""Checked deterministic mask samplers compose with property-atomic builders."""
import json

import pytest
from groggy import Graph
from groggy.algorithms.base import AlgorithmHandle
from groggy.builder import AlgorithmBuilder


class AtomicSteps(AlgorithmHandle):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        from groggy import AttrValue
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({"atomic": True, "steps": self.steps}))}}


def fixture():
    graph = Graph(directed=True)
    graph.add_nodes([{} for _ in range(10)])
    for node in range(10):
        graph.add_edge(node, (node + 1) % 10)
    return graph


def attach(kind="nodes", source="sample", attr="sampled"):
    return {"id": "core.attach_node_attr" if kind == "nodes" else "core.attach_edge_attr", "params": {"source": source, "attr": attr}}


def selection(graph, kind, attr="sampled"):
    read = graph.get_node_attr if kind == "nodes" else graph.get_edge_attr
    return {entity for entity in range(10) if read(entity, attr) == 1}


@pytest.mark.parametrize("kind", ["nodes", "edges"])
@pytest.mark.parametrize("control,expected", [({"count": 0}, 0), ({"count": 4}, 4), ({"fraction": 0.25}, 3), ({"fraction": 1}, 10)])
@pytest.mark.parametrize("persist", [False, True])
def test_count_fraction_masks_have_exact_cardinality_and_one_publication(kind, control, expected, persist):
    graph = fixture()
    before = graph.current_state_id
    steps = [{"id": "core.sample_" + kind, "params": {"target": "sample", "seed": 43, **control}}, attach(kind)]
    _, profile = graph.view().apply(AtomicSteps(steps), persist=persist, return_profile=True)
    if persist:
        assert len(selection(graph, kind)) == expected
    else:
        key = "node_attributes.sampled" if kind == "nodes" else "edge_attributes.sampled"
        assert len(profile["outputs"][key]) == expected
        assert not selection(graph, kind)
    assert graph.current_state_id == before + int(persist and expected > 0)


@pytest.mark.parametrize("kind", ["nodes", "edges"])
@pytest.mark.parametrize("k", [0, 3, 10, 1000000000])
def test_reservoir_caps_to_source_population_including_zero_values(kind, k):
    graph = fixture()
    before = graph.current_state_id
    steps = [{"id": "core.init_nodes" if kind == "nodes" else "core.init_edges", "params": {"target": "source", "value": 0}},
             {"id": "core.reservoir_sample", "params": {"source": "source", "target": "sample", "entity_type": kind, "k": k, "seed": 2}}, attach(kind)]
    graph.view().apply(AtomicSteps(steps), persist=True)
    assert len(selection(graph, kind)) == min(k, 10)
    assert graph.current_state_id == before + int(k > 0)


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_repeated_seeded_samples_are_stable_and_match_reservoir_over_the_same_ids(kind):
    graph = fixture()
    steps = [{"id": "core.sample_" + kind, "params": {"target": "sample", "count": 4, "seed": 7}}, attach(kind),
             {"id": "core.sample_" + kind, "params": {"target": "all", "count": 10, "seed": 99}},
             {"id": "core.reservoir_sample", "params": {"source": "all", "target": "again", "entity_type": kind, "k": 4, "seed": 7}}, attach(kind, "again", "again")]
    graph.view().apply(AtomicSteps(steps), persist=True)
    expected = selection(graph, kind)
    assert expected == selection(graph, kind, "again")
    for _ in range(3):
        graph.view().apply(AtomicSteps(steps), persist=True)
        assert selection(graph, kind) == expected


@pytest.mark.parametrize("kind", ["nodes", "edges"])
@pytest.mark.parametrize("control", [
    {"count": -1}, {"count": 11}, {"fraction": -0.1}, {"fraction": 1.1},
    {"count": 1, "fraction": 0.5}, {"count": "3"}, {"fraction": "0.2"},
    {"count": True}, {"count": 1, "seed": -1}, {"count": 1, "seed": "7"}, {},
])
def test_invalid_sampling_controls_do_not_publish_prefix_writes(kind, control):
    graph = fixture()
    before = graph.current_state_id
    steps = [{"id": "core.init_nodes", "params": {"target": "prefix", "value": 1}}, attach("nodes", "prefix", "prefix"),
             {"id": "core.sample_" + kind, "params": {"target": "sample", **control}}, attach(kind)]
    with pytest.raises(RuntimeError):
        graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == before
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("params", [{"k": -1}, {"k": "2"}, {"k": 2, "entity_type": 2}, {"k": 2, "entity_type": "typo"}, {"k": 2, "seed": -1}])
def test_reservoir_controls_are_strict(params):
    graph = fixture()
    before = graph.current_state_id
    with pytest.raises(RuntimeError):
        graph.view().apply(AtomicSteps([{"id": "core.reservoir_sample", "params": {"source": "missing", "target": "sample", **params}}]), persist=True)
    assert graph.current_state_id == before


@pytest.mark.parametrize("use_ir", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
def test_public_node_sampler_preserves_seed_and_controls_in_both_lowering_paths(use_ir, optimize, atomic):
    graph = fixture()
    builder = AlgorithmBuilder("sample_mask", use_ir=use_ir)
    sampled = builder.sample_nodes(count=4, seed=19)
    builder.attach_as("sampled", sampled)
    before = graph.current_state_id
    graph.view().apply(builder.build(atomic=atomic, optimize=optimize), persist=True)
    assert len(selection(graph, "nodes")) == 4
    assert graph.current_state_id == before + 1


def test_sampling_in_nested_loops_shares_atomic_boundary_and_later_errors_rollback():
    graph = fixture()
    before = graph.current_state_id
    body = [{"id": "core.sample_nodes", "params": {"target": "sample", "count": 3, "seed": 1}}, attach()]
    loop = {"id": "iter.loop", "params": {"iterations": 2, "body": body}}
    graph.view().apply(AtomicSteps([loop]), persist=True)
    assert graph.current_state_id == before + 1
    expected = selection(graph, "nodes")
    before = graph.current_state_id
    failing = {"id": "core.sample_edges", "params": {"target": "sample", "count": 11}}
    with pytest.raises(RuntimeError):
        graph.view().apply(AtomicSteps([loop, failing]), persist=True)
    assert graph.current_state_id == before
    assert selection(graph, "nodes") == expected
