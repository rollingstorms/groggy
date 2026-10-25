"""Bounded sampler selectors, induction modes and property atomic composition."""
import json
import warnings

import pytest
import groggy as gr
from groggy.algorithms.base import AlgorithmHandle
from groggy.builder import AlgorithmBuilder


class Steps(AlgorithmHandle):
    def __init__(self, steps, atomic=False):
        self.steps = steps
        self.atomic = atomic

    @property
    def id(self):
        return "builder.step_pipeline" if self.atomic else "builder.sample_pipeline"

    def to_spec(self):
        return {"id": self.id, "params": {"steps": gr.AttrValue(json.dumps({"atomic": self.atomic, "steps": self.steps}))}}


def fixture(directed=True):
    graph = gr.Graph(directed=directed)
    graph.add_nodes([{} for _ in range(4)])
    for source, target in [(0, 1), (1, 2), (2, 3), (1, 1), (1, 2)]:
        graph.add_edge(source, target)
    return graph


def emit(source, induced=True, mode="per_item"):
    return {"id": "sample.emit_subgraphs", "params": {"source": source, "target": "__sample_output__", "induced": induced, "mode": mode}}


def contents(array):
    return [(sorted(item.node_ids), sorted(item.edge_ids)) for item in array]


@pytest.mark.parametrize("kind,expected", [
    ("nodes", [([0], []), ([1], [3]), ([2], []), ([3], [])]),
    ("edges", [([0, 1], [0]), ([1, 2], [1]), ([2, 3], [2]), ([1], [3]), ([1, 2], [4])]),
])
def test_iteration_is_stable_and_keeps_edge_identity(kind, expected):
    graph = fixture()
    before = graph.current_state_id
    samples = graph.view().sample(Steps([{"id": "sample.iterate_" + kind, "params": {"target": "seeds"}}, emit("seeds", False)]))
    assert contents(samples) == expected
    assert graph.current_state_id == before


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("hops", [0, 1, 1000000])
def test_neighborhoods_stay_inside_selected_nodes_and_selected_edges(directed, hops):
    graph = fixture(directed)
    selected = graph.induced_subgraph([1, 2])
    steps = [{"id": "sample.iterate_nodes", "params": {"target": "seeds"}},
             {"id": "sample.neighbors", "params": {"source": "seeds", "target": "expanded", "hops": hops}}, emit("expanded", False)]
    samples = contents(selected.sample(Steps(steps)))
    assert all(set(nodes) <= {1, 2} and set(edges) <= {1, 3, 4} for nodes, edges in samples)
    if hops == 0:
        assert samples == [([1], [3]), ([2], [])]
    elif directed:
        assert samples == [([1, 2], [1, 3, 4]), ([2], [])]
    else:
        assert samples == [([1, 2], [1, 3, 4]), ([1, 2], [1, 3, 4])]


@pytest.mark.parametrize("induced", [False, True])
def test_per_item_edge_emission_honors_induced_and_parallel_edges(induced):
    graph = fixture()
    steps = [{"id": "core.sample_edges", "params": {"target": "mask", "fraction": 1}}, emit("mask", induced)]
    samples = contents(graph.view().sample(Steps(steps)))
    assert samples[1] == ([1, 2], [1, 3, 4] if induced else [1])
    assert samples[4] == ([1, 2], [1, 3, 4] if induced else [4])


@pytest.mark.parametrize("use_ir", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_public_mask_to_neighborhood_pipeline_works_in_both_lowering_paths(use_ir, optimize, kind):
    graph = fixture()
    builder = AlgorithmBuilder("mask_neighbors", use_ir=use_ir)
    seeds = builder.sample_nodes(count=2, seed=19) if kind == "nodes" else builder.sample_edges(count=2, seed=19)
    neighbors = builder.neighbors(seeds, hops=1)
    builder.emit_subgraphs(neighbors, induced=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        samples = graph.view().sample(builder.build_sampler(optimize=optimize))
    assert not any("doesn't attach" in str(warning.message) for warning in caught)
    assert len(samples) == 2
    assert all(sample.node_count() >= 1 for sample in samples)


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_iteration_and_expansion_are_certified_inside_atomic_property_loops(kind):
    graph = fixture()
    before = graph.current_state_id
    body = [{"id": "sample.iterate_" + kind, "params": {"target": "seeds"}},
            {"id": "sample.neighbors", "params": {"source": "seeds", "target": "expanded", "hops": 1}}, emit("expanded"),
            {"id": "core.init_nodes", "params": {"target": "values", "value": 1}},
            {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "answer"}}]
    graph.view().apply(Steps([{"id": "iter.loop", "params": {"iterations": 2, "body": body}}], atomic=True), persist=True)
    assert graph.current_state_id == before + 1
    assert [graph.get_node_attr(n, "answer") for n in range(4)] == [1] * 4


@pytest.mark.parametrize("step", [
    {"id": "sample.neighbors", "params": {"source": "seeds", "target": "out", "hops": -1}},
    {"id": "sample.neighbors", "params": {"source": "seeds", "target": "out", "hops": "1"}},
    {"id": "sample.emit_subgraphs", "params": {"source": "seeds", "target": "out", "mode": 2}},
    {"id": "sample.emit_subgraphs", "params": {"source": "seeds", "target": "out", "mode": "typo"}},
    {"id": "sample.emit_subgraphs", "params": {"source": "seeds", "target": "out", "induced": 1}},
])
def test_supplied_neighborhood_and_emission_controls_are_strict(step):
    graph = fixture()
    before = graph.current_state_id
    with pytest.raises(RuntimeError):
        graph.view().apply(Steps([step], atomic=True), persist=True)
    assert graph.current_state_id == before


def test_empty_map_unified_and_empty_array_unified_preserve_cardinality_conventions():
    graph = gr.Graph()
    masks = [{"id": "core.sample_nodes", "params": {"target": "mask", "count": 0}}, emit("mask", mode="unified")]
    arrays = [{"id": "sample.iterate_nodes", "params": {"target": "seeds"}}, emit("seeds", mode="unified")]
    assert contents(graph.view().sample(Steps(masks))) == [([], [])]
    assert contents(graph.view().sample(Steps(arrays))) == []


def test_nested_edge_selection_cannot_reintroduce_parallel_edges_or_self_loops():
    graph = fixture()
    parent = graph.subgraph_from_edges([1])
    steps = [{"id": "sample.iterate_nodes", "params": {"target": "seeds"}},
             {"id": "sample.neighbors", "params": {"source": "seeds", "target": "expanded", "hops": 1}}, emit("expanded")]
    assert contents(parent.sample(Steps(steps))) == [([1, 2], [1]), ([2], [])]
