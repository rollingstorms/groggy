"""Checked reductions, owned histograms and standalone neighbor aggregation."""
import json
import statistics
import warnings

import pytest
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms.base import AlgorithmHandle


class AtomicSteps(AlgorithmHandle):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        from groggy import AttrValue
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({"atomic": True, "steps": self.steps}))}}


def loop_step(body, iterations):
    return {"id": "iter.loop", "params": {"iterations": iterations, "body": body}}


def load_values():
    return {"id": "core.load_node_attr", "params": {"attr": "score", "target": "values"}}


def scalar_steps(kernel, **params):
    return [load_values(),
            {"id": kernel, "params": {"source": "values", "target": "stat", **params}},
            {"id": "core.broadcast_scalar", "params": {"scalar": "stat", "reference": "values", "target": "result"}},
            {"id": "core.attach_node_attr", "params": {"source": "result", "attr": "answer"}}]


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("kernel,params,expected", [
    ("core.reduce_nodes", {"reducer": "sum"}, 10),
    ("core.reduce_nodes", {"reducer": "mean"}, 2.5),
    ("core.reduce_nodes", {"reducer": "min"}, 1),
    ("core.reduce_nodes", {"reducer": "max"}, 4),
    ("core.reduce_scalar", {"op": "sum"}, 10),
    ("core.reduce_scalar", {"op": "mean"}, 2.5),
    ("core.std", {}, statistics.stdev([1, 2, 3, 4])),
    ("core.median", {}, 2.5),
    ("core.quantile", {"q": 0.25}, 1.75),
])
def test_scalar_statistics_match_references_and_publish_once(kernel, params, expected, persist):
    graph = Graph()
    graph.add_nodes([{"score": value} for value in [1, 2, 3, 4]])
    before = graph.current_state_id
    _, profile = graph.view().apply(AtomicSteps(scalar_steps(kernel, **params)), persist=persist, return_profile=True)
    values = [graph.get_node_attr(n, "answer") for n in range(4)] if persist else [v for _, v in profile["outputs"]["node_attributes.answer"]]
    assert values == pytest.approx([expected] * 4)
    assert graph.current_state_id == before + int(persist)


@pytest.mark.parametrize("kernel,params,values,expected", [
    ("core.std", {}, [2**60, 2**60 + 1, 2**60 + 2], 1),
    ("core.reduce_nodes", {"reducer": "sum"}, [2**53 + 1, -float(2**53)], 1),
    ("core.reduce_scalar", {"op": "sum"}, [2**53 + 1, -float(2**53)], 1),
    ("core.reduce_nodes", {"reducer": "mean"}, [1e308, 1e308], 1e308),
    ("core.median", {}, [1e308, 1e308], 1e308),
    ("core.quantile", {"q": 0.5}, [-1e308, 1e308], 0),
])
def test_statistics_retain_integer_residuals_and_avoid_intermediate_overflow(kernel, params, values, expected):
    graph = Graph()
    graph.add_nodes([{"score": value} for value in values])
    graph.view().apply(AtomicSteps(scalar_steps(kernel, **params)), persist=True)
    assert graph.get_node_attr(0, "answer") == pytest.approx(expected)


@pytest.mark.parametrize("kernel", ["core.std", "core.median", "core.reduce_nodes"])
@pytest.mark.parametrize("invalid", [None, "bad", float("nan"), float("inf")])
def test_invalid_statistics_discard_prefix_writes(kernel, invalid):
    graph = Graph()
    graph.add_nodes([{"score": 1}, {"score": invalid}])
    steps = [{"id": "core.init_nodes", "params": {"target": "prefix", "value": 1}},
             {"id": "core.attach_node_attr", "params": {"source": "prefix", "attr": "prefix"}}] + scalar_steps(kernel)
    with pytest.raises(RuntimeError, match="finite numeric"):
        graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("kernel,expected", [("core.mode", "red"), ("core.entropy", 1.0)])
def test_categorical_ties_use_lowest_contributor_id_and_entropy_is_repeatable(kernel, expected):
    graph = Graph()
    graph.add_nodes([{"score": value} for value in ["red", "blue", "blue", "red"]])
    graph.view().apply(AtomicSteps(scalar_steps(kernel)), persist=True)
    assert graph.get_node_attr(0, "answer") == expected


@pytest.mark.parametrize("values,bins,expected", [
    ([0, 1, 2, 3], 2, [2, 2]), ([1, 1, 1], 4, [3, 0, 0, 0]),
    ([0.0, 1e-12], 2, [1, 1]), ([2**60, 2**60 + 1, 2**60 + 2], 2, [1, 2]),
    ([-1e308, 1e308], 2, [1, 1]), ([], 3, [0, 0, 0]),
])
def test_histograms_are_owned_count_vectors_without_graph_identity(values, bins, expected):
    graph = Graph()
    graph.add_nodes([{"score": value} for value in values])
    before = graph.current_state_id
    steps = [load_values(), {"id": "core.histogram", "params": {"source": "values", "target": "hist", "bins": bins}}]
    _, profile = graph.view().apply(AtomicSteps(steps), persist=True, return_profile=True)
    assert profile["outputs"]["histograms.hist"] == expected
    assert graph.current_state_id == before
    profile["outputs"]["histograms.hist"].clear()
    assert graph.current_state_id == before
    if values:
        with pytest.raises(RuntimeError, match="map"):
            graph.view().apply(AtomicSteps(steps + [{"id": "core.attach_node_attr", "params": {"source": "hist", "attr": "wrong"}}]), persist=True)
        assert graph.get_node_attr(0, "wrong") is None


@pytest.mark.parametrize("direction,aggregation,expected", [
    ("in", "sum", [4, 2, 0]), ("in", "mean", [2, 1, 0]),
    ("in", "min", [1, 1, None]), ("in", "max", [3, 1, None]), ("in", "mode", [1, 1, None]),
    ("out", "sum", [5, 0, 1]), ("out", "mean", [5/3, 0, 1]),
    ("out", "min", [1, None, 1]), ("out", "max", [2, None, 1]), ("out", "mode", [2, None, 1]),
    ("undirected", "sum", [8, 2, 1]), ("undirected", "mean", [2, 1, 1]),
    ("undirected", "min", [1, 1, 1]), ("undirected", "max", [3, 1, 1]), ("undirected", "mode", [2, 1, 1]),
])
def test_all_neighbor_reducers_use_selected_direction_parallel_edges_and_loops(direction, aggregation, expected):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": value} for value in [1, 2, 3]])
    graph.add_edges([(0, 1), (0, 1), (0, 0), (2, 0)])
    steps = [load_values(), {"id": "core.neighbor_agg", "params": {"source": "values", "agg": aggregation, "direction": direction, "target": "result"}},
             {"id": "core.attach_node_attr", "params": {"source": "result", "attr": "answer"}}]
    before = graph.current_state_id
    graph.view().apply(AtomicSteps([loop_step(steps, 2)]), persist=True)
    for node, value in enumerate(expected):
        actual = graph.get_node_attr(node, "answer")
        assert actual is None if value is None else actual == pytest.approx(value)
    assert graph.current_state_id == before + 1


@pytest.mark.parametrize("use_ir", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("direction,expected", [("in", [8, 4, 0]), ("out", [10, 0, 2]), ("undirected", [16, 4, 2])])
def test_public_neighbor_builder_preserves_direction_and_weight_operands(use_ir, atomic, direction, expected):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": value} for value in [1, 2, 3]])
    graph.add_edges([(0, 1), (0, 1), (0, 0), (2, 0)])
    builder = AlgorithmBuilder("weighted neighbors", use_ir=use_ir)
    values = builder.load_attr("score")
    weights = builder.init_nodes(default=2)
    result = builder.graph_ops.neighbor_agg(values, "sum", weights=weights, direction=direction)
    builder.attach_as("answer", result)
    graph.view().apply(builder.build(atomic=atomic, optimize=False), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == expected


def test_neighbor_weighted_sum_avoids_overflowing_individual_products():
    graph = Graph(directed=True)
    graph.add_nodes([{"score": value} for value in [1e308, -1e308, 0]])
    graph.add_edges([(0, 2), (1, 2)])
    steps = [load_values(), {"id": "core.init_nodes", "params": {"target": "weights", "value": 2}},
             {"id": "core.neighbor_agg", "params": {"source": "values", "weights": "weights", "agg": "sum", "target": "result"}},
             {"id": "core.attach_node_attr", "params": {"source": "result", "attr": "answer"}}]
    graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.get_node_attr(2, "answer") == 0


@pytest.mark.parametrize("step", [
    {"id": "core.reduce_nodes", "params": {"source": "n", "target": "r", "reducer": 3}},
    {"id": "core.neighbor_agg", "params": {"source": "n", "target": "r", "agg": 3}},
    {"id": "core.neighbor_agg", "params": {"source": "n", "target": "r", "direction": 3}},
    {"id": "core.neighbor_agg", "params": {"source": "n", "target": "r", "weights": 3}},
    {"id": "core.quantile", "params": {"source": "n", "target": "q", "q": 1.1}},
    {"id": "core.histogram", "params": {"source": "n", "target": "hist", "bins": 0}},
    {"id": "core.histogram", "params": {"source": "n", "target": "hist", "bins": -1}},
    {"id": "core.neighbor_agg", "params": {"source": "n", "target": "out", "direction": "typo"}},
    {"id": "core.neighbor_agg", "params": {"source": "n", "target": "out", "weights": "n", "agg": "mean"}},
])
def test_invalid_aggregation_controls_reject_before_prefix_writes(step):
    graph = Graph()
    graph.add_node()
    steps = [{"id": "core.init_nodes", "params": {"target": "n", "value": 1}},
             {"id": "core.attach_node_attr", "params": {"source": "n", "attr": "prefix"}}, step]
    with pytest.raises(RuntimeError):
        graph.view().apply(AtomicSteps(steps), persist=True)
    assert graph.current_state_id == 1
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("use_ir", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("method,params,expected", [
    ("std", {}, statistics.stdev([1, 2, 3, 4])),
    ("median", {}, 2.5), ("quantile", {"q": 0.25}, 1.75),
    ("mode_values", {}, 1), ("entropy", {}, 2),
    ("reduce_nodes", {"reducer": "sum"}, 10),
])
def test_public_statistics_builders_lower_in_both_builder_paths(method, params, expected, atomic, use_ir):
    graph = Graph()
    graph.add_nodes([{"score": value} for value in [1, 2, 3, 4]])
    builder = AlgorithmBuilder("public statistics", use_ir=use_ir)
    values = builder.load_attr("score")
    scalar = getattr(builder.core, method)(values, **params)
    builder.attach_as("answer", builder.core.broadcast_scalar(scalar, values))
    graph.view().apply(builder.build(atomic=atomic), persist=True)
    assert graph.get_node_attr(0, "answer") == pytest.approx(expected)


@pytest.mark.parametrize("use_ir", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
def test_histogram_is_observable_without_attachment_under_default_optimizer(use_ir, atomic):
    graph = Graph()
    graph.add_nodes([{"score": 1}, {"score": 2}])
    builder = AlgorithmBuilder("observable histogram", use_ir=use_ir)
    histogram = builder.core.histogram(builder.load_attr("score"), bins=2)
    with warnings.catch_warnings(record=True) as notices:
        handle = builder.build(atomic=atomic)
    assert not any("doesn't attach" in str(notice.message) for notice in notices)
    _, profile = graph.view().apply(handle, persist=True, return_profile=True)
    assert profile["outputs"][f"histograms.{histogram.name}"] == [1, 1]
    assert graph.current_state_id == 1


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("aggregation", ["sum", "mean", "min", "max", "mode"])
def test_output_scaling_is_not_moved_inside_neighbor_aggregation(optimize, aggregation):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": value, "scale": scale} for value, scale in [(1, 10), (2, 100), (3, 1000)]])
    graph.add_edges([(0, 1), (1, 2), (2, 0)])
    builder = AlgorithmBuilder("scale after aggregation")
    values = builder.load_attr("score")
    scales = builder.load_attr("scale")
    neighbors = builder.graph_ops.neighbor_agg(values, aggregation, direction="in")
    builder.attach_as("answer", neighbors * scales)
    graph.view().apply(builder.build(optimize=optimize), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == [30, 100, 2000]


@pytest.mark.parametrize("optimize", [False, True])
def test_weight_operand_survives_optimizer_rebuild(optimize):
    graph = Graph(directed=True)
    graph.add_nodes([{"score": value} for value in [1, 2, 3]])
    graph.add_edges([(0, 1), (1, 2), (2, 0)])
    builder = AlgorithmBuilder("weighted optimized")
    values = builder.load_attr("score")
    weights = builder.init_nodes(default=2)
    builder.attach_as("answer", builder.graph_ops.neighbor_agg(values, weights=weights, direction="out"))
    graph.view().apply(builder.build(optimize=optimize), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == [4, 6, 2]


@pytest.mark.parametrize("use_ir", [False, True])
@pytest.mark.parametrize("atomic", [False, True])
def test_nested_histogram_output_survives_loop_lowering_without_publication(use_ir, atomic):
    graph = Graph()
    graph.add_nodes([{"score": 1}, {"score": 2}])
    builder = AlgorithmBuilder("nested owned output", use_ir=use_ir)
    values = builder.load_attr("score")
    with builder.iterate(2):
        histogram = builder.core.histogram(values, bins=2)
    _, profile = graph.view().apply(builder.build(atomic=atomic), persist=True, return_profile=True)
    assert profile["outputs"][f"histograms.{histogram.name}"] == [1, 1]
    assert graph.current_state_id == 1
