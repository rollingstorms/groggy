"""Selected checked path primitives compose with staged property writes."""
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


def scalar(name, value):
    return {"id": "core.init_scalar", "params": {"target": name, "value": value}}


def attach(source="distances", attr="answer"):
    return {"id": "core.attach_node_attr", "params": {"source": source, "attr": attr}}


def expose_scalar(source):
    return [{"id": "core.init_nodes", "params": {"target": "reference", "value": 0}},
            {"id": "core.broadcast_scalar", "params": {"scalar": source, "reference": "reference", "target": "result"}}, attach("result")]


def fixture(directed=True):
    graph = gr.Graph(directed=directed)
    graph.add_nodes([{} for _ in range(4)])
    for u, v, weight in [(0, 1, 10.0), (0, 1, 2.0), (1, 2, 1.0), (0, 2, 8.0), (2, 3, 1.0)]:
        graph.add_edge(u, v, weight=weight)
    return graph


def shortest(weight=True):
    return {"id": "core.shortest_path_map", "params": {"source": "source", "output": "distances", **({"weight_attr": "weight"} if weight else {})}}


def kpaths(k=10, **params):
    return {"id": "core.k_shortest_paths", "params": {"source": "source", "target": "target", "k": k, "weight_attr": "weight", "output": "paths", **params}}


def walk(**params):
    return {"id": "core.random_walk", "params": {"start_nodes": "source", "length": 3, "seed": 19, "output": "walks", **params}}


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("persist", [False, True])
def test_shortest_distance_maps_have_selected_identity_and_one_publication(directed, weighted, persist):
    graph = fixture(directed)
    before = graph.current_state_id
    _, profile = graph.view().apply(Steps([scalar("source", 0), shortest(weighted), attach()]), persist=persist, return_profile=True)
    expected = [0, 2, 3, 4] if weighted else [0, 1, 1, 2]
    actual = [graph.get_node_attr(n, "answer") for n in range(4)] if persist else [v for _, v in profile["outputs"]["node_attributes.answer"]]
    assert actual == expected
    assert graph.current_state_id == before + int(persist)


@pytest.mark.parametrize("kernel", ["distances", "paths", "walks"])
def test_path_weights_read_staged_edge_columns_and_publish_once(kernel):
    graph = fixture()
    before = graph.current_state_id
    prefix = [scalar("source", 0), scalar("target", 3),
              {"id": "core.init_edges", "params": {"target": "weights", "value": 4.0}},
              {"id": "core.attach_edge_attr", "params": {"source": "weights", "attr": "weight"}}]
    operation = shortest() if kernel == "distances" else kpaths(1) if kernel == "paths" else walk(weight_attr="weight")
    tail = [attach()] if kernel == "distances" else expose_scalar(kernel)
    graph.view().apply(Steps(prefix + [operation] + tail), persist=True)
    value = graph.get_node_attr(3, "answer")
    if kernel == "distances":
        assert value == 8.0
    elif kernel == "paths":
        assert json.loads(value) == [[[0, 2, 3], 8.0]]
    else:
        nodes = json.loads(value)[0]
        assert nodes[0] == 0
        assert all((u, v) in {(0, 1), (1, 2), (0, 2), (2, 3)} for u, v in zip(nodes, nodes[1:]))
    assert graph.current_state_id == before + 1


@pytest.mark.parametrize("kernel", ["distances", "paths", "walks"])
@pytest.mark.parametrize("invalid", [-1, None, "bad", True, float("inf"), float("nan")])
def test_invalid_selected_weights_discard_prefix_assignments(kernel, invalid):
    graph = fixture()
    graph.set_edge_attr(0, "weight", invalid)
    before = graph.current_state_id
    prefix = [scalar("source", 0), scalar("target", 3), {"id": "core.init_nodes", "params": {"target": "prefix", "value": 9}}, attach("prefix", "prefix")]
    operation = shortest() if kernel == "distances" else kpaths() if kernel == "paths" else walk(weight_attr="weight")
    with pytest.raises(RuntimeError, match="finite non-negative"):
        graph.view().apply(Steps(prefix + [operation]), persist=True)
    assert graph.current_state_id == before
    assert graph.get_node_attr(0, "prefix") is None


@pytest.mark.parametrize("source", [-1, 999, "-1", True])
@pytest.mark.parametrize("kernel", ["distances", "paths", "walks"])
def test_invalid_sources_reject_even_for_zero_sized_requests(source, kernel):
    graph = fixture()
    operation = shortest() if kernel == "distances" else kpaths(0) if kernel == "paths" else walk(length=0)
    with pytest.raises(RuntimeError):
        graph.view().apply(Steps([scalar("source", source), scalar("target", 3), operation]), persist=True)


def test_yen_orders_tiny_costs_without_quantization_and_keeps_only_simple_node_paths():
    graph = gr.Graph(directed=True)
    graph.add_nodes([{} for _ in range(4)])
    for u, v, weight in [(0, 1, 1e-9), (1, 3, 0.0), (0, 2, 2e-9), (2, 3, 0.0), (1, 0, 0.0), (2, 1, 1e-9), (0, 1, 5e-9)]:
        graph.add_edge(u, v, weight=weight)
    steps = [scalar("source", 0), scalar("target", 3), kpaths(100)] + expose_scalar("paths")
    graph.view().apply(Steps(steps), persist=True)
    paths = json.loads(graph.get_node_attr(0, "answer"))
    assert [path for path, _ in paths] == [[0, 1, 3], [0, 2, 3], [0, 2, 1, 3]]
    assert [cost for _, cost in paths] == pytest.approx([1e-9, 2e-9, 3e-9])
    assert all(len(path) == len(set(path)) for path, _ in paths)


@pytest.mark.parametrize("k,source,target,expected", [(0, 0, 3, []), (5, 1, 1, [[[1], 0.0]]), (3, 3, 0, [])])
def test_empty_trivial_and_unreachable_path_results(k, source, target, expected):
    graph = fixture()
    graph.view().apply(Steps([scalar("source", source), scalar("target", target), kpaths(k)] + expose_scalar("paths")), persist=True)
    assert json.loads(graph.get_node_attr(0, "answer")) == expected


@pytest.mark.parametrize("restart,length,expected", [(0.0, 0, [[3]]), (0.0, 100, [[3]]), (1.0, 3, [[3, 3, 3, 3]])])
def test_walk_dead_ends_restart_and_zero_length(restart, length, expected):
    graph = fixture()
    graph.view().apply(Steps([scalar("source", 3), walk(length=length, restart_prob=restart)] + expose_scalar("walks")), persist=True)
    assert json.loads(graph.get_node_attr(0, "answer")) == expected


def test_walks_keep_large_weight_normalization_and_seed_repeatability():
    graph = gr.Graph(directed=True)
    graph.add_nodes([{} for _ in range(3)])
    for u, v, weight in [(0, 1, 0.0), (0, 2, 1e308), (0, 2, 1e308), (2, 2, 1.0)]:
        graph.add_edge(u, v, weight=weight)
    steps = [scalar("source", 0), walk(length=6, weight_attr="weight")] + expose_scalar("walks")
    for _ in range(3):
        graph.view().apply(Steps(steps), persist=True)
        assert json.loads(graph.get_node_attr(0, "answer")) == [[0, 2, 2, 2, 2, 2, 2]]


@pytest.mark.parametrize("step", [kpaths(-1), kpaths(weight_attr=3), walk(length=-1), walk(seed=-1), walk(seed="1"), walk(restart_prob="bad"), walk(restart_prob=1.1), walk(weight_attr=3)])
def test_path_controls_are_strict(step):
    graph = fixture()
    with pytest.raises(RuntimeError):
        graph.view().apply(Steps([step]), persist=True)


def test_selected_parent_edges_and_undirected_reverse_are_respected():
    graph = fixture(False)
    parent = graph.subgraph_from_edges([0, 2])
    parent.apply(Steps([scalar("source", 2), shortest(), attach()]), persist=True)
    assert [graph.get_node_attr(node, "answer") for node in range(4)] == [11.0, 1.0, 0.0, None]


def test_walks_sample_parallel_edge_incidence_instead_of_unique_neighbors():
    graph = gr.Graph(directed=True)
    graph.add_nodes([{} for _ in range(3)])
    for u, v in [(0, 1), (0, 1), (0, 2), (1, 0), (2, 0)]:
        graph.add_edge(u, v)
    graph.view().apply(Steps([scalar("source", 0), walk(length=6000)] + expose_scalar("walks")), persist=True)
    nodes = json.loads(graph.get_node_attr(0, "answer"))[0]
    assert 1850 < nodes.count(1) < 2150
    assert nodes.count(1) + nodes.count(2) == 3000


def test_node_map_walk_starts_are_sorted_and_eligible_independently_of_values():
    graph = fixture()
    starts = {"id": "core.init_nodes", "params": {"target": "source", "value": 0}}
    graph.view().apply(Steps([starts, walk(length=0)] + expose_scalar("walks")), persist=True)
    assert json.loads(graph.get_node_attr(0, "answer")) == [[0], [1], [2], [3]]


@pytest.mark.parametrize("operation", [shortest(), kpaths(1)])
def test_unrepresentable_explored_path_costs_reject_instead_of_returning_infinity(operation):
    graph = gr.Graph(directed=True)
    graph.add_nodes([{}, {}, {}])
    graph.add_edge(0, 1, weight=1e308)
    graph.add_edge(1, 2, weight=1e308)
    before = graph.current_state_id
    with pytest.raises(RuntimeError, match="cost overflow"):
        graph.view().apply(Steps([scalar("source", 0), scalar("target", 2), operation]), persist=True)
    assert graph.current_state_id == before


def test_absent_weight_cells_default_to_one_while_explicit_null_remains_invalid():
    graph = gr.Graph(directed=True)
    graph.add_nodes([{}, {}, {}])
    graph.add_edge(0, 1)
    graph.add_edge(1, 2, weight=4.0)
    steps = [scalar("source", 0), shortest(), attach()]
    graph.view().apply(Steps(steps), persist=True)
    assert [graph.get_node_attr(node, "answer") for node in range(3)] == [0.0, 1.0, 5.0]
    graph.set_edge_attr(0, "weight", None)
    with pytest.raises(RuntimeError, match="finite non-negative"):
        graph.view().apply(Steps(steps), persist=True)
