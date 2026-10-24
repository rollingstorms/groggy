"""Community step identity, numerical and atomic publication contracts."""
import json

import pytest
from groggy import Graph
from groggy.algorithms.base import AlgorithmHandle


class Steps(AlgorithmHandle):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        from groggy import AttrValue
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({"atomic": True, "steps": self.steps}))}}


def attach(source="labels", attr="answer"):
    return {"id": "core.attach_node_attr", "params": {"source": source, "attr": attr}}


def seed(strategy="singleton", **params):
    return {"id": "core.community_seed", "params": {"strategy": strategy, "target": "labels", **params}}


def fixture(directed=False):
    graph = Graph(directed=directed)
    graph.add_nodes([{} for _ in range(4)])
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    return graph


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("strategy,params", [("singleton", {}), ("degree", {}), ("random", {"k": 3, "seed": 43})])
@pytest.mark.parametrize("persist", [False, True])
def test_seeding_accepts_once_or_returns_owned_columns(directed, strategy, params, persist):
    graph = fixture(directed)
    before = graph.current_state_id
    _, profile = graph.view().apply(Steps([seed(strategy, **params), attach()]), persist=persist, return_profile=True)
    values = [graph.get_node_attr(n, "answer") for n in range(4)] if persist else [value for _, value in profile["outputs"]["node_attributes.answer"]]
    assert all(isinstance(value, int) for value in values)
    if strategy == "singleton":
        assert values == [0, 1, 2, 3]
    if strategy == "random":
        assert all(0 <= value < 3 for value in values)
    assert graph.current_state_id == before + int(persist)


@pytest.mark.parametrize("params", [
    {"strategy": "random"}, {"strategy": "random", "k": 0}, {"strategy": "random", "k": -1},
    {"strategy": "random", "k": "3"}, {"strategy": "random", "k": True},
    {"strategy": "random", "k": 3, "seed": -1}, {"strategy": "random", "k": 3, "seed": "43"},
    {"strategy": "typo"}, {"strategy": "singleton", "k": 0},
])
def test_seed_controls_are_strict_even_on_empty_graphs(params):
    graph = Graph()
    before = graph.current_state_id
    with pytest.raises(RuntimeError):
        graph.view().apply(Steps([{"id": "core.community_seed", "params": {"target": "labels", **params}}]), persist=True)
    assert graph.current_state_id == before


@pytest.mark.parametrize("directed,expected", [(False, [0, 0, 1, 2]), (True, [0, 1, 2, 3])])
def test_label_propagation_is_synchronous_with_explicit_self_votes(directed, expected):
    graph = fixture(directed)
    before = graph.current_state_id
    propagate = {"id": "core.label_propagate_step", "params": {"labels": "labels", "target": "result"}}
    graph.view().apply(Steps([seed(), propagate, attach("result")]), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(4)] == expected
    assert graph.current_state_id == before + 1


def test_parallel_edges_and_self_loop_vote_as_stored_incidence():
    graph = Graph(directed=True)
    graph.add_nodes([{"label": 10}, {"label": 2}, {"label": 3}])
    graph.add_edge(0, 1)
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(2, 2)
    steps = [{"id": "core.load_node_attr", "params": {"attr": "label", "target": "labels"}},
             {"id": "core.label_propagate_step", "params": {"labels": "labels", "target": "result"}}, attach("result")]
    graph.view().apply(Steps(steps), persist=True)
    assert [graph.get_node_attr(n, "answer") for n in range(3)] == [2, 2, 3]


def objective(edges, labels, directed):
    if not edges:
        return 0.0
    m = len(edges)
    out = [0] * len(labels)
    incoming = [0] * len(labels)
    internal = sum(labels[u] == labels[v] for u, v in edges)
    for u, v in edges:
        out[u] += 1
        if directed:
            incoming[v] += 1
        else:
            out[v] += 1
    penalty = 0.0
    for label in set(labels):
        members = [node for node, value in enumerate(labels) if value == label]
        volume = sum(out[node] for node in members)
        penalty += volume * sum(incoming[node] for node in members) / m**2 if directed else (volume / (2 * m))**2
    return internal / m - penalty


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("edges,labels", [
    ([(0, 1)], [0, 1, 2, 3]),
    ([(0, 0), (0, 1), (0, 1), (1, 2), (2, 3)], [0, 0, 1, 1]),
    ([(0, 1), (1, 2), (2, 3), (3, 0)], [0, 0, 0, 0]),
    ([], [0, 1, 2, 3]),
])
def test_modularity_gain_matches_full_objective_move_differences(directed, edges, labels):
    graph = Graph(directed=directed)
    graph.add_nodes([{"label": label} for label in labels])
    for u, v in edges:
        graph.add_edge(u, v)
    steps = [{"id": "core.load_node_attr", "params": {"attr": "label", "target": "labels"}},
             {"id": "core.modularity_gain", "params": {"partition": "labels", "target": "gain"}}, attach("gain")]
    before = graph.current_state_id
    graph.view().apply(Steps(steps), persist=True)
    baseline = objective(edges, labels, directed)
    for node in range(4):
        candidates = {labels[v] for u, v in edges if u == node and v != node} | {labels[u] for u, v in edges if v == node and u != node}
        expected = 0.0
        for candidate in candidates:
            moved = labels.copy()
            moved[node] = candidate
            expected = max(expected, objective(edges, moved, directed) - baseline)
        assert graph.get_node_attr(node, "answer") == pytest.approx(expected, abs=1e-12)
    assert graph.current_state_id == before + 1


@pytest.mark.parametrize("bad_label", ["one", None, 1.5, True])
def test_modularity_rejects_noninteger_labels_and_discards_prefix_properties(bad_label):
    graph = Graph()
    graph.add_nodes([{"label": bad_label}, {"label": 1}])
    graph.add_edge(0, 1)
    before = graph.current_state_id
    steps = [seed(), attach(attr="prefix"),
             {"id": "core.load_node_attr", "params": {"attr": "label", "target": "partition"}},
             {"id": "core.modularity_gain", "params": {"partition": "partition", "target": "gain"}}, attach("gain")]
    with pytest.raises(RuntimeError, match="labels must be integers"):
        graph.view().apply(Steps(steps), persist=True)
    assert graph.current_state_id == before
    assert graph.get_node_attr(0, "prefix") is None


def test_community_helpers_share_atomic_loop_and_staged_property_reads():
    graph = fixture()
    before = graph.current_state_id
    body = [seed(), attach(attr="staged"),
            {"id": "core.load_node_attr", "params": {"attr": "staged", "target": "loaded"}},
            {"id": "core.label_propagate_step", "params": {"labels": "loaded", "target": "result"}}, attach("result")]
    graph.view().apply(Steps([{"id": "iter.loop", "params": {"iterations": 2, "body": body}}]), persist=True)
    assert graph.current_state_id == before + 1
    assert [graph.get_node_attr(n, "answer") for n in range(4)] == [0, 0, 1, 2]
