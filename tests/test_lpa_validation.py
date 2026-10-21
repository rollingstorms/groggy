import pytest
from groggy import Graph, algorithms


def run(graph, nodes, **kwargs):
    graph.induced_subgraph(nodes).apply(algorithms.community.lpa(**kwargs))
    return [graph.get_node_attr(n, "community") for n in nodes]


def test_equal_seed_values_are_one_label_and_values_do_not_drift():
    graph = Graph(directed=True)
    nodes = [graph.add_node() for _ in range(3)]
    for node, value in zip(nodes, ["a", "b", "a"]):
        graph.set_node_attr(node, "initial", value)
    graph.add_edge(nodes[1], nodes[0])
    graph.add_edge(nodes[1], nodes[2])
    assert run(graph, nodes, seed_attr="initial", tolerance=0.0) == ["a"] * 3


def test_direction_parallel_votes_loops_and_isolates():
    for directed in (False, True):
        graph = Graph(directed=directed)
        a, b, c = nodes = [graph.add_node() for _ in range(3)]
        graph.add_edge(b, a)
        graph.add_edge(b, a)
        graph.add_edge(b, b)
        values = run(graph, nodes, tolerance=0.0)
        assert values[0] == values[1]
        assert values[2] != values[0]
        assert len(graph.edge_ids) == 3


def test_seed_is_honored_and_original_positional_api_still_works():
    graph = Graph()
    nodes = [graph.add_node() for _ in range(8)]
    for u in range(8):
        graph.add_edge(nodes[u], nodes[(u + 1) % 8])
    first = run(graph, nodes, seed=42, tolerance=0.0)
    assert run(graph, nodes, seed=42, tolerance=0.0) == first
    graph.induced_subgraph(nodes).apply(algorithms.community.lpa(10, 42, "other"))
    assert all(graph.get_node_attr(n, "other") is not None for n in nodes)
    trials = {tuple(run(graph, nodes, max_iter=1, seed=seed)) for seed in range(8)}
    assert len(trials) > 1  # The seed must affect execution, not just be accepted.


@pytest.mark.parametrize("kwargs", [
    {"max_iter": 0}, {"max_iter": -1}, {"seed": -1},
    {"tolerance": -0.1}, {"tolerance": 1.1},
    {"tolerance": float("nan")}, {"tolerance": float("inf")},
])
def test_invalid_controls_reject(kwargs):
    graph = Graph()
    node = graph.add_node()
    with pytest.raises(RuntimeError):
        run(graph, [node], **kwargs)
