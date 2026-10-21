import pytest

from groggy import Graph, algorithms


def fixture():
    graph = Graph()
    nodes = [graph.add_node() for _ in range(50)]
    graph.remove_node(nodes[48])
    for c in range(16):
        a = c * 3
        for u, v in [(a, a + 1), (a + 1, a + 2), (a + 2, a), (a + 2, (a + 3) % 48)]:
            graph.add_edge(nodes[u], nodes[v])
    return graph, nodes[:48] + [nodes[49]]


def objective(graph, nodes, labels):
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    a = [[0] * n for _ in range(n)]
    degrees = [0] * n
    # The fixture's original edges, including the isolate in normalization.
    edges = [(c * 3 + u, c * 3 + v) for c in range(16) for u, v in [(0, 1), (1, 2), (2, 0)]]
    edges += [(c * 3 + 2, ((c + 1) * 3) % 48) for c in range(16)]
    for u, v in edges:
        i, j = index[u], index[v]
        a[i][j] += 1
        a[j][i] += 1
        degrees[i] += 1
        degrees[j] += 1
    twice_m = 2 * len(edges)
    return sum(a[i][j] - degrees[i] * degrees[j] / twice_m
               for i in range(n) for j in range(n) if labels[i] == labels[j]) / twice_m


def test_refined_levels_improve_the_original_objective_and_preserve_source():
    graph, nodes = fixture()
    view = graph.induced_subgraph(nodes)
    view.apply(algorithms.community.leiden(max_iter=100, max_phases=1, seed=42))
    single = [graph.get_node_attr(n, "community") for n in nodes]
    view.apply(algorithms.community.leiden(max_iter=100, max_phases=10, seed=42))
    multi = [graph.get_node_attr(n, "community") for n in nodes]
    assert objective(graph, nodes, multi) > objective(graph, nodes, single)
    assert multi[-1] not in multi[:-1]
    assert len(graph.edge_ids) == 64
    view.apply(algorithms.community.leiden(max_iter=100, max_phases=10, seed=42))
    assert [graph.get_node_attr(n, "community") for n in nodes] == multi


@pytest.mark.parametrize("theta", [0, -1, float("nan"), float("inf")])
def test_invalid_refinement_temperature_preserves_prior_labels(theta):
    graph, nodes = fixture()
    view = graph.induced_subgraph(nodes)
    view.apply(algorithms.community.leiden(seed=42))
    before = [graph.get_node_attr(n, "community") for n in nodes]
    with pytest.raises(Exception, match="theta"):
        view.apply(algorithms.community.leiden(seed=42, theta=theta))
    assert [graph.get_node_attr(n, "community") for n in nodes] == before


@pytest.mark.parametrize("theta", [1e-300, 0.01, 1e100])
def test_temperature_extremes_remain_finite_and_repeatable(theta):
    graph, nodes = fixture()
    view = graph.induced_subgraph(nodes)
    outputs = []
    for _ in range(2):
        view.apply(algorithms.community.leiden(1.0, 100, 10, 42, "legacy", theta=theta))
        outputs.append([graph.get_node_attr(n, "legacy") for n in nodes])
    assert outputs[0] == outputs[1]
    assert all(value is not None for value in outputs[0])
