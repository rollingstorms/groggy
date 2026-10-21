import pytest

from groggy import Graph, algorithms


def ring():
    graph = Graph()
    nodes = [graph.add_node() for _ in range(49)]
    edges = []
    for c in range(16):
        a = c * 3
        for u, v in [(a, a + 1), (a + 1, a + 2), (a + 2, a), (a + 2, (a + 3) % 48)]:
            graph.add_edge(nodes[u], nodes[v])
            edges.append((u, v))
    return graph, nodes, edges


def modularity(labels, edges):
    n = len(labels)
    a = [[0] * n for _ in range(n)]
    degree = [0] * n
    for u, v in edges:
        a[u][v] += 1
        a[v][u] += 1
        degree[u] += 1
        degree[v] += 1
    twice_m = 2 * len(edges)
    return sum(
        a[i][j] - degree[i] * degree[j] / twice_m
        for i in range(n) for j in range(n) if labels[i] == labels[j]
    ) / twice_m


def test_multilevel_improves_original_objective_and_preserves_source_topology():
    graph, nodes, edges = ring()
    view = graph.induced_subgraph(nodes)
    view.apply(algorithms.community.louvain(max_iter=100, max_phases=1))
    local = [graph.get_node_attr(n, "community") for n in nodes]
    view.apply(algorithms.community.louvain(max_iter=100, max_phases=10))
    multi = [graph.get_node_attr(n, "community") for n in nodes]
    assert modularity(multi, edges) > modularity(local, edges)
    assert multi[-1] not in multi[:-1]
    assert len(graph.edge_ids) == len(edges)
    view.apply(algorithms.community.louvain(max_iter=100))
    assert [graph.get_node_attr(n, "community") for n in nodes] == multi


@pytest.mark.parametrize("phases", [0, -1])
def test_phase_budget_validation_preserves_prior_result(phases):
    graph, nodes, _ = ring()
    view = graph.induced_subgraph(nodes)
    view.apply(algorithms.community.louvain())
    before = [graph.get_node_attr(n, "community") for n in nodes]
    with pytest.raises(Exception, match="max_phases"):
        view.apply(algorithms.community.louvain(max_phases=phases))
    assert [graph.get_node_attr(n, "community") for n in nodes] == before


def test_existing_positional_arguments_keep_their_meaning():
    graph, nodes, _ = ring()
    view = graph.induced_subgraph(nodes)
    view.apply(algorithms.community.louvain(1.0, 100, "legacy", max_phases=1))
    assert all(graph.get_node_attr(n, "legacy") is not None for n in nodes)
