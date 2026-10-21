import pytest

from groggy import Graph, algorithms


def run(graph, nodes, levels=None, weighted=False, threshold=None):
    graph.induced_subgraph(nodes).apply(
        algorithms.community.girvan_newman(
            num_levels=levels, weight_attr="w" if weighted else None,
            modularity_threshold=threshold, output_attr="c",
        )
    )
    return [graph.get_node_attr(n, "c") for n in nodes]


def fixture(reverse=False, bridges=1):
    graph = Graph()
    nodes = [graph.add_node() for _ in range(7)]
    for u, v in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]:
        if reverse:
            u, v = v, u
        graph.add_edge(nodes[u], nodes[v])
    for i in range(bridges):
        u, v = (3, 2) if reverse else (2, 3)
        graph.add_edge(nodes[u], nodes[v], w=1 if i == 0 else 100)
    return graph, nodes


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_final_bridge_removal_is_scored_and_topology_is_preserved(reverse, weighted):
    graph, nodes = fixture(reverse)
    c = run(graph, nodes, levels=1, weighted=weighted)
    assert c[0] == c[1] == c[2]
    assert c[3] == c[4] == c[5]
    assert len({c[0], c[3], c[6]}) == 3
    assert len(graph.edge_ids) == 7


@pytest.mark.parametrize("weighted", [False, True])
def test_parallel_bridges_are_removed_individually(weighted):
    graph, nodes = fixture(True, 2)
    c = run(graph, nodes, levels=1, weighted=weighted)
    assert c[0] == c[3]
    c = run(graph, nodes, levels=2, weighted=weighted)
    assert c[0] != c[3]


@pytest.mark.parametrize("bad", [0, -1, None, True, "bad", float("nan"), float("inf")])
def test_invalid_lengths_reject_before_replacing_results(bad):
    graph, nodes = fixture()
    before = run(graph, nodes, levels=1)
    graph.set_edge_attr(6, "w", bad)
    with pytest.raises(Exception, match="weight"):
        run(graph, nodes, levels=0, weighted=True)
    assert [graph.get_node_attr(n, "c") for n in nodes] == before


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
def test_invalid_modularity_threshold_rejects(threshold):
    graph, nodes = fixture()
    with pytest.raises(Exception, match="threshold"):
        run(graph, nodes, levels=0, threshold=threshold)


def test_negative_level_and_directed_input_reject_explicitly():
    graph, nodes = fixture()
    with pytest.raises(Exception, match="num_levels"):
        run(graph, nodes, levels=-1)
    graph = Graph(directed=True)
    node = graph.add_node()
    with pytest.raises(Exception, match="undirected"):
        run(graph, [node])


def test_zero_budget_keeps_initial_components_and_isolates():
    graph, nodes = fixture(bridges=0)
    graph.add_edge(nodes[6], nodes[6])
    c = run(graph, nodes, levels=0)
    assert c[0] == c[1] == c[2]
    assert c[3] == c[4] == c[5]
    assert len({c[0], c[3], c[6]}) == 3
