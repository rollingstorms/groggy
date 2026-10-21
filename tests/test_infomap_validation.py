import pytest
from groggy import Graph, algorithms


def fixture():
    graph = Graph()
    nodes = [graph.add_node() for _ in range(8)]
    for start in (0, 4):
        for u in range(start, start + 4):
            for v in range(u + 1, start + 4):
                graph.add_edge(nodes[u], nodes[v])
    graph.add_edge(nodes[3], nodes[4])
    return graph, nodes


def run(graph, nodes, **kwargs):
    graph.induced_subgraph(nodes).apply(algorithms.community.infomap(seed=42, **kwargs))
    return [graph.get_node_attr(n, "community") for n in nodes]


def test_map_equation_detects_flow_modules_and_teleportation_changes_the_result():
    graph, nodes = fixture()
    labels = run(graph, nodes, teleportation=0.15, num_trials=5)
    assert len(set(labels[:4])) == len(set(labels[4:])) == 1
    assert labels[0] != labels[4]
    assert run(graph, nodes, teleportation=0.15, num_trials=5) == labels
    assert len(set(run(graph, nodes, teleportation=1, num_trials=5))) == 1
    assert len(graph.edge_ids) == 13


def test_failed_flow_solve_preserves_prior_labels():
    graph, nodes = fixture()
    before = run(graph, nodes)
    with pytest.raises(RuntimeError, match="did not converge"):
        run(graph, nodes, flow_max_iter=1)
    assert [graph.get_node_attr(n, "community") for n in nodes] == before


@pytest.mark.parametrize("name", ["num_trials", "max_iter", "flow_max_iter"])
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_positive_budgets_reject(name, value):
    graph, nodes = fixture()
    with pytest.raises(RuntimeError, match=name):
        run(graph, nodes, **{name: value})


@pytest.mark.parametrize("value", [-1.0, 1.1, float("nan"), float("inf")])
def test_invalid_teleportation_rejects(value):
    graph, nodes = fixture()
    with pytest.raises(RuntimeError, match="teleportation"):
        run(graph, nodes, teleportation=value)


def test_directed_dangling_and_edgeless_selections_are_supported():
    graph = Graph(directed=True)
    a, b = graph.add_node(), graph.add_node()
    graph.add_edge(a, b)
    assert run(graph, [a, b], teleportation=0.25) == [0, 0]
    graph = Graph()
    nodes = [graph.add_node() for _ in range(3)]
    # Recorded dangling jumps connect these states in the flow model.
    assert run(graph, nodes) == [0, 0, 0]


def test_negative_seed_rejects():
    graph, nodes = fixture()
    with pytest.raises(RuntimeError, match="seed"):
        graph.induced_subgraph(nodes).apply(algorithms.community.infomap(seed=-1))
