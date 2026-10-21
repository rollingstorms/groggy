import pytest

from groggy import Graph, algorithms


def algorithm(name, **kwargs):
    return getattr(algorithms.community, name)(output_attr="c", **kwargs)


@pytest.mark.parametrize("name", ["louvain", "leiden"])
@pytest.mark.parametrize("resolution", [0.5, 3.0])
def test_resolution_changes_partition_in_both_local_optimizers(name, resolution):
    graph = Graph()
    a, b = graph.add_node(), graph.add_node()
    graph.add_edge(b, a)
    graph.induced_subgraph([a, b]).apply(algorithm(name, resolution=resolution))
    assert (graph.get_node_attr(a, "c") == graph.get_node_attr(b, "c")) == (resolution == 0.5)


@pytest.mark.parametrize("name", ["louvain", "leiden"])
@pytest.mark.parametrize("resolution", [0, -1, float("nan"), float("inf")])
def test_invalid_resolution_rejects_without_replacing_scores(name, resolution):
    graph = Graph()
    a, b = graph.add_node(), graph.add_node()
    graph.add_edge(a, b)
    view = graph.induced_subgraph([a, b])
    view.apply(algorithm(name))
    before = [graph.get_node_attr(n, "c") for n in (a, b)]
    with pytest.raises(Exception, match="resolution"):
        view.apply(algorithm(name, resolution=resolution))
    assert [graph.get_node_attr(n, "c") for n in (a, b)] == before


@pytest.mark.parametrize("name", ["louvain", "leiden"])
@pytest.mark.parametrize("max_iter", [0, -1])
def test_invalid_iteration_budget_rejects(name, max_iter):
    graph = Graph()
    node = graph.add_node()
    with pytest.raises(Exception, match="max_iter"):
        graph.induced_subgraph([node]).apply(algorithm(name, max_iter=max_iter))


@pytest.mark.parametrize("name", ["louvain", "leiden"])
def test_edgeless_nodes_get_distinct_labels_and_directed_input_rejects(name):
    graph = Graph()
    nodes = [graph.add_node() for _ in range(3)]
    graph.induced_subgraph(nodes).apply(algorithm(name))
    assert len({graph.get_node_attr(n, "c") for n in nodes}) == 3
    graph = Graph(directed=True)
    node = graph.add_node()
    with pytest.raises(Exception, match="undirected"):
        graph.induced_subgraph([node]).apply(algorithm(name))


def test_leiden_seed_repeats_labels_and_negative_seed_rejects():
    graph = Graph()
    nodes = [graph.add_node() for _ in range(8)]
    for i in range(8):
        graph.add_edge(nodes[i], nodes[(i + 1) % 8])
        graph.add_edge(nodes[i], nodes[(i + 3) % 8])
    view = graph.induced_subgraph(nodes)
    labels = []
    for _ in range(4):
        view.apply(algorithm("leiden", seed=42))
        labels.append([graph.get_node_attr(n, "c") for n in nodes])
    assert all(value == labels[0] for value in labels)
    with pytest.raises(Exception, match="seed"):
        view.apply(algorithm("leiden", seed=-1))
