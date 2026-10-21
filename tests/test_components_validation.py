import pytest
from groggy import Graph, algorithms


def run(graph, nodes, mode):
    graph.induced_subgraph(nodes).apply(algorithms.community.connected_components(mode))
    return [graph.get_node_attr(n, "component") for n in nodes]


@pytest.mark.parametrize("mode", ["undirected", "weak", "strong"])
@pytest.mark.parametrize("directed", [False, True])
def test_reversed_chain_respects_component_mode_and_graph_direction(mode, directed):
    graph = Graph(directed=directed)
    a, b, c, d = nodes = [graph.add_node() for _ in range(4)]
    graph.add_edge(c, b)
    graph.add_edge(b, a)
    graph.add_edge(b, a)
    graph.add_edge(c, c)
    labels = run(graph, nodes, mode)
    assert labels[3] not in labels[:3]
    if directed and mode == "strong":
        assert len(set(labels)) == 4
    else:
        assert len(set(labels[:3])) == 1
    assert run(graph, nodes, mode) == labels
    assert len(graph.edge_ids) == 4


def test_strong_mode_distinguishes_cycles_and_one_way_links():
    graph = Graph(directed=True)
    nodes = [graph.add_node() for _ in range(5)]
    for u, v in [(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)]:
        graph.add_edge(nodes[u], nodes[v])
    c = run(graph, nodes, "strong")
    assert c[0] == c[1]
    assert c[2] == c[3]
    assert len({c[0], c[2], c[4]}) == 3


def test_invalid_mode_rejects():
    graph = Graph()
    node = graph.add_node()
    with pytest.raises(RuntimeError, match="mode"):
        run(graph, [node], "invalid")
