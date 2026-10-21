import pytest
from groggy import Graph, algorithms
from groggy.pipeline import apply


def fixture(directed=True):
    g = Graph(directed=directed)
    nodes = [g.add_node() for _ in range(4)]
    for u, v, w in [(0, 1, 2), (0, 1, 8), (1, 2, 1), (1, 1, 1)]:
        edge = g.add_edge(nodes[u], nodes[v])
        g.set_edge_attr(edge, "w", w)
    return g, nodes


@pytest.mark.parametrize("harmonic,incoming,normalized,expected", [
    (True, False, False, [5 / 6, 1, 0, 0]),
    (True, True, False, [0, 1 / 2, 4 / 3, 0]),
    (True, False, True, [5 / 18, 1 / 3, 0, 0]),
    (False, False, False, [2 / 5, 1, 0, 0]),
    (False, False, True, [4 / 15, 1 / 3, 0, 0]),
    (False, True, True, [0, 1 / 6, 1 / 3, 0]),
])
def test_direction_definitions_and_disconnected_normalization(harmonic, incoming, normalized, expected):
    g, nodes = fixture()
    _, profile = apply(g.induced_subgraph(nodes), algorithms.centrality.closeness(
        "w", "c", harmonic=harmonic,
        direction="incoming" if incoming else "outgoing", normalized=normalized,
    ), persist=False, return_profile=True)
    rows = profile["outputs"]["centrality.closeness.scores"]
    assert [pair[0] for pair in rows] == nodes
    assert [pair[1] for pair in rows] == pytest.approx(expected, abs=1e-12)
    assert all(g.get_node_attr(n, "c") is None for n in nodes)


def test_unweighted_undirected_defaults_and_old_positional_arguments():
    g, nodes = fixture(False)
    g.induced_subgraph(nodes).apply(algorithms.centrality.closeness(None, "c"))
    assert [g.get_node_attr(n, "c") for n in nodes] == pytest.approx([1.5, 2, 1.5, 0])


@pytest.mark.parametrize("value", [0, -1, True, None, "bad", float("nan"), float("inf")])
def test_invalid_lengths_preserve_previous_output(value):
    g = Graph()
    a, b = g.add_node(), g.add_node()
    e = g.add_edge(a, b)
    g.set_edge_attr(e, "w", value)
    g.set_node_attr(a, "c", 7)
    with pytest.raises(RuntimeError):
        g.induced_subgraph([a, b]).apply(algorithms.centrality.closeness("w", "c"))
    assert g.get_node_attr(a, "c") == 7


def test_f64_results_remain_available_when_f32_properties_would_overflow():
    g = Graph()
    a, b = g.add_node(), g.add_node()
    e = g.add_edge(a, b)
    g.set_edge_attr(e, "w", 1e-40)
    sub = g.induced_subgraph([a, b])
    algo = algorithms.centrality.closeness("w", "c")
    _, profile = apply(sub, algo, persist=False, return_profile=True)
    assert profile["outputs"]["centrality.closeness.scores"][0][1] > 1e39
    with pytest.raises(RuntimeError, match="f32"):
        sub.apply(algo)
    assert g.get_node_attr(a, "c") is None


def test_invalid_direction_rejects():
    g, nodes = fixture()
    with pytest.raises(RuntimeError, match="direction"):
        g.induced_subgraph(nodes).apply(algorithms.centrality.closeness(direction="sideways"))
