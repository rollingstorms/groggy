import pytest
from groggy import Graph, algorithms
from groggy.pipeline import apply


def run(graph, nodes, **kwargs):
    return apply(graph.induced_subgraph(nodes), algorithms.pathfinding.dijkstra(
        "source", "w", "out"
    ), persist=False, return_profile=True, **kwargs)


def test_nonpersistent_f64_distances_include_zero_edges_and_reverse_parallel_minima():
    g = Graph()
    a, b, c, isolated = nodes = [g.add_node() for _ in range(4)]
    g.set_node_attr(c, "source", False)
    for u, v, weight in [(a, b, 8), (a, b, 2), (b, c, 0)]:
        e = g.add_edge(u, v)
        g.set_edge_attr(e, "w", weight)
    result, profile = run(g, nodes)
    assert profile["outputs"]["pathfinding.dijkstra.distances"] == [(a, 2.0), (b, 0.0), (c, 0.0)]
    assert profile["stats"]["dijkstra.count.reached_nodes"] == 3
    assert "out" not in result.nodes.attribute_names()
    assert all(g.get_node_attr(n, "out") is None for n in nodes)
    # Carry the returned selection into a real cache-hit run.
    _, again = apply(result, algorithms.pathfinding.dijkstra("source", "w", "out"),
                     persist=False, return_profile=True)
    assert "dijkstra.csr_cache_hit" in again["call_counters"]


@pytest.mark.parametrize("count", [0, 2])
def test_missing_and_ambiguous_sources_reject_without_writes(count):
    g = Graph()
    nodes = [g.add_node() for _ in range(3)]
    for n in nodes[:count]:
        g.set_node_attr(n, "source", True)
    g.set_node_attr(nodes[0], "out", 7)
    with pytest.raises(RuntimeError, match="exactly one"):
        g.induced_subgraph(nodes).apply(algorithms.pathfinding.dijkstra("source", "w", "out"))
    assert g.get_node_attr(nodes[0], "out") == 7


@pytest.mark.parametrize("marker", [False, 0, None, "marker"])
def test_marker_value_is_ignored(marker):
    g = Graph()
    node = g.add_node()
    g.set_node_attr(node, "source", marker)
    _, profile = run(g, [node])
    assert profile["outputs"]["pathfinding.dijkstra.distances"] == [(node, 0.0)]


@pytest.mark.parametrize("bad", [-1, True, None, "bad", float("nan"), float("inf")])
def test_invalid_unreachable_weights_are_not_silently_ignored(bad):
    g = Graph(directed=True)
    a, b, c = nodes = [g.add_node() for _ in range(3)]
    g.set_node_attr(a, "source", True)
    e = g.add_edge(b, c)
    g.set_edge_attr(e, "w", bad)
    with pytest.raises(RuntimeError):
        run(g, nodes)


def test_f64_output_survives_when_persistent_f32_distances_would_overflow():
    g = Graph(directed=True)
    a, b, c = nodes = [g.add_node() for _ in range(3)]
    g.set_node_attr(a, "source", True)
    for u, v in [(a, b), (b, c)]:
        e = g.add_edge(u, v)
        g.set_edge_attr(e, "w", 3e38)
    _, profile = run(g, nodes)
    rows = profile["outputs"]["pathfinding.dijkstra.distances"]
    assert rows[-1][0] == c and rows[-1][1] > 5.9e38
    with pytest.raises(RuntimeError, match="f32"):
        g.induced_subgraph(nodes).apply(algorithms.pathfinding.dijkstra("source", "w", "out"))
    assert all(g.get_node_attr(n, "out") is None for n in nodes)
