import pytest
from groggy import Graph, algorithms
from groggy.pipeline import apply


def test_nonpersistent_astar_returns_ordered_integer_path_and_f64_cost():
    g = Graph()
    nodes = [g.add_node() for _ in range(4)]
    g.set_node_attr(nodes[2], "s", False)
    g.set_node_attr(nodes[0], "g", None)
    for u, v, w in [(0, 2, 9), (0, 1, 2), (1, 2, 2), (0, 1, 8)]:
        e = g.add_edge(nodes[u], nodes[v])
        g.set_edge_attr(e, "w", w)
    for node, h in zip(nodes, [0, 2, 4, 0]):
        g.set_node_attr(node, "h", h)
    sub = apply(g.induced_subgraph(nodes), algorithms.pathfinding.bfs("s"), persist=False)
    result, profile = apply(sub, algorithms.pathfinding.astar("s", "g", "h", "w", "out"),
                            persist=False, return_profile=True)
    rows = profile["outputs"]["pathfinding.astar.path"]
    assert rows == [(nodes[2], 0), (nodes[1], 1), (nodes[0], 2)]
    assert all(type(index) is int for _, index in rows)
    assert profile["stats"]["astar.path_cost"] == 4.0
    assert profile["stats"]["astar.count.path_nodes"] == 3.0
    assert "astar.csr_cache_hit" in profile["call_counters"]
    assert "out" not in result.nodes.attribute_names()
    assert all(g.get_node_attr(node, "out") is None for node in nodes)


def test_start_equals_goal_returns_one_row_without_mutation():
    g = Graph()
    node = g.add_node()
    g.set_node_attr(node, "s", 0)
    g.set_node_attr(node, "g", False)
    g.set_node_attr(node, "out", 7)
    _, profile = apply(g.induced_subgraph([node]), algorithms.pathfinding.astar("s", "g", output_attr="out"),
                       persist=False, return_profile=True)
    assert profile["outputs"]["pathfinding.astar.path"] == [(node, 0)]
    assert profile["stats"]["astar.path_cost"] == 0.0
    assert g.get_node_attr(node, "out") == 7


def test_unreachable_goal_preserves_old_output():
    g = Graph(directed=True)
    a, b = nodes = [g.add_node() for _ in range(2)]
    g.add_edge(a, b)
    g.set_node_attr(b, "s", True)
    g.set_node_attr(a, "g", True)
    g.set_node_attr(b, "out", 7)
    with pytest.raises(RuntimeError, match="unreachable"):
        apply(g.induced_subgraph(nodes), algorithms.pathfinding.astar("s", "g", output_attr="out"),
              persist=False, return_profile=True)
    assert g.get_node_attr(b, "out") == 7
