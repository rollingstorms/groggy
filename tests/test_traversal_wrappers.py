import pytest
from groggy import Graph, algorithms
from groggy.pipeline import apply


def fixture(directed=True):
    g = Graph(directed=directed)
    nodes = [g.add_node() for _ in range(5)]
    # Reverse insertion forces DFS to sort neighbors rather than trust CSR order.
    for u, v in [(0, 2), (0, 1), (2, 3), (1, 3), (0, 2), (1, 1)]:
        g.add_edge(nodes[u], nodes[v])
    g.set_node_attr(nodes[0], "start", False)
    return g, nodes


def test_pipeline_returns_exact_bfs_distances_and_deterministic_dfs_preorder():
    g, nodes = fixture()
    sub = g.induced_subgraph(nodes)
    # Warm shared adjacency through another algorithm.
    sub = apply(sub, algorithms.centrality.pagerank(max_iter=1), persist=False)
    _, profile = apply(sub, [
        algorithms.pathfinding.bfs("start", "b"),
        algorithms.pathfinding.dfs("start", "d"),
    ], persist=False, return_profile=True)
    rows = profile["outputs"]
    assert rows["pathfinding.bfs.distances"] == [(nodes[0], 0), (nodes[1], 1), (nodes[2], 1), (nodes[3], 2)]
    assert rows["pathfinding.dfs.order"] == [(nodes[0], 0), (nodes[1], 1), (nodes[3], 2), (nodes[2], 3)]
    assert all(type(value) is int for _, value in rows["pathfinding.dfs.order"])
    assert all(g.get_node_attr(n, "b") is None and g.get_node_attr(n, "d") is None for n in nodes)


@pytest.mark.parametrize("name", ["bfs", "dfs"])
@pytest.mark.parametrize("markers", [0, 2])
def test_missing_or_ambiguous_source_rejects_before_persistence(name, markers):
    g = Graph()
    nodes = [g.add_node() for _ in range(3)]
    for node in nodes[:markers]:
        g.set_node_attr(node, "start", True)
    g.set_node_attr(nodes[0], "out", 7)
    algo = getattr(algorithms.pathfinding, name)("start", "out")
    with pytest.raises(RuntimeError, match="exactly one"):
        g.induced_subgraph(nodes).apply(algo)
    assert g.get_node_attr(nodes[0], "out") == 7


@pytest.mark.parametrize("marker", [False, 0, None, "marker"])
@pytest.mark.parametrize("name", ["bfs", "dfs"])
def test_source_attribute_presence_is_the_contract(name, marker):
    g = Graph()
    node = g.add_node()
    g.set_node_attr(node, "start", marker)
    g.induced_subgraph([node]).apply(getattr(algorithms.pathfinding, name)("start", "out"))
    assert g.get_node_attr(node, "out") == 0


@pytest.mark.parametrize("directed", [False, True])
def test_direction_and_selected_boundary_are_respected(directed):
    g = Graph(directed=directed)
    a, b, outside = [g.add_node() for _ in range(3)]
    g.add_edge(a, b)
    g.add_edge(b, outside)
    g.set_node_attr(b, "start", True)
    sub = g.induced_subgraph([a, b])
    for name in ["bfs", "dfs"]:
        output = "pathfinding.bfs.distances" if name == "bfs" else "pathfinding.dfs.order"
        _, profile = apply(sub, getattr(algorithms.pathfinding, name)("start"), persist=False, return_profile=True)
        reached = {node for node, _ in profile["outputs"][output]}
        assert reached == ({b} if directed else {a, b})
        assert outside not in reached


def test_persistent_output_uses_integer_properties_for_reached_nodes():
    g, nodes = fixture()
    sub = g.induced_subgraph(nodes)
    sub.apply(algorithms.pathfinding.bfs("start", "b"))
    sub.apply(algorithms.pathfinding.dfs("start", "d"))
    assert [g.get_node_attr(n, "b") for n in nodes] == [0, 1, 1, 2, None]
    assert [g.get_node_attr(n, "d") for n in nodes] == [0, 1, 3, 2, None]
