"""
Tests for Phase 4 Python algorithm API.
"""

import pytest

import groggy
from groggy import Graph, algorithms, pipeline


def build_test_graph():
    """Build a small test graph."""
    g = Graph()
    nodes = [g.add_node() for _ in range(10)]

    # Create connected graph
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    g.add_edge(nodes[-1], nodes[0])  # Make it cyclic

    # Add some cross-edges
    g.add_edge(nodes[0], nodes[5])
    g.add_edge(nodes[2], nodes[7])

    return g, nodes


# ==============================================================================
# Test Algorithm Handles
# ==============================================================================


def test_algorithm_function():
    """Test the generic algorithm() factory function."""
    algo = algorithms.algorithm("centrality.pagerank", max_iter=20, damping=0.85)
    assert algo.id == "centrality.pagerank"
    assert isinstance(algo, algorithms.RustAlgorithmHandle)


def test_algorithm_handle_with_params():
    """Test algorithm handle parameter updates."""
    algo = algorithms.centrality.pagerank(max_iter=10)
    algo2 = algo.with_params(max_iter=50, damping=0.9)

    # Original should be unchanged
    assert algo._params["max_iter"] == 10
    # New handle should have updated params
    assert algo2._params["max_iter"] == 50
    assert algo2._params["damping"] == 0.9


def test_algorithm_handle_validation():
    """Test parameter validation."""
    algo = algorithms.centrality.pagerank(max_iter=20)
    # Should validate successfully
    assert algo.validate() == True

    # Invalid params should raise
    bad_algo = algorithms.algorithm("centrality.pagerank", unknown_param=123)
    with pytest.raises(ValueError, match="unknown_param"):
        bad_algo.validate()


def test_algorithm_handle_to_spec():
    """Test conversion to pipeline spec."""
    algo = algorithms.centrality.pagerank(max_iter=20, output_attr="pr")
    spec = algo.to_spec()

    assert spec["id"] == "centrality.pagerank"
    assert "params" in spec
    assert "max_iter" in spec["params"]


# ==============================================================================
# Test Algorithm Module Functions
# ==============================================================================


def test_centrality_pagerank():
    """Test PageRank algorithm handle."""
    pr = algorithms.centrality.pagerank(max_iter=50, damping=0.9)
    assert pr.id == "centrality.pagerank"


def test_centrality_betweenness():
    """Test betweenness centrality handle."""
    bc = algorithms.centrality.betweenness(normalized=True)
    assert bc.id == "centrality.betweenness"


def test_centrality_closeness():
    """Test closeness centrality handle."""
    cc = algorithms.centrality.closeness()
    assert cc.id == "centrality.closeness"


def test_community_lpa():
    """Test LPA algorithm handle."""
    lpa = algorithms.community.lpa(max_iter=50)
    assert lpa.id == "community.lpa"


def test_community_louvain():
    """Test Louvain algorithm handle."""
    louv = algorithms.community.louvain(resolution=1.5)
    assert louv.id == "community.louvain"


def test_pathfinding_dijkstra():
    """Test Dijkstra algorithm handle."""
    dijk = algorithms.pathfinding.dijkstra(start_attr="is_start")
    assert dijk.id == "pathfinding.dijkstra"


def test_pathfinding_bfs():
    """Test BFS algorithm handle."""
    bfs = algorithms.pathfinding.bfs(start_attr="is_root")
    assert bfs.id == "pathfinding.bfs"


def test_pathfinding_astar():
    """Test A* algorithm handle."""
    astar = algorithms.pathfinding.astar(
        start_attr="is_start", goal_attr="is_goal", heuristic_attr="h"
    )
    assert astar.id == "pathfinding.astar"


# ==============================================================================
# Test Pipeline Class
# ==============================================================================


def test_pipeline_creation():
    """Test pipeline creation."""
    pipe = pipeline(
        [
            algorithms.centrality.pagerank(max_iter=20),
            algorithms.centrality.betweenness(),
        ]
    )
    assert len(pipe) == 2
    assert isinstance(pipe, groggy.Pipeline)


def test_pipeline_execution():
    """Test pipeline execution on a graph."""
    g, nodes = build_test_graph()
    sub = g.induced_subgraph(nodes)

    pipe = pipeline([algorithms.centrality.pagerank(max_iter=20, output_attr="pr")])

    result = pipe.run(sub)
    assert result is not None
    assert len(result.nodes) == len(nodes)


def test_pipeline_callable():
    """Test that pipeline can be called as a function."""
    g, nodes = build_test_graph()
    sub = g.induced_subgraph(nodes)

    pipe = pipeline([algorithms.centrality.pagerank(max_iter=10, output_attr="rank")])

    # Should work with __call__
    result = pipe(sub)
    assert result is not None


def test_pipeline_multi_algorithm():
    """Test pipeline with multiple algorithms."""
    g, nodes = build_test_graph()
    g.nodes.set_attrs({nodes[0]: {"is_start": True}})
    sub = g.induced_subgraph(nodes)

    pipe = pipeline(
        [
            algorithms.centrality.pagerank(max_iter=20, output_attr="pr"),
            algorithms.pathfinding.bfs(start_attr="is_start", output_attr="dist"),
        ]
    )

    result = pipe(sub)
    assert result is not None


# ==============================================================================
# Test Discovery Functions
# ==============================================================================


def test_algorithms_list():
    """Test listing all algorithms."""
    algos = algorithms.list()
    assert isinstance(algos, list)
    assert "centrality.pagerank" in algos
    assert "community.lpa" in algos


def test_algorithms_list_by_category():
    """Test listing algorithms by category."""
    centrality_algos = algorithms.list(category="centrality")
    assert isinstance(centrality_algos, list)
    assert "centrality.pagerank" in centrality_algos
    assert "community.lpa" not in centrality_algos


def test_algorithms_categories():
    """Test getting all categories."""
    cats = algorithms.categories()
    assert isinstance(cats, dict)
    assert "centrality" in cats
    assert "community" in cats
    assert "pathfinding" in cats


def test_algorithms_info():
    """Test getting algorithm info."""
    info = algorithms.info("centrality.pagerank")
    assert info["id"] == "centrality.pagerank"
    assert "description" in info
    assert "parameters" in info
    assert isinstance(info["parameters"], list)


def test_algorithms_search():
    """Test searching for algorithms."""
    results = algorithms.search("pagerank")
    assert "centrality.pagerank" in results

    results = algorithms.search("community")
    assert any("community" in r for r in results)


# ==============================================================================
# Integration Tests
# ==============================================================================


def test_full_integration_example():
    """Test a complete workflow using the high-level API."""
    # Create graph
    g, nodes = build_test_graph()

    # Set start node for BFS
    g.nodes.set_attrs({nodes[0]: {"is_start": True}})

    # Create subgraph
    sub = g.induced_subgraph(nodes)

    # Create and run pipeline
    pipe = pipeline(
        [
            algorithms.centrality.pagerank(max_iter=20, output_attr="pagerank"),
            algorithms.pathfinding.bfs(start_attr="is_start", output_attr="distance"),
        ]
    )

    result = pipe(sub)

    # Verify result
    assert result is not None
    assert len(result.nodes) == len(nodes)


def test_algorithm_reuse():
    """Test that algorithm handles can be reused."""
    g, nodes = build_test_graph()
    sub1 = g.induced_subgraph(nodes[:5])
    sub2 = g.induced_subgraph(nodes[5:])

    algo = algorithms.centrality.pagerank(max_iter=20)

    pipe1 = pipeline([algo])
    pipe2 = pipeline([algo])

    result1 = pipe1(sub1)
    result2 = pipe2(sub2)

    assert result1 is not None
    assert result2 is not None


def test_mixed_spec_types():
    """Test pipeline with mixed algorithm handles and dict specs."""
    g, nodes = build_test_graph()
    sub = g.induced_subgraph(nodes)

    # Mix handle and raw spec
    pipe = pipeline(
        [
            algorithms.centrality.pagerank(max_iter=20, output_attr="pr"),
            {
                "id": "centrality.betweenness",
                "params": {
                    "output_attr": groggy.AttrValue("bc"),
                    "normalized": groggy.AttrValue(True),
                },
            },
        ]
    )

    result = pipe(sub)
    assert result is not None
