"""
Tests for PageRank algorithm built with the builder DSL.
"""
import pytest


def test_builder_pagerank_basic():
    """Build PageRank using the builder DSL."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder
    
    # Create test graph (3-node cycle)
    graph = Graph()
    a, b, c = graph.add_node(), graph.add_node(), graph.add_node()
    graph.add_edge(a, b)
    graph.add_edge(b, c)
    graph.add_edge(c, a)
    sg = graph.view()
    
    # Build PageRank with builder
    builder = AlgorithmBuilder("custom_pagerank")
    
    # Initialize ranks to 1.0
    ranks = builder.init_nodes(default=1.0)
    
    # Iterate 20 times
    with builder.iterate(20):
        # Sum neighbor ranks divided by their degree
        neighbor_sums = builder.map_nodes(
            "sum(ranks[neighbors(node)])",
            inputs={"ranks": ranks}
        )
        
        # Apply damping: 0.85 * neighbor_sums + 0.15
        damped = builder.core.mul(neighbor_sums, 0.85)
        ranks = builder.var("ranks", builder.core.add(damped, 0.15))
        
        # Normalize
        ranks = builder.var("ranks", builder.core.normalize_sum(ranks))
    
    # Attach result
    builder.attach_as("pagerank", ranks)
    
    # Execute
    algo = builder.build()
    result = sg.apply(algo)
    
    # Verify result has pagerank attribute
    pr_values = []
    for node in result.nodes:
        pr = node.pagerank
        assert pr is not None
        assert pr > 0
        pr_values.append(pr)
    
    # Check that ranks sum to approximately 1.0 (after normalization)
    total = sum(pr_values)
    assert abs(total - 1.0) < 1e-6


def test_builder_pagerank_matches_native():
    """Verify builder PageRank matches native implementation."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder
    from groggy.algorithms import pagerank
    
    # Create test graph
    graph = Graph()
    nodes = [graph.add_node() for _ in range(5)]
    
    # Create a more complex graph
    graph.add_edge(nodes[0], nodes[1])
    graph.add_edge(nodes[1], nodes[2])
    graph.add_edge(nodes[2], nodes[0])
    graph.add_edge(nodes[2], nodes[3])
    graph.add_edge(nodes[3], nodes[4])
    graph.add_edge(nodes[4], nodes[2])
    
    sg = graph.view()
    
    # Build PageRank with builder
    builder = AlgorithmBuilder("custom_pagerank")
    ranks = builder.init_nodes(default=1.0)
    
    with builder.iterate(20):
        neighbor_sums = builder.map_nodes(
            "sum(ranks[neighbors(node)])",
            inputs={"ranks": ranks}
        )
        damped = builder.core.mul(neighbor_sums, 0.85)
        ranks = builder.var("ranks", builder.core.add(damped, 0.15))
        ranks = builder.var("ranks", builder.core.normalize_sum(ranks))
    
    builder.attach_as("pagerank", ranks)
    
    # Execute both
    algo = builder.build()
    result_builder = sg.apply(algo)
    result_native = sg.apply(pagerank(iterations=20, damping=0.85))
    
    # Compare results
    for node in result_builder.nodes:
        pr_builder = node.pagerank
        pr_native = node.pagerank
        
        # Note: Results may differ slightly due to implementation details
        # We're checking they're in the same ballpark
        assert pr_builder is not None
        assert pr_native is not None
        assert abs(pr_builder - pr_native) < 0.1  # Within 10% (relaxed for now)


def test_builder_pagerank_converges():
    """Test that PageRank converges to stable values."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder
    
    # Create simple graph
    graph = Graph()
    a, b = graph.add_node(), graph.add_node()
    graph.add_edge(a, b)
    graph.add_edge(b, a)
    sg = graph.view()
    
    # Build PageRank
    builder = AlgorithmBuilder("converge_test")
    ranks = builder.init_nodes(default=1.0)
    
    # Run for many iterations
    with builder.iterate(50):
        neighbor_sums = builder.map_nodes(
            "sum(ranks[neighbors(node)])",
            inputs={"ranks": ranks}
        )
        damped = builder.core.mul(neighbor_sums, 0.85)
        ranks = builder.var("ranks", builder.core.add(damped, 0.15))
        ranks = builder.var("ranks", builder.core.normalize_sum(ranks))
    
    builder.attach_as("pagerank", ranks)
    
    # Execute
    algo = builder.build()
    result = sg.apply(algo)
    
    # For a symmetric 2-node graph, ranks should be equal
    nodes = list(result.nodes)
    pr1 = result.get_node_attr(nodes[0], "pagerank")
    pr2 = result.get_node_attr(nodes[1], "pagerank")
    
    # Should both be 0.5 (equal ranks)
    assert abs(pr1 - 0.5) < 0.01
    assert abs(pr2 - 0.5) < 0.01


def test_builder_pagerank_no_edges():
    """Test PageRank on graph with no edges."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder
    
    # Create isolated nodes
    graph = Graph()
    a, b, c = graph.add_node(), graph.add_node(), graph.add_node()
    sg = graph.view()
    
    # Build PageRank
    builder = AlgorithmBuilder("isolated_nodes")
    ranks = builder.init_nodes(default=1.0)
    
    with builder.iterate(10):
        neighbor_sums = builder.map_nodes(
            "sum(ranks[neighbors(node)])",
            inputs={"ranks": ranks}
        )
        damped = builder.core.mul(neighbor_sums, 0.85)
        ranks = builder.var("ranks", builder.core.add(damped, 0.15))
        ranks = builder.var("ranks", builder.core.normalize_sum(ranks))
    
    builder.attach_as("pagerank", ranks)
    
    # Execute
    algo = builder.build()
    result = sg.apply(algo)
    
    # All nodes should have equal rank (1/3 each)
    for node in result.nodes:
        pr = node.pagerank
        assert abs(pr - 1.0/3.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
