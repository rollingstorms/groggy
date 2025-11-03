"""
Tests for PageRank algorithm built with the builder DSL.
"""
import pytest


def _pagerank_step(builder, ranks, node_count, damping=0.85):
    """
    Execute one PageRank iteration using builder primitives that match the native implementation.
    """
    degrees = builder.node_degrees(ranks)
    inv_degrees = builder.core.recip(degrees, epsilon=1e-9)
    is_sink = builder.core.compare(degrees, "eq", 0.0)

    weighted = builder.core.mul(ranks, inv_degrees)
    weighted = builder.core.where(is_sink, 0.0, weighted)

    neighbor_sums = builder.core.neighbor_agg(weighted, agg="sum")

    damped = builder.core.mul(neighbor_sums, damping)

    inv_n_scalar = builder.core.recip(node_count, epsilon=1e-9)
    inv_n_map = builder.core.broadcast_scalar(inv_n_scalar, degrees)
    teleport_map = builder.core.mul(inv_n_map, 1.0 - damping)

    sink_ranks = builder.core.where(is_sink, ranks, 0.0)
    sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
    sink_map = builder.core.mul(inv_n_map, sink_mass)
    sink_map = builder.core.mul(sink_map, damping)

    total = builder.core.add(damped, teleport_map)
    total = builder.core.add(total, sink_map)
    ranks = builder.var("ranks", total)
    return ranks


def test_builder_pagerank_basic():
    """Build PageRank using the builder DSL."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder

    # Create test graph (3-node cycle)
    graph = Graph(directed=True)
    a, b, c = graph.add_node(), graph.add_node(), graph.add_node()
    graph.add_edge(a, b)
    graph.add_edge(b, c)
    graph.add_edge(c, a)
    sg = graph.view()
    
    # Build PageRank with builder
    builder = AlgorithmBuilder("builder_pagerank_basic")
    
    # Initialize ranks to 1.0
    ranks = builder.init_nodes(default=1.0)
    node_count = builder.graph_node_count()
    inv_n_scalar = builder.core.recip(node_count, epsilon=1e-9)
    uniform = builder.core.broadcast_scalar(inv_n_scalar, ranks)
    ranks = builder.var("ranks", uniform)
    
    # Iterate 20 times using the full PageRank update
    with builder.iterate(20):
        ranks = _pagerank_step(builder, ranks, node_count, damping=0.85)
    
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
    from groggy.algorithms.centrality import pagerank
    
    # Create test graph
    graph = Graph(directed=True)
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
    builder = AlgorithmBuilder("builder_pagerank_matches_native")
    ranks = builder.init_nodes(default=1.0)
    node_count = builder.graph_node_count()
    inv_n_scalar = builder.core.recip(node_count, epsilon=1e-9)
    uniform = builder.core.broadcast_scalar(inv_n_scalar, ranks)
    ranks = builder.var("ranks", uniform)
    
    with builder.iterate(20):
        ranks = _pagerank_step(builder, ranks, node_count, damping=0.85)
    
    builder.attach_as("pagerank", ranks)
    
    # Execute both
    algo = builder.build()
    result_builder = sg.apply(algo)
    builder_values = {node.id: node.pagerank for node in result_builder.nodes}

    # Run native PageRank on a fresh view of the same graph
    result_native = graph.view().apply(pagerank(max_iter=20, damping=0.85))
    
    # Compare results
    for node_id, pr_builder in builder_values.items():
        pr_native = result_native.get_node_attribute(node_id, "pagerank")

        assert pr_builder is not None
        assert pr_native is not None
        assert abs(pr_builder - pr_native) < 1e-6


def test_builder_pagerank_converges():
    """Test that PageRank converges to stable values."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder
    
    # Create simple graph
    graph = Graph(directed=True)
    a, b = graph.add_node(), graph.add_node()
    graph.add_edge(a, b)
    graph.add_edge(b, a)
    sg = graph.view()
    
    # Build PageRank
    builder = AlgorithmBuilder("builder_pagerank_converges")
    ranks = builder.init_nodes(default=1.0)
    node_count = builder.graph_node_count()
    inv_n_scalar = builder.core.recip(node_count, epsilon=1e-9)
    uniform = builder.core.broadcast_scalar(inv_n_scalar, ranks)
    ranks = builder.var("ranks", uniform)
    
    # Run for many iterations
    with builder.iterate(50):
        ranks = _pagerank_step(builder, ranks, node_count, damping=0.85)
    
    builder.attach_as("pagerank", ranks)
    
    # Execute
    algo = builder.build()
    result = sg.apply(algo)
    
    # For a symmetric 2-node graph, ranks should be equal
    nodes = list(result.nodes)
    pr1 = nodes[0].pagerank
    pr2 = nodes[1].pagerank
    
    # Should both be 0.5 (equal ranks)
    assert abs(pr1 - 0.5) < 0.01
    assert abs(pr2 - 0.5) < 0.01


def test_builder_pagerank_no_edges():
    """Test PageRank on graph with no edges."""
    from groggy import Graph
    from groggy.builder import AlgorithmBuilder
    
    # Create isolated nodes
    graph = Graph(directed=True)
    a, b, c = graph.add_node(), graph.add_node(), graph.add_node()
    sg = graph.view()
    
    # Build PageRank
    builder = AlgorithmBuilder("builder_pagerank_no_edges")
    ranks = builder.init_nodes(default=1.0)
    node_count = builder.graph_node_count()
    
    with builder.iterate(10):
        ranks = _pagerank_step(builder, ranks, node_count, damping=0.85)
    
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
