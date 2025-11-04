"""Debug PageRank divergence between builder and native."""
import sys
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms import centrality


def build_pagerank_algorithm(damping=0.85, max_iter=100):
    """Build PageRank using the builder DSL with proper primitives."""
    builder = AlgorithmBuilder("custom_pagerank")
    
    # Get node count from the graph at runtime
    node_count = builder.graph_node_count()
    
    # Initialize ranks uniformly (will be 1/N at runtime)
    ranks = builder.init_nodes(default=1.0)
    inv_n_scalar = builder.core.recip(node_count, epsilon=1e-9)
    uniform = builder.core.broadcast_scalar(inv_n_scalar, ranks)
    ranks = builder.var("ranks", uniform)
    
    # Compute out-degrees  
    degrees = builder.node_degrees(ranks)
    
    # Safe reciprocal for division (avoid division by zero)
    inv_degrees = builder.core.recip(degrees, epsilon=1e-9)
    
    # Identify sinks (nodes with no outgoing edges)
    is_sink = builder.core.compare(degrees, "eq", 0.0)
    
    with builder.iterate(max_iter):
        # Compute contribution from each node: rank / out_degree
        contrib = builder.core.mul(ranks, inv_degrees)
        contrib = builder.core.where(is_sink, 0.0, contrib)
        
        # Sum neighbor contributions (via incoming edges)
        neighbor_sum = builder.core.neighbor_agg(contrib, agg="sum")
        
        # Apply damping to neighbor contributions
        damped_neighbors = builder.core.mul(neighbor_sum, damping)
        
        # Compute teleport term: (1-damping)/N broadcast to all nodes
        inv_n_map = builder.core.broadcast_scalar(inv_n_scalar, degrees)
        teleport_map = builder.core.mul(inv_n_map, 1.0 - damping)
        
        # Handle sink redistribution: collect rank from sinks and redistribute
        sink_ranks = builder.core.where(is_sink, ranks, 0.0)
        sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
        sink_map = builder.core.mul(inv_n_map, sink_mass)
        sink_map = builder.core.mul(sink_map, damping)
        
        # Combine all components
        updated = builder.core.add(damped_neighbors, teleport_map)
        updated = builder.core.add(updated, sink_map)
        ranks = builder.var("ranks", updated)
    
    # Normalize once after iterations to ensure sum = 1.0
    ranks = builder.core.normalize_sum(ranks)
    
    builder.attach_as("pagerank", ranks)
    return builder.build()


def test_simple_chain():
    """Test on a simple 3-node chain: 0 -> 1 -> 2"""
    print("=" * 60)
    print("Testing simple chain: 0 -> 1 -> 2")
    print("=" * 60)
    
    g = Graph(directed=True)
    n0 = g.add_node()
    n1 = g.add_node()
    n2 = g.add_node()
    g.add_edge(n0, n1)
    g.add_edge(n1, n2)
    
    damping = 0.85
    
    # Native
    print("\nNative PageRank:")
    sg = g.view()
    result_native, _ = sg.apply(
        centrality.pagerank(max_iter=100, damping=damping, output_attr="pr_native"),
        persist=True,
        return_profile=True
    )
    
    for node in result_native.nodes:
        print(f"  Node {node.id}: {node.pr_native:.10f}")
    
    # Builder
    print("\nBuilder PageRank:")
    algo = build_pagerank_algorithm(damping=damping, max_iter=100)
    result_builder = sg.apply(algo)
    
    builder_map = {node.id: node.pagerank for node in result_builder.nodes}
    for node_id in [n0, n1, n2]:
        print(f"  Node {node_id}: {builder_map[node_id]:.10f}")
    
    # Compare
    print("\nDifferences:")
    for node in result_native.nodes:
        native_val = node.pr_native
        builder_val = builder_map[node.id]
        diff = abs(native_val - builder_val)
        print(f"  Node {node.id}: diff = {diff:.10e}")
    
    return result_native, result_builder


def test_simple_loop():
    """Test on a simple cycle: 0 -> 1 -> 2 -> 0"""
    print("\n" + "=" * 60)
    print("Testing simple cycle: 0 -> 1 -> 2 -> 0")
    print("=" * 60)
    
    g = Graph(directed=True)
    n0 = g.add_node()
    n1 = g.add_node()
    n2 = g.add_node()
    g.add_edge(n0, n1)
    g.add_edge(n1, n2)
    g.add_edge(n2, n0)
    
    damping = 0.85
    
    # Native
    print("\nNative PageRank:")
    sg = g.view()
    result_native, _ = sg.apply(
        centrality.pagerank(max_iter=100, damping=damping, output_attr="pr_native"),
        persist=True,
        return_profile=True
    )
    
    for node in result_native.nodes:
        print(f"  Node {node.id}: {node.pr_native:.10f}")
    
    # Builder
    print("\nBuilder PageRank:")
    algo = build_pagerank_algorithm(damping=damping, max_iter=100)
    result_builder = sg.apply(algo)
    
    builder_map = {node.id: node.pagerank for node in result_builder.nodes}
    for node_id in [n0, n1, n2]:
        print(f"  Node {node_id}: {builder_map[node_id]:.10f}")
    
    # Compare
    print("\nDifferences:")
    for node in result_native.nodes:
        native_val = node.pr_native
        builder_val = builder_map[node.id]
        diff = abs(native_val - builder_val)
        print(f"  Node {node.id}: diff = {diff:.10e}")
    
    return result_native, result_builder


def test_with_sink():
    """Test with a sink node: 0 -> 1, 2 (isolated sink)"""
    print("\n" + "=" * 60)
    print("Testing with sink: 0 -> 1, 2 (isolated sink)")
    print("=" * 60)
    
    g = Graph(directed=True)
    n0 = g.add_node()
    n1 = g.add_node()
    n2 = g.add_node()  # sink (no outgoing edges)
    g.add_edge(n0, n1)
    g.add_edge(n1, n2)
    
    damping = 0.85
    
    # Native
    print("\nNative PageRank:")
    sg = g.view()
    result_native, _ = sg.apply(
        centrality.pagerank(max_iter=100, damping=damping, output_attr="pr_native"),
        persist=True,
        return_profile=True
    )
    
    for node in result_native.nodes:
        print(f"  Node {node.id}: {node.pr_native:.10f}")
    
    # Builder
    print("\nBuilder PageRank:")
    algo = build_pagerank_algorithm(damping=damping, max_iter=100)
    result_builder = sg.apply(algo)
    
    builder_map = {node.id: node.pagerank for node in result_builder.nodes}
    for node_id in [n0, n1, n2]:
        print(f"  Node {node_id}: {builder_map[node_id]:.10f}")
    
    # Compare
    print("\nDifferences:")
    for node in result_native.nodes:
        native_val = node.pr_native
        builder_val = builder_map[node_id]
        diff = abs(native_val - builder_val)
        print(f"  Node {node.id}: diff = {diff:.10e}")
    
    return result_native, result_builder


if __name__ == "__main__":
    test_simple_chain()
    test_simple_loop()
    test_with_sink()
    print("\n" + "=" * 60)
    print("Done!")
