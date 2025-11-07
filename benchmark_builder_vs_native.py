"""
Benchmark builder-based PageRank and LPA vs native implementations.
Tests on 50k and 200k node graphs.
Validates that results match between builder and native implementations.

Updated to use new decorator-based DSL syntax.
"""
import time
from groggy import Graph, print_profile
from groggy.builder import algorithm
from groggy.algorithms import centrality, community

# Set to True to see detailed per-step profiling
SHOW_PROFILING = False


@algorithm("pagerank")
def pagerank(sG, damping=0.85, max_iter=100):
    """Build simple PageRank using the new decorator-based DSL."""
    # Initialize ranks uniformly
    ranks = sG.nodes(1.0)
    
    # Simple iteration without sink handling for now
    with sG.builder.iter.loop(max_iter):
        # Aggregate neighbor ranks (without normalization for simplicity)
        neighbor_sum = sG @ ranks
        
        # Update: damped neighbor contributions + teleport
        ranks = sG.builder.var("ranks",
            damping * neighbor_sum + (1 - damping))
    
    return ranks.normalize()


@algorithm("lpa")
def lpa(sG, max_iter=10):
    """Build LPA using the new decorator-based DSL."""
    # Initialize each node with unique label (use node sequence)
    labels = sG.nodes(unique=True)
    labels = sG.var("labels", labels)  # Create logical loop variable
    b = sG.builder
    
    with sG.iterate(max_iter):
        updated = b.core.neighbor_mode_update(
            target=labels,
            include_self=True,
            tie_break="lowest",
            ordered=True,
        )
        labels = sG.var("labels", updated)
    
    return labels


def create_test_graph(num_nodes, avg_degree=10):
    """Create a random graph for testing (creates one large connected component)."""
    import random
    random.seed(42)
    
    graph = Graph()  # Undirected by default
    
    # Create nodes explicitly
    nodes = []
    for i in range(num_nodes):
        nodes.append(graph.add_node())
    
    # Generate edges efficiently
    num_edges = num_nodes * avg_degree // 2
    edges_data = []
    seen = set()
    
    for _ in range(num_edges * 2):  # Try enough times
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)
        if i != j:
            # Normalize for undirected graph
            edge = (min(i, j), max(i, j))
            if edge not in seen:
                edges_data.append((nodes[i], nodes[j]))
                seen.add(edge)
                if len(edges_data) >= num_edges:
                    break
    
    # Add all edges at once
    graph.add_edges(edges_data)
    
    print(f"Created graph: {num_nodes} nodes, {len(edges_data)} edges")
    print(f"  (Note: Dense random graphs create 1 giant component - LPA converges to few communities)")
    return graph


def benchmark_pagerank(graph, name):
    """Benchmark PageRank builder vs native."""
    print(f"\n{'='*60}")
    print(f"PageRank Benchmark - {name}")
    print(f"{'='*60}")
    
    sg = graph.view()
    n = len(list(sg.nodes))
    
    # Native version
    print("\nNative PageRank:")
    start = time.perf_counter()
    result_native, profile_native = sg.apply(
        centrality.pagerank(max_iter=100, damping=0.85, output_attr="pagerank_native"),
        persist=True,  # Need to persist to access attributes
        return_profile=True
    )
    native_time = time.perf_counter() - start
    
    print(f"  Time: {native_time:.3f}s")
    
    if SHOW_PROFILING:
        print_profile(profile_native, show_steps=True, show_details=False)
    
    # Get some sample values
    native_nodes = list(result_native.nodes)
    sample_nodes = [node.id for node in native_nodes[:5]]
    print(f"  Sample values:")
    for node_id in sample_nodes:
        native_val = result_native.get_node_attribute(node_id, "pagerank_native")
        print(f"    Node {node_id}: {native_val:.6f}")
    
    # Verify normalization
    total_rank_native = sum(node.pagerank_native for node in native_nodes)
    print(f"  Total rank: {total_rank_native:.6f}")
    
    # Builder version
    print("\nBuilder PageRank:")
    algo = pagerank(damping=0.85, max_iter=100)
    
    start = time.perf_counter()
    result_builder, profile_builder = sg.apply(algo, return_profile=True)
    builder_time = time.perf_counter() - start
    
    print(f"  Time: {builder_time:.3f}s")
    
    if SHOW_PROFILING:
        print_profile(profile_builder, show_steps=True, show_details=False)
    
    # Get some sample values
    print(f"  Sample values:")
    builder_nodes = list(result_builder.nodes)
    builder_map = {node.id: node.pagerank for node in builder_nodes}
    for node_id in sample_nodes:
        builder_val = builder_map.get(node_id, 0.0)
        print(f"    Node {node_id}: {builder_val:.6f}")
    
    # Verify normalization
    total_rank = sum(builder_map.values())
    print(f"  Total rank: {total_rank:.6f}")
    
    # Compare results
    print(f"\n  Comparison:")
    print(f"    Builder/Native time ratio: {builder_time/native_time:.2f}x")
    
    # Check if values are similar
    diffs = []
    for node in native_nodes:
        node_id = node.id
        builder_val = builder_map.get(node_id, 0.0)
        native_val = node.pagerank_native
        diffs.append(abs(builder_val - native_val))
    
    avg_diff = sum(diffs) / len(diffs)
    max_diff = max(diffs)
    print(f"    Avg difference: {avg_diff:.8f}")
    print(f"    Max difference: {max_diff:.8f}")
    
    # Reasonable tolerance for convergence
    # Avg should be very tight, max can be slightly looser due to edge cases
    if avg_diff < 0.000001 and max_diff < 0.0001:
        print(f"    ✅ Results match!")
    elif max_diff < 0.001:
        print(f"    ⚠️  Results differ slightly (max diff: {max_diff:.10f})")
    else:
        print(f"    ❌ Results differ significantly (max diff: {max_diff:.10f})")
    
    return builder_time, native_time, result_builder


def benchmark_lpa(graph, name):
    """Benchmark LPA builder vs native."""
    print(f"\n{'='*60}")
    print(f"LPA Benchmark - {name}")
    print(f"{'='*60}")
    
    sg = graph.view()
    
    # Native version
    print("\nNative LPA:")
    start = time.perf_counter()
    result_native, stats_native = sg.apply(
        community.lpa(max_iter=10, output_attr="community_native"),
        persist=True,  # Need to persist to access attributes
        return_profile=True
    )
    native_time = time.perf_counter() - start
    
    print(f"  Time: {native_time:.3f}s")
    
    # Count communities
    communities_native = {}
    for node in result_native.nodes:
        comm = node.community_native
        communities_native[comm] = communities_native.get(comm, 0) + 1
    
    print(f"  Communities found: {len(communities_native)}")
    print(f"  Top 5 communities by size:")
    top_comms_native = sorted(communities_native.items(), key=lambda x: x[1], reverse=True)[:5]
    for comm_id, size in top_comms_native:
        print(f"    Community {comm_id}: {size} nodes")
    
    # Builder version
    print("\nBuilder LPA:")
    algo = lpa(max_iter=10)
    
    start = time.perf_counter()
    result_builder, profile_builder = sg.apply(algo, return_profile=True)
    builder_time = time.perf_counter() - start
    
    print(f"  Time: {builder_time:.3f}s")
    
    if SHOW_PROFILING:
        print_profile(profile_builder, show_steps=True, show_details=False)
    
    # Count communities
    communities = {}
    for node in result_builder.nodes:
        comm = node.lpa  # Updated attribute name
        communities[comm] = communities.get(comm, 0) + 1
    
    print(f"  Communities found: {len(communities)}")
    print(f"  Top 5 communities by size:")
    top_comms = sorted(communities.items(), key=lambda x: x[1], reverse=True)[:5]
    for comm_id, size in top_comms:
        print(f"    Community {comm_id}: {size} nodes")
    
    # Compare results
    print(f"\n  Comparison:")
    print(f"    Builder/Native time ratio: {builder_time/native_time:.2f}x")
    print(f"    Native communities: {len(communities_native)}")
    print(f"    Builder communities: {len(communities)}")
    
    # Note: Builder LPA implementation is incomplete (missing collect_neighbor_values + mode ops)
    # So we just compare performance, not correctness yet
    print(f"    ⚠️  Builder LPA implementation incomplete - performance comparison only")
    
    return builder_time, native_time, result_builder


def main():
    print("Builder vs Native Algorithm Benchmark")
    print("=" * 60)
    
    # Test on 50k graph
    print("\n\nBuilding 50k node graph...")
    graph_50k = create_test_graph(50000, avg_degree=10)
    
    pr_builder_50k, pr_native_50k, pr_result_50k = benchmark_pagerank(graph_50k, "50k nodes")
    lpa_builder_50k, lpa_native_50k, lpa_result_50k = benchmark_lpa(graph_50k, "50k nodes")
    
    # Test on 200k graph
    print("\n\nBuilding 200k node graph...")
    graph_200k = create_test_graph(200000, avg_degree=10)
    
    pr_builder_200k, pr_native_200k, pr_result_200k = benchmark_pagerank(graph_200k, "200k nodes")
    lpa_builder_200k, lpa_native_200k, lpa_result_200k = benchmark_lpa(graph_200k, "200k nodes")
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nPageRank Times (Builder vs Native):")
    print(f"  50k nodes:  {pr_builder_50k:.3f}s vs {pr_native_50k:.3f}s ({pr_builder_50k/pr_native_50k:.2f}x)")
    print(f"  200k nodes: {pr_builder_200k:.3f}s vs {pr_native_200k:.3f}s ({pr_builder_200k/pr_native_200k:.2f}x)")
    print(f"  Builder scaling: {pr_builder_200k/pr_builder_50k:.2f}x")
    print(f"  Native scaling:  {pr_native_200k/pr_native_50k:.2f}x")
    
    print(f"\nLPA Times (Builder vs Native):")
    print(f"  50k nodes:  {lpa_builder_50k:.3f}s vs {lpa_native_50k:.3f}s ({lpa_builder_50k/lpa_native_50k:.2f}x)")
    print(f"  200k nodes: {lpa_builder_200k:.3f}s vs {lpa_native_200k:.3f}s ({lpa_builder_200k/lpa_native_200k:.2f}x)")
    print(f"  Builder scaling: {lpa_builder_200k/lpa_builder_50k:.2f}x")
    print(f"  Native scaling:  {lpa_native_200k/lpa_native_50k:.2f}x")
    
    print(f"\n{'='*60}")
    print("✅ All benchmarks complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
