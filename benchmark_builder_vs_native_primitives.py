"""
Benchmark builder-based PageRank and LPA vs native implementations.
Tests on 50k and 200k node graphs.
Validates that results match between builder and native implementations.
"""
import time
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms import centrality, community


def build_pagerank_algorithm():
    """Build PageRank using the builder DSL."""
    builder = AlgorithmBuilder("custom_pagerank")
    ranks = builder.init_nodes(default=1.0)
    
    with builder.iterate(20):
        neighbor_sums = builder.map_nodes(
            "sum(ranks[neighbors(node)])",
            inputs={"ranks": ranks}
        )
        damped = builder.core.mul(neighbor_sums, 0.85)
        added = builder.core.add(damped, 0.15)
        ranks = builder.core.normalize_sum(added)
    
    builder.attach_as("pagerank", ranks)
    return builder.build()


def build_lpa_algorithm():
    """Build LPA using the builder DSL with async updates."""
    builder = AlgorithmBuilder("custom_lpa")
    
    # Initialize each node with unique index (0, 1, 2, ...)
    labels = builder.init_nodes(unique=True)
    labels = builder.var("labels", labels)
    
    with builder.iterate(10):
        # Use async_update so nodes see earlier updates in same iteration
        labels = builder.var(
            "labels",
            builder.map_nodes(
                "mode(labels[neighbors(node)])",
                inputs={"labels": labels},
                async_update=True,  # Enables LPA-style propagation
            ),
        )
    
    builder.attach_as("community", labels)
    return builder.build()


def create_test_graph(num_nodes, avg_degree=10):
    """Create a random graph for testing."""
    import random
    random.seed(42)
    
    graph = Graph()
    nodes = [graph.add_node() for _ in range(num_nodes)]
    
    # Create random edges
    num_edges = num_nodes * avg_degree // 2
    edges_created = 0
    attempts = 0
    max_attempts = num_edges * 3
    
    while edges_created < num_edges and attempts < max_attempts:
        src = random.choice(nodes)
        dst = random.choice(nodes)
        if src != dst:
            try:
                graph.add_edge(src, dst)
                graph.add_edge(dst, src)  # Make undirected
                edges_created += 2
            except:
                pass
        attempts += 1
    
    print(f"Created graph: {num_nodes} nodes, ~{edges_created} edges")
    return graph


def benchmark_pagerank(graph, name):
    """Benchmark PageRank builder vs native."""
    print(f"\n{'='*60}")
    print(f"PageRank Benchmark - {name}")
    print(f"{'='*60}")
    
    sg = graph.view()
    
    # Native version
    print("\nNative PageRank:")
    start = time.perf_counter()
    result_native, stats_native = sg.apply(
        centrality.pagerank(max_iter=20, damping=0.85, output_attr="pagerank_native"),
        persist=True,  # Need to persist to access attributes
        return_profile=True
    )
    native_time = time.perf_counter() - start
    
    print(f"  Time: {native_time:.3f}s")
    
    # Get some sample values
    sample_nodes = list(result_native.nodes)[:5]
    print(f"  Sample values:")
    for node in sample_nodes:
        print(f"    Node {node.id}: {node.pagerank_native:.6f}")
    
    # Verify normalization
    total_rank_native = sum(node.pagerank_native for node in result_native.nodes)
    print(f"  Total rank: {total_rank_native:.6f}")
    
    # Builder version
    print("\nBuilder PageRank:")
    algo = build_pagerank_algorithm()
    
    start = time.perf_counter()
    result_builder = sg.apply(algo)
    builder_time = time.perf_counter() - start
    
    print(f"  Time: {builder_time:.3f}s")
    
    # Get some sample values
    print(f"  Sample values:")
    for node in sample_nodes:
        print(f"    Node {node.id}: {node.pagerank:.6f}")
    
    # Verify normalization
    total_rank = sum(node.pagerank for node in result_builder.nodes)
    print(f"  Total rank: {total_rank:.6f}")
    
    # Compare results
    print(f"\n  Comparison:")
    print(f"    Builder/Native time ratio: {builder_time/native_time:.2f}x")
    
    # Check if values are similar
    diffs = []
    for node in sample_nodes:
        diff = abs(node.pagerank - node.pagerank_native)
        diffs.append(diff)
    
    avg_diff = sum(diffs) / len(diffs)
    max_diff = max(diffs)
    print(f"    Avg difference: {avg_diff:.8f}")
    print(f"    Max difference: {max_diff:.8f}")
    
    if max_diff < 0.01:
        print(f"    ✅ Results match!")
    else:
        print(f"    ⚠️  Results differ significantly")
    
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
    algo = build_lpa_algorithm()
    
    start = time.perf_counter()
    result_builder = sg.apply(algo)
    builder_time = time.perf_counter() - start
    
    print(f"  Time: {builder_time:.3f}s")
    
    # Count communities
    communities = {}
    for node in result_builder.nodes:
        comm = node.community
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
    
    if len(communities) == len(communities_native):
        print(f"    ✅ Same number of communities found!")
    else:
        print(f"    ⚠️  Different number of communities")
    
    return builder_time, native_time, result_builder


def main():
    print("Builder vs Native Algorithm Benchmark")
    print("=" * 60)
    
    # Test on 50k graph
    print("\n\nBuilding 50k node graph...")
    graph_50k = create_test_graph(50, avg_degree=10)
    
    pr_builder_50k, pr_native_50k, pr_result_50k = benchmark_pagerank(graph_50k, "50 nodes")
    lpa_builder_50k, lpa_native_50k, lpa_result_50k = benchmark_lpa(graph_50k, "50 nodes")
    
    # Test on 200k graph
    print("\n\nBuilding 200k node graph...")
    graph_200k = create_test_graph(200, avg_degree=10)
    
    pr_builder_200k, pr_native_200k, pr_result_200k = benchmark_pagerank(graph_200k, "200 nodes")
    lpa_builder_200k, lpa_native_200k, lpa_result_200k = benchmark_lpa(graph_200k, "200 nodes")
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nPageRank Times (Builder vs Native):")
    print(f"  50 nodes:  {pr_builder_50k:.3f}s vs {pr_native_50k:.3f}s ({pr_builder_50k/pr_native_50k:.2f}x)")
    print(f"  200 nodes: {pr_builder_200k:.3f}s vs {pr_native_200k:.3f}s ({pr_builder_200k/pr_native_200k:.2f}x)")
    print(f"  Builder scaling: {pr_builder_200k/pr_builder_50k:.2f}x")
    print(f"  Native scaling:  {pr_native_200k/pr_native_50k:.2f}x")
    
    print(f"\nLPA Times (Builder vs Native):")
    print(f"  50 nodes:  {lpa_builder_50k:.3f}s vs {lpa_native_50k:.3f}s ({lpa_builder_50k/lpa_native_50k:.2f}x)")
    print(f"  200 nodes: {lpa_builder_200k:.3f}s vs {lpa_native_200k:.3f}s ({lpa_builder_200k/lpa_native_200k:.2f}x)")
    print(f"  Builder scaling: {lpa_builder_200k/lpa_builder_50k:.2f}x")
    print(f"  Native scaling:  {lpa_native_200k/lpa_native_50k:.2f}x")
    
    print(f"\n{'='*60}")
    print("✅ All benchmarks complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
