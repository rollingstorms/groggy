"""
Detailed profiling of PageRank builder primitives.
Mirrors profile_cc_detailed.py structure to identify systemic bottlenecks.
"""

import time
import groggy as gg
from groggy.builder import AlgorithmBuilder


def build_pagerank_algorithm(damping=0.85, max_iter=20):
    """Build PageRank using the builder DSL with proper primitives."""
    builder = AlgorithmBuilder("pagerank_profiling")
    
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
        
        # New rank = damped_neighbors + teleport + sink_contribution
        ranks = builder.core.add(damped_neighbors, teleport_map)
        
        # Normalize (optional, but helps numerical stability)
        ranks = builder.normalize(ranks, method="sum")
    
    builder.attach_as("pagerank", ranks)
    return builder.build()


def profile_graph(graph, name, damping=0.85, max_iter=20):
    """Profile PageRank on a specific graph."""
    print(f"\n{'='*80}")
    print(f"Profiling: {name}")
    print(f"Nodes: {len(list(graph.nodes)):,}, Edges: {len(list(graph.edges)):,}")
    print(f"{'='*80}\n")
    
    # Build algorithm
    print("Building algorithm...")
    algo = build_pagerank_algorithm(damping=damping, max_iter=max_iter)
    
    # Execute with profiling and manual timing
    print("Executing with profiling enabled...")
    t_start = time.perf_counter()
    result, profile_data = graph.view().apply(algo, return_profile=True)
    t_total = time.perf_counter() - t_start
    
    print(f"\n{'─'*80}")
    print(f"Total execution time: {t_total*1000:.2f} ms")
    print(f"{'─'*80}\n")
    
    # Analyze profiling data
    if profile_data:
        analyze_profiling_data(profile_data, t_total)
    else:
        print("⚠️  No profiling data returned")
    
    return result, t_total


def analyze_profiling_data(profile_data, total_time):
    """Analyze and display profiling data in detail."""
    if not profile_data or 'timers' not in profile_data:
        print("No profiling data captured")
        return
    
    timers = profile_data.get('timers', {})
    
    print("\nPer-Step Timing Breakdown:")
    print(f"{'─'*80}")
    print(f"{'Step':<50} {'Time (ms)':<12} {'% Total':<10}")
    print(f"{'─'*80}")
    
    # Sort by time descending
    sorted_steps = sorted(
        timers.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for step_name, step_time in sorted_steps:
        pct = (step_time / total_time * 100) if total_time > 0 else 0
        
        print(f"{step_name:<50} {step_time*1000:<12.3f} {pct:<10.1f}")
    
    print(f"{'─'*80}\n")
    
    # Identify bottlenecks
    print("\nBottleneck Analysis:")
    print(f"{'─'*80}")
    
    total_accounted = sum(timers.values())
    overhead = total_time - total_accounted
    
    # Group by operation type
    groups = {
        'Initialization': [],
        'Graph Ops': [],
        'Arithmetic': [],
        'Aggregation': [],
        'Normalization': [],
        'Other': []
    }
    
    for step_name, step_time in timers.items():
        
        if 'init' in step_name.lower():
            groups['Initialization'].append((step_name, step_time))
        elif 'degree' in step_name.lower() or 'neighbor' in step_name.lower():
            groups['Graph Ops'].append((step_name, step_time))
        elif any(op in step_name.lower() for op in ['add', 'mul', 'recip', 'clip', 'compare', 'where']):
            groups['Arithmetic'].append((step_name, step_time))
        elif 'agg' in step_name.lower() or 'reduce' in step_name.lower():
            groups['Aggregation'].append((step_name, step_time))
        elif 'normalize' in step_name.lower():
            groups['Normalization'].append((step_name, step_time))
        else:
            groups['Other'].append((step_name, step_time))
    
    for group_name, steps in groups.items():
        if steps:
            group_total = sum(t for _, t in steps)
            pct = (group_total / total_time * 100) if total_time > 0 else 0
            print(f"\n{group_name}: {group_total*1000:.2f} ms ({pct:.1f}%)")
            for step_name, step_time in sorted(steps, key=lambda x: x[1], reverse=True):
                step_pct = (step_time / total_time * 100) if total_time > 0 else 0
                print(f"  • {step_name}: {step_time*1000:.2f} ms ({step_pct:.1f}%)")
    
    print(f"\nFFI Overhead (unaccounted): {overhead*1000:.2f} ms ({overhead/total_time*100:.1f}%)")
    print(f"{'─'*80}\n")


def main():
    """Run profiling on various graph sizes."""
    print("PageRank Primitives Profiling")
    print("=" * 80)
    
    test_cases = [
        # Small test
        ("Small (50 nodes, sparse)", lambda: gg.generators.barabasi_albert(50, 2)),
        
        # Medium test
        ("Medium (1K nodes)", lambda: gg.generators.barabasi_albert(1000, 3)),
        
        # Large test
        ("Large (10K nodes)", lambda: gg.generators.barabasi_albert(10000, 3)),
        
        # Very large test
        ("Very Large (50K nodes)", lambda: gg.generators.barabasi_albert(50000, 3)),
    ]
    
    results = []
    
    for name, graph_fn in test_cases:
        try:
            graph = graph_fn()
            result, t_total = profile_graph(graph, name, max_iter=20)
            results.append((name, len(list(graph.nodes)), len(list(graph.edges)), t_total))
        except Exception as e:
            print(f"❌ Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Test Case':<30} {'Nodes':>10} {'Edges':>10} {'Time (ms)':>12}")
    print("-"*80)
    for name, nodes, edges, t_total in results:
        print(f"{name:<30} {nodes:>10,} {edges:>10,} {t_total*1000:>12.2f}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("""
Based on the profiling data, the main bottlenecks are likely:

1. FFI Overhead
   - Each primitive step crosses the Python/Rust boundary
   - Small operations (add, mul) pay disproportionate FFI cost
   - Solution: Batch operations or create composite steps

2. Neighbor Aggregation (core.neighbor_agg)
   - CSR construction or lookup overhead
   - Per-node iteration cost
   - Solution: Optimize CSR caching, use bulk operations

3. Repeated Operations in Loops
   - Each iteration pays full FFI cost
   - Temporary allocations
   - Solution: Move loop logic to Rust, or fuse operations

4. Normalization
   - Two-pass operation (sum + divide)
   - Could be fused with other operations
   - Solution: Fused normalize+add or single-pass normalize

Next steps:
- Identify top 3 bottlenecks from profiling data
- Optimize those specific primitives
- Consider fusing hot paths (e.g., damped_sum_normalize)
- Measure impact after each optimization
""")


if __name__ == "__main__":
    main()
