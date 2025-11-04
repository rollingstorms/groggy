#!/usr/bin/env python3
"""
Detailed profiling of builder-based PageRank vs native PageRank.
Focus on identifying systemic bottlenecks in the primitives approach.

Based on the analysis in benchmark_builder_vs_native.py, we target:
- neighbor_agg (the main bottleneck at ~90% of time)
- FFI overhead in core operations
"""

import groggy as gg
from groggy.builder import AlgorithmBuilder
from groggy.algorithms import centrality
import time
import statistics

def build_pagerank_algorithm(damping=0.85, max_iter=100):
    """Build PageRank using primitives (matching native implementation)."""
    builder = AlgorithmBuilder("pagerank_primitives")
    
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


def profile_step_times(graph, algo, num_runs=5):
    """Run algorithm with profiling enabled and collect step times."""
    times = []
    sg = graph.view()
    
    for _ in range(num_runs):
        _, profile = sg.apply(algo, return_profile=True)
        times.append(profile)
    
    return times


def analyze_profile_data(profile_runs):
    """Aggregate and analyze profiling data from multiple runs."""
    if not profile_runs:
        return {}
    
    # Extract timer data from profiles
    timer_runs = [p.get('timers', {}) for p in profile_runs if 'timers' in p]
    if not timer_runs:
        return {}
    
    # Collect all step names across all runs
    all_step_names = set()
    for timers in timer_runs:
        all_step_names.update(timers.keys())
    
    analysis = {}
    for step_name in all_step_names:
        step_times = [t[step_name] for t in timer_runs if step_name in t]
        
        if step_times:
            analysis[step_name] = {
                'mean': statistics.mean(step_times),
                'median': statistics.median(step_times),
                'min': min(step_times),
                'max': max(step_times),
                'stdev': statistics.stdev(step_times) if len(step_times) > 1 else 0,
                'total': sum(step_times),
            }
    
    return analysis


def run_profiling_comparison():
    """Compare builder vs native PageRank with detailed profiling."""
    
    # Test on different graph sizes
    sizes = [100, 1000, 5000, 10000]
    
    for n in sizes:
        print(f"\n{'='*80}")
        print(f"Graph size: {n} nodes")
        print('='*80)
        
        # Create test graph (undirected, matching benchmark)
        import random
        random.seed(42)
        
        g = gg.Graph()  # Undirected by default
        
        # Create nodes explicitly
        nodes = []
        for i in range(n):
            nodes.append(g.add_node())
        
        # Generate edges efficiently
        avg_degree = 10
        num_edges = n * avg_degree // 2
        edges = []
        seen = set()
        
        for _ in range(num_edges * 2):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                edge = (min(i, j), max(i, j))
                if edge not in seen:
                    edges.append((nodes[i], nodes[j]))
                    seen.add(edge)
                    if len(edges) >= num_edges:
                        break
        
        g.add_edges(edges)
        
        print(f"Graph: {n} nodes, {len(edges)} edges")
        
        # Profile native PageRank
        print("\n--- Native PageRank ---")
        native_algo = centrality.pagerank(damping=0.85, max_iter=100, tolerance=1e-6)
        native_times = []
        for _ in range(5):
            start = time.perf_counter()
            native_result = g.view().apply(native_algo)
            elapsed = time.perf_counter() - start
            native_times.append(elapsed)
        
        print(f"Mean time: {statistics.mean(native_times)*1000:.2f} ms")
        print(f"Median time: {statistics.median(native_times)*1000:.2f} ms")
        
        # Profile builder PageRank
        print("\n--- Builder PageRank (with profiling) ---")
        algo = build_pagerank_algorithm(damping=0.85, max_iter=100)
        profile_runs = profile_step_times(g, algo, num_runs=5)
        
        # Analyze step-by-step timings
        analysis = analyze_profile_data(profile_runs)
        
        # Sort by total time (across all runs)
        sorted_steps = sorted(analysis.items(), key=lambda x: x[1]['total'], reverse=True)
        
        print("\nTop 10 bottlenecks (by total time across 5 runs):")
        print(f"{'Step':<40} {'Mean (ms)':<12} {'Total (ms)':<12} {'% of Total':<10}")
        print('-'*80)
        
        total_time = sum(s[1]['total'] for s in sorted_steps)
        
        for step_name, stats in sorted_steps[:10]:
            pct = (stats['total'] / total_time * 100) if total_time > 0 else 0
            print(f"{step_name:<40} {stats['mean']*1000:>10.2f}  {stats['total']*1000:>10.2f}  {pct:>8.1f}%")
        
        # Compare total builder time vs native
        builder_mean = total_time / 5  # Average across runs
        native_mean = statistics.mean(native_times)
        
        print(f"\n--- Comparison ---")
        print(f"Native mean:  {native_mean*1000:>10.2f} ms")
        print(f"Builder mean: {builder_mean*1000:>10.2f} ms")
        print(f"Overhead:     {(builder_mean/native_mean - 1)*100:>10.1f}%")
        
        # Verify correctness
        builder_result, _ = g.view().apply(algo, return_profile=False)
        native_pr = {node.id: node.pagerank for node in native_result.nodes}
        builder_pr = {node.id: node.pagerank for node in builder_result.nodes}
        
        diffs = [abs(native_pr.get(i, 0) - builder_pr.get(i, 0)) for i in range(n)]
        max_diff = max(diffs)
        avg_diff = statistics.mean(diffs)
        
        print(f"\n--- Correctness ---")
        print(f"Max diff:  {max_diff:.2e}")
        print(f"Avg diff:  {avg_diff:.2e}")
        
        if max_diff > 1e-4:
            print("⚠️  Results differ significantly!")
        else:
            print("✓  Results match within tolerance")


if __name__ == "__main__":
    run_profiling_comparison()
