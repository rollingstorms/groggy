#!/usr/bin/env python3
"""
Detailed profiling demonstration for PageRank algorithm using builder primitives.

This script shows granular profiling capabilities including:
- Phase-by-phase timing with call counts
- Node/edge processing statistics  
- Memory allocation tracking
- Cache hit/miss statistics
- Per-step primitive timings

Run with: python profile_pr_detailed.py
"""

import os
import groggy as gr
from groggy.builder import AlgorithmBuilder
from groggy.pipeline import Pipeline


def build_pagerank_algorithm(damping=0.85, max_iter=20):
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


def _print_phase_table(title: str, rows, unit: str = "ms", scale: float = 1000.0):
    """Print a formatted table of timing data."""
    if not rows:
        return
    print(f"\n{title}")
    print("-" * 75)
    for name, value in rows:
        print(f"  {name:<40}{value * scale:>12.3f} {unit}")


def _print_call_counters(counters):
    """Print call counters with average times."""
    if not counters:
        return
    print("\nAlgorithm Call Counters")
    print("-" * 75)
    for name, data in counters:
        count = data.get("count", 0)
        total = data.get("total", 0.0)
        avg = data.get("avg", 0.0)
        print(
            f"  {name:<40}count={count:<8} total={total*1000:7.3f} ms  avg={avg*1e6:7.3f} µs"
        )


def _print_stats(stats):
    """Print algorithm statistics."""
    if not stats:
        return
    print("\nAlgorithm Statistics")
    print("-" * 75)
    for name, value in stats:
        if isinstance(value, float):
            print(f"  {name:<40}{value:>12.3f}")
        else:
            print(f"  {name:<40}{value:>12}")


def profile_pagerank_small():
    """Profile PageRank on a small directed graph."""
    print("=" * 80)
    print("PageRank - Small Graph Profiling (Directed)")
    print("=" * 80)
    
    g = gr.Graph(directed=True)
    
    # Create a simple directed graph: 0 -> 1 -> 2 -> 0 (cycle)
    #                                      |
    #                                      v
    #                                      3 (sink)
    nodes = g.add_nodes(4)
    edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0]), (nodes[1], nodes[3])]
    g.add_edges(edges)
    
    print(f"Graph created:")
    print(f"  Nodes: {len(list(g.nodes.ids()))}")
    print(f"  Edges: {len(list(g.edges.ids()))}")
    print(f"  Structure: 0→1→2→0 (cycle) with 1→3 (sink)")
    
    print("\nRunning PageRank algorithm...")
    algo = build_pagerank_algorithm(damping=0.85, max_iter=20)
    pipe = Pipeline([algo])
    result, profile = pipe(g.view(), persist=False, return_profile=True)
    
    # Print results
    print("\nPageRank scores:")
    for node in g.nodes:
        score = result.get_node_attribute(node.id, "pagerank")
        print(f"  Node {node.id}: {score:.6f}")
    
    # Print profiling data
    timers = profile.get("timers", {})
    _print_phase_table(
        "Algorithm Phase Timings",
        sorted(timers.items(), key=lambda item: item[1], reverse=True),
    )
    
    ffi_timers = profile.get("ffi_timers", {})
    _print_phase_table(
        "FFI Timings",
        sorted(ffi_timers.items(), key=lambda item: item[1], reverse=True),
    )
    
    call_counters = profile.get("call_counters", {})
    _print_call_counters(sorted(call_counters.items()))
    
    stats = profile.get("stats", {})
    _print_stats(sorted(stats.items()))


def profile_pagerank_medium():
    """Profile PageRank on a medium-sized random graph."""
    print("=" * 80)
    print("PageRank - Medium Graph Profiling (10k nodes)")
    print("=" * 80)
    
    import random
    random.seed(42)
    
    g = gr.Graph(directed=True)
    
    # Create nodes
    n = 10000
    nodes = g.add_nodes(n)
    
    # Create random edges (avg degree ~5)
    print(f"Building random graph with {n} nodes...")
    edges = []
    for i in range(n):
        num_edges = random.randint(3, 8)
        targets = random.sample(range(n), min(num_edges, n-1))
        for t in targets:
            if t != i:
                edges.append((nodes[i], nodes[t]))
    
    g.add_edges(edges)
    
    print(f"Graph created:")
    print(f"  Nodes: {len(list(g.nodes.ids())):,}")
    print(f"  Edges: {len(list(g.edges.ids())):,}")
    print(f"  Avg degree: {len(edges) / n:.2f}")
    
    print("\nRunning PageRank algorithm...")
    algo = build_pagerank_algorithm(damping=0.85, max_iter=20)
    pipe = Pipeline([algo])
    result, profile = pipe(g.view(), persist=False, return_profile=True)
    
    # Show top 10 nodes by PageRank
    print("\nTop 10 nodes by PageRank:")
    scores = [(node.id, result.get_node_attribute(node.id, "pagerank")) for node in g.nodes]
    scores.sort(key=lambda x: x[1], reverse=True)
    for node_id, score in scores[:10]:
        print(f"  Node {node_id}: {score:.6f}")
    
    # Print profiling data
    timers = profile.get("timers", {})
    _print_phase_table(
        "Algorithm Phase Timings",
        sorted(timers.items(), key=lambda item: item[1], reverse=True),
    )
    
    ffi_timers = profile.get("ffi_timers", {})
    _print_phase_table(
        "FFI Timings",
        sorted(ffi_timers.items(), key=lambda item: item[1], reverse=True),
    )
    
    call_counters = profile.get("call_counters", {})
    _print_call_counters(sorted(call_counters.items()))
    
    stats = profile.get("stats", {})
    _print_stats(sorted(stats.items()))


def profile_pagerank_large():
    """Profile PageRank on a large graph."""
    print("=" * 80)
    print("PageRank - Large Graph Profiling (100k nodes)")
    print("=" * 80)
    
    import random
    random.seed(42)
    
    g = gr.Graph(directed=True)
    
    # Create nodes
    n = 100000
    nodes = g.add_nodes(n)
    
    # Create random edges (avg degree ~8)
    print(f"Building random graph with {n:,} nodes...")
    edges = []
    for i in range(n):
        num_edges = random.randint(5, 12)
        targets = random.sample(range(n), min(num_edges, n-1))
        for t in targets:
            if t != i:
                edges.append((nodes[i], nodes[t]))
    
    g.add_edges(edges)
    
    print(f"Graph created:")
    print(f"  Nodes: {len(list(g.nodes.ids())):,}")
    print(f"  Edges: {len(list(g.edges.ids())):,}")
    print(f"  Avg degree: {len(edges) / n:.2f}")
    
    print("\nRunning PageRank algorithm...")
    algo = build_pagerank_algorithm(damping=0.85, max_iter=20)
    pipe = Pipeline([algo])
    result, profile = pipe(g.view(), persist=False, return_profile=True)
    
    # Show top 10 nodes by PageRank
    print("\nTop 10 nodes by PageRank:")
    scores = [(node.id, result.get_node_attribute(node.id, "pagerank")) for node in g.nodes]
    scores.sort(key=lambda x: x[1], reverse=True)
    for node_id, score in scores[:10]:
        print(f"  Node {node_id}: {score:.6f}")
    
    # Print profiling data
    timers = profile.get("timers", {})
    _print_phase_table(
        "Algorithm Phase Timings",
        sorted(timers.items(), key=lambda item: item[1], reverse=True),
    )
    
    ffi_timers = profile.get("ffi_timers", {})
    _print_phase_table(
        "FFI Timings",
        sorted(ffi_timers.items(), key=lambda item: item[1], reverse=True),
        unit="s",
        scale=1.0,
    )
    
    call_counters = profile.get("call_counters", {})
    _print_call_counters(sorted(call_counters.items()))
    
    stats = profile.get("stats", {})
    _print_stats(sorted(stats.items()))


def profile_native_comparison():
    """Compare builder PageRank with native implementation."""
    print("=" * 80)
    print("PageRank - Builder vs Native Comparison")
    print("=" * 80)
    
    import random
    random.seed(42)
    
    g = gr.Graph(directed=True)
    
    # Create medium-sized graph
    n = 50000
    nodes = g.add_nodes(n)
    
    edges = []
    for i in range(n):
        num_edges = random.randint(3, 8)
        targets = random.sample(range(n), min(num_edges, n-1))
        for t in targets:
            if t != i:
                edges.append((nodes[i], nodes[t]))
    
    g.add_edges(edges)
    
    print(f"Graph: {n:,} nodes, {len(edges):,} edges")
    
    # Run builder version
    print("\n--- Builder PageRank ---")
    builder_algo = build_pagerank_algorithm(damping=0.85, max_iter=20)
    builder_pipe = Pipeline([builder_algo])
    builder_result, builder_profile = builder_pipe(g.view(), persist=True, return_profile=True)
    
    builder_timers = builder_profile.get("timers", {})
    builder_total = sum(builder_timers.values())
    print(f"Total time: {builder_total*1000:.3f} ms")
    
    _print_phase_table(
        "Builder - Top 5 Phases",
        sorted(builder_timers.items(), key=lambda x: x[1], reverse=True)[:5],
    )
    
    # Run native version
    print("\n--- Native PageRank ---")
    from groggy.algorithms import centrality
    native_algo = centrality.pagerank(damping=0.85, max_iter=20, output_attr="native_pr")
    native_pipe = Pipeline([native_algo])
    native_result, native_profile = native_pipe(g.view(), persist=True, return_profile=True)
    
    native_timers = native_profile.get("timers", {})
    native_total = sum(native_timers.values())
    print(f"Total time: {native_total*1000:.3f} ms")
    
    _print_phase_table(
        "Native - Top 5 Phases",
        sorted(native_timers.items(), key=lambda x: x[1], reverse=True)[:5],
    )
    
    # Compare results
    print("\n--- Result Comparison ---")
    
    # Check if attributes exist
    builder_has_pr = "pagerank" in builder_result.nodes.attribute_names()
    native_has_pr = "native_pr" in native_result.nodes.attribute_names()
    print(f"Builder has 'pagerank': {builder_has_pr}")
    print(f"Native has 'native_pr': {native_has_pr}")
    
    if not builder_has_pr or not native_has_pr:
        print("Missing attributes - skipping comparison")
    else:
        diffs = []
        none_count = 0
        node_count = 0
        for node in g.nodes:
            node_count += 1
            builder_val = builder_result.get_node_attribute(node.id, "pagerank")
            native_val = native_result.get_node_attribute(node.id, "native_pr")
            if builder_val is not None and native_val is not None:
                diffs.append(abs(builder_val - native_val))
            else:
                none_count += 1
                if node_count <= 3:  # Debug first few
                    print(f"  Debug: Node {node.id} - builder={builder_val}, native={native_val}")
        
        print(f"Processed {node_count} nodes, {len(diffs)} valid, {none_count} None")
        
        if diffs:
            print(f"Max difference: {max(diffs):.10f}")
            print(f"Avg difference: {sum(diffs)/len(diffs):.10f}")
        else:
            print("No valid comparisons could be made!")
    
    print(f"Speedup: {builder_total/native_total:.2f}x {'(builder slower)' if builder_total > native_total else '(builder faster)'}")


if __name__ == '__main__':
    try:
        # Run profiling demonstrations
        profile_pagerank_small()
        print("\n")
        profile_pagerank_medium()
        print("\n")
        profile_pagerank_large()
        print("\n")
        profile_native_comparison()
        
        print("\n" + "=" * 80)
        print("Profiling demonstration completed successfully!")
        print("\nKey insights:")
        print("- Identify bottlenecks in primitive steps")
        print("- FFI overhead analysis")
        print("- Call counts reveal iteration behavior")
        print("- Compare builder vs native performance")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
