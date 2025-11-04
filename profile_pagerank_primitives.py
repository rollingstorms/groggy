#!/usr/bin/env python3
"""
Detailed profiling for primitive-based PageRank algorithm.

This script profiles the PageRank implementation built from core primitives,
showing where time is spent across the primitive operations:
- node_degree computation
- arithmetic operations (mul, add, recip)
- neighbor aggregation
- normalization
- scalar reductions

The goal is to identify bottlenecks in the primitive composition approach
and guide optimization efforts.
"""

import os
import time
import groggy as gr
from groggy.builder import AlgorithmBuilder
from groggy.pipeline import Pipeline


def build_pagerank_primitive(damping=0.85, max_iter=20, tolerance=1e-6):
    """Build PageRank using core primitives."""
    builder = AlgorithmBuilder("pagerank_primitive")
    
    # Initialize ranks uniformly
    n = builder.graph_node_count()
    ranks = builder.init_nodes(default=1.0)
    ranks = builder.core.normalize_values(ranks, method="sum")
    
    # Compute out-degrees (once, outside loop)
    degrees = builder.node_degrees()
    
    # Prep for safe division: use recip with epsilon
    inv_deg = builder.core.recip(degrees, epsilon=1e-12)
    
    # Identify sink nodes (out-degree == 0)
    zero_mask = builder.core.compare(degrees, op="eq", rhs=0.0)
    
    # Power iteration loop
    with builder.iterate(max_iter):
        # Compute sink mass (nodes with no outgoing edges)
        sink_ranks = builder.core.where(zero_mask, ranks, 0.0)
        sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
        
        # Weight each node's rank by its inverse out-degree
        contrib = builder.core.mul(ranks, inv_deg)
        
        # Aggregate weighted contributions from neighbors
        neighbor_sum = builder.neighbor_agg(values=contrib, agg="sum")
        
        # Apply damping to neighbor contributions
        damped_neighbors = builder.core.mul_scalar(neighbor_sum, damping)
        
        # Redistribute sink mass evenly
        sink_per_node = builder.core.broadcast_scalar(sink_mass, scale=damping)
        sink_per_node = builder.core.div_scalar(sink_per_node, n)
        
        # Add sink contribution
        ranks = builder.core.add(damped_neighbors, sink_per_node)
        
        # Add teleport term
        teleport = builder.core.constant_scalar((1.0 - damping))
        teleport = builder.core.div_scalar(teleport, n)
        ranks = builder.core.add_scalar(ranks, teleport)
        
        # Re-normalize to maintain probability distribution
        ranks = builder.core.normalize_values(ranks, method="sum")
    
    builder.attach_as("pagerank", ranks)
    return builder.build()


def _print_phase_table(title: str, rows, unit: str = "ms", scale: float = 1000.0):
    if not rows:
        return
    print(f"\n{title}")
    print("-" * 75)
    total = sum(v for _, v in rows)
    for name, value in rows:
        pct = (value / total * 100) if total > 0 else 0
        print(f"  {name:<40}{value * scale:>12.3f} {unit}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<40}{total * scale:>12.3f} {unit}")


def _print_call_counters(counters):
    if not counters:
        return
    print("\nPrimitive Call Counters")
    print("-" * 75)
    for name, data in counters:
        count = data.get("count", 0)
        total = data.get("total", 0.0)
        avg = data.get("avg", 0.0)
        print(
            f"  {name:<40}count={count:<8} total={total*1000:7.3f} ms  avg={avg*1e6:7.3f} µs"
        )


def _print_stats(stats):
    if not stats:
        return
    print("\nAlgorithm Statistics")
    print("-" * 75)
    for name, value in stats:
        print(f"  {name:<40}{value:>12.3f}")


def identify_bottlenecks():
    """Run focused profiling to identify the main bottlenecks."""
    print("=" * 80)
    print("PageRank Bottleneck Identification - 20k Node Graph")
    print("=" * 80)
    
    # Medium-sized graph
    g = gr.Graph(directed=True)
    nodes = g.add_nodes(20000)
    
    import random
    random.seed(42)
    edge_data = []
    for i in range(20000):
        # Each node connects to ~5 others
        n_edges = random.randint(1, 10)
        for _ in range(n_edges):
            target = random.randint(0, 19999)
            if target != i:
                edge_data.append((nodes[i], nodes[target]))
    
    g.add_edges(edge_data)
    
    print(f"Graph: {len(nodes):,} nodes, {len(edge_data):,} edges")
    
    algo = build_pagerank_primitive(max_iter=15)
    pipe = Pipeline([algo])
    
    print("\nRunning PageRank (15 iterations)...")
    start = time.perf_counter()
    result, profile = pipe(g.view(), persist=False, return_profile=True)
    elapsed = time.perf_counter() - start
    
    print(f"Total time: {elapsed*1000:.3f} ms")
    
    # Analyze where time is spent
    timers = profile.get("timers", {})
    if timers:
        total_time = sum(timers.values())
        print(f"\n{'='*80}")
        print("TOP 10 BOTTLENECKS (by time spent)")
        print('='*80)
        
        sorted_timers = sorted(timers.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (name, t) in enumerate(sorted_timers, 1):
            pct = (t / total_time * 100) if total_time > 0 else 0
            print(f"{i:2d}. {name:<50} {t*1000:8.3f} ms ({pct:5.1f}%)")
    
    call_counters = profile.get("call_counters", {})
    if call_counters:
        print(f"\n{'='*80}")
        print("MOST CALLED PRIMITIVES")
        print('='*80)
        
        sorted_calls = sorted(call_counters.items(), key=lambda x: x[1].get("count", 0), reverse=True)[:10]
        for i, (name, data) in enumerate(sorted_calls, 1):
            count = data.get("count", 0)
            avg = data.get("avg", 0.0)
            total = data.get("total", 0.0)
            print(f"{i:2d}. {name:<40} count={count:>6}  avg={avg*1e6:7.2f} µs  total={total*1000:7.2f} ms")
    
    # FFI overhead analysis
    ffi_timers = profile.get("ffi_timers", {})
    if ffi_timers:
        ffi_total = sum(ffi_timers.values())
        rust_total = sum(timers.values())
        overhead_pct = (ffi_total / (rust_total + ffi_total) * 100) if (rust_total + ffi_total) > 0 else 0
        
        print(f"\n{'='*80}")
        print("FFI OVERHEAD ANALYSIS")
        print('='*80)
        print(f"  Rust execution time: {rust_total*1000:.3f} ms")
        print(f"  FFI overhead time:   {ffi_total*1000:.3f} ms")
        print(f"  FFI overhead:        {overhead_pct:.1f}% of total")
    
    # Validate results
    pr_values = result.nodes["pagerank"]
    print(f"\nPageRank validation:")
    print(f"  Sum: {sum(pr_values):.6f} (should be ~1.0)")
    print(f"  Min: {min(pr_values):.6f}")
    print(f"  Max: {max(pr_values):.6f}")


def profile_primitive_breakdown():
    """Profile with detailed primitive category breakdown."""
    print("=" * 80)
    print("PageRank Primitive Category Breakdown - 10k Node Graph")
    print("=" * 80)
    
    # Create test graph
    g = gr.Graph(directed=True)
    nodes = g.add_nodes(10000)
    
    # Random graph
    import random
    random.seed(42)
    edge_data = []
    for i in range(10000):
        n_edges = random.randint(1, 8)
        for _ in range(n_edges):
            target = random.randint(0, 9999)
            if target != i:
                edge_data.append((nodes[i], nodes[target]))
    
    g.add_edges(edge_data)
    
    print(f"Graph: {len(nodes):,} nodes, {len(edge_data):,} edges")
    
    # Build algorithm with moderate iterations
    algo = build_pagerank_primitive(damping=0.85, max_iter=10)
    pipe = Pipeline([algo])
    
    print("\nRunning PageRank (10 iterations)...")
    result, profile = pipe(g.view(), persist=False, return_profile=True)
    
    # Detailed breakdown by primitive category
    print("\n" + "="*80)
    print("PRIMITIVE OPERATION BREAKDOWN BY CATEGORY")
    print("="*80)
    
    timers = profile.get("timers", {})
    if timers:
        # Categorize primitives
        categories = {
            "Initialization": [],
            "Degree Computation": [],
            "Arithmetic (mul/add/div/recip)": [],
            "Neighbor Aggregation": [],
            "Normalization": [],
            "Comparison/Masking": [],
            "Scalar Operations": [],
            "Other": []
        }
        
        for name, value in timers.items():
            name_lower = name.lower()
            if "init" in name_lower:
                categories["Initialization"].append((name, value))
            elif "degree" in name_lower:
                categories["Degree Computation"].append((name, value))
            elif any(x in name_lower for x in ["mul", "add", "div", "recip", "sub"]):
                categories["Arithmetic (mul/add/div/recip)"].append((name, value))
            elif "agg" in name_lower or "neighbor" in name_lower:
                categories["Neighbor Aggregation"].append((name, value))
            elif "norm" in name_lower:
                categories["Normalization"].append((name, value))
            elif "compare" in name_lower or "where" in name_lower:
                categories["Comparison/Masking"].append((name, value))
            elif "scalar" in name_lower or "reduce" in name_lower or "broadcast" in name_lower:
                categories["Scalar Operations"].append((name, value))
            else:
                categories["Other"].append((name, value))
        
        # Print each category
        for cat_name, ops in categories.items():
            if ops:
                _print_phase_table(cat_name, sorted(ops, key=lambda x: x[1], reverse=True))
    
    ffi_timers = profile.get("ffi_timers", {})
    if ffi_timers:
        _print_phase_table(
            "FFI Overhead",
            sorted(ffi_timers.items(), key=lambda item: item[1], reverse=True),
        )


def profile_scaling():
    """Profile across different graph sizes."""
    print("=" * 80)
    print("PageRank Scaling Analysis")
    print("=" * 80)
    
    sizes = [1000, 5000, 10000, 20000]
    results = []
    
    for n_nodes in sizes:
        print(f"\nTesting {n_nodes:,} nodes...")
        
        g = gr.Graph(directed=True)
        nodes = g.add_nodes(n_nodes)
        
        import random
        random.seed(42)
        edge_data = []
        for i in range(n_nodes):
            n_edges = random.randint(2, 8)
            for _ in range(n_edges):
                target = random.randint(0, n_nodes - 1)
                if target != i:
                    edge_data.append((nodes[i], nodes[target]))
        
        g.add_edges(edge_data)
        
        algo = build_pagerank_primitive(max_iter=10)
        pipe = Pipeline([algo])
        
        start = time.perf_counter()
        result, profile = pipe(g.view(), persist=False, return_profile=True)
        elapsed = time.perf_counter() - start
        
        timers = profile.get("timers", {})
        total_rust = sum(timers.values()) if timers else 0
        
        ffi_timers = profile.get("ffi_timers", {})
        total_ffi = sum(ffi_timers.values()) if ffi_timers else 0
        
        results.append({
            'nodes': n_nodes,
            'edges': len(edge_data),
            'total': elapsed,
            'rust': total_rust,
            'ffi': total_ffi
        })
    
    # Print scaling summary
    print("\n" + "="*80)
    print("SCALING SUMMARY")
    print("="*80)
    print(f"{'Nodes':>8} {'Edges':>10} {'Total (ms)':>12} {'Rust (ms)':>12} {'FFI (ms)':>12} {'FFI %':>8}")
    print("-" * 80)
    for r in results:
        ffi_pct = (r['ffi'] / r['total'] * 100) if r['total'] > 0 else 0
        print(f"{r['nodes']:>8,} {r['edges']:>10,} {r['total']*1000:>12.2f} {r['rust']*1000:>12.2f} {r['ffi']*1000:>12.2f} {ffi_pct:>7.1f}%")


if __name__ == '__main__':
    try:
        # Enable detailed profiling
        os.environ['GROGGY_PROFILE_PRIMITIVES'] = '1'
        
        print("\n" + "="*80)
        print("PAGERANK PRIMITIVE-BASED PROFILING SUITE")
        print("="*80 + "\n")
        
        # Run profiling suites
        identify_bottlenecks()
        print("\n\n")
        
        profile_primitive_breakdown()
        print("\n\n")
        
        profile_scaling()
        
        print("\n" + "="*80)
        print("PROFILING COMPLETE")
        print("="*80)
        print("\nKey insights to investigate:")
        print("1. Neighbor aggregation - typically the slowest primitive")
        print("2. FFI overhead - measure Python↔Rust crossing cost")
        print("3. Primitive call frequency - can we batch operations?")
        print("4. Normalization cost - is it needed every iteration?")
        print("5. Arithmetic fusion opportunities")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
