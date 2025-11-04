#!/usr/bin/env python3
"""Simple, fast profiling of connected components."""

import time
import groggy as gg

def create_fast_graph(num_nodes, num_edges):
    """Create graph efficiently using batch operations."""
    g = gg.Graph()
    
    # Add all nodes at once
    nodes = g.add_nodes(num_nodes)
    
    # Create edge list efficiently
    import random
    edges = []
    for _ in range(num_edges):
        src = random.randint(0, num_nodes - 1)
        tgt = random.randint(0, num_nodes - 1)
        if src != tgt:
            edges.append((nodes[src], nodes[tgt]))
    
    # Add edges in batch
    g.add_edges(edges)
    
    return g

def benchmark(name, num_nodes, num_edges, trials=3):
    """Run benchmark."""
    print(f"\n{name}: {num_nodes:,} nodes, {num_edges:,} edges")
    
    # Create graph once
    print("  Creating graph...", end=" ", flush=True)
    t0 = time.perf_counter()
    g = create_fast_graph(num_nodes, num_edges)
    create_time = time.perf_counter() - t0
    print(f"{create_time:.2f}s")
    
    # Warm-up
    _ = g.connected_components()
    
    # Time trials
    times = []
    for i in range(trials):
        start = time.perf_counter()
        result = g.connected_components()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Trial {i+1}: {elapsed:.4f}s")
    
    avg = sum(times) / len(times)
    print(f"  → Average: {avg:.4f}s ({num_edges/avg:,.0f} edges/sec)")
    return avg

if __name__ == "__main__":
    print("="*60)
    print("Connected Components: Union-Find (Optimized)")
    print("="*60)
    
    # Small
    benchmark("Small", 1_000, 5_000, trials=3)
    
    # Medium
    benchmark("Medium", 10_000, 50_000, trials=3)
    
    # Large
    benchmark("Large", 100_000, 300_000, trials=2)
    
    # Critical - 200k/600k
    print("\n" + "="*60)
    print("*** CRITICAL TEST ***")
    print("="*60)
    time_200k = benchmark("200k/600k", 200_000, 600_000, trials=2)
    
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    print(f"200k/600k time: {time_200k:.4f}s")
    if time_200k < 0.20:
        print("✓ EXCELLENT: Much faster than 0.33s (old) and 0.15s (BFS)")
    elif time_200k < 0.25:
        print("✓ GOOD: Competitive with 0.15s BFS baseline")
    elif time_200k < 0.35:
        print("~ OK: Better than 0.33s regression but slower than 0.15s BFS")
    else:
        print("✗ SLOW: Still slower than both baselines")
