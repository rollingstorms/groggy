#!/usr/bin/env python3
"""
Simple benchmark to measure Connected Components optimization.
Tests the Union-Find optimization vs previous TraversalEngine approach.
"""

import sys
import time
import gc
import statistics
from typing import List, Tuple

# Use local groggy
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr
from groggy.algorithms import community

def create_test_graph(n_nodes: int, avg_degree: int = 3) -> Tuple[List[int], List[Tuple[int, int, float]]]:
    """Create a random graph for testing."""
    import random
    
    nodes = list(range(n_nodes))
    edges = []
    
    # Create a connected graph using spanning tree
    for i in range(1, n_nodes):
        parent = random.randint(0, i-1)
        edges.append((i, parent, 1.0))
        edges.append((parent, i, 1.0))  # Make it undirected
    
    # Add extra edges to reach target degree
    n_extra = (n_nodes * avg_degree) // 2 - n_nodes
    for _ in range(n_extra):
        src = random.randint(0, n_nodes-1)
        dst = random.randint(0, n_nodes-1)
        if src != dst:
            edges.append((src, dst, 1.0))
            edges.append((dst, src, 1.0))
    
    return nodes, edges

def benchmark_connected_components(n_nodes: int, trials: int = 5) -> dict:
    """Benchmark Connected Components."""
    print(f"\nTesting with {n_nodes:,} nodes...")
    
    # Create test graph
    nodes, edges = create_test_graph(n_nodes)
    print(f"  Generated graph: {len(nodes):,} nodes, {len(edges):,} edges")
    
    times = []
    for trial in range(trials):
        gc.collect()
        
        # Build graph
        g = gr.Graph()
        node_ids = g.add_nodes(len(nodes))
        edge_list = [(node_ids[src], node_ids[dst]) for src, dst, _ in edges]
        g.add_edges(edge_list)
        
        # Run algorithm
        start = time.perf_counter()
        result, stats = g.apply(
            community.connected_components(output_attr="component"),
            persist=False,
            return_profile=True
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
        
        # Print some info from first trial
        if trial == 0:
            # Count unique components by checking a sample of nodes
            components_seen = set()
            for node_id in node_ids[:min(100, len(node_ids))]:
                comp = result.get_node_attribute(node_id, "component")
                if comp is not None:
                    components_seen.add(comp)
            print(f"  Found at least {len(components_seen)} components (sampled)")
            
            # Print timing breakdown
            if stats:
                print(f"  Timing breakdown:")
                for key, value in sorted(stats.items()):
                    if 'time' in key.lower() or 'ms' in key.lower():
                        print(f"    {key}: {value}")
        
        del result
        del g
        gc.collect()
    
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    print(f"  Time: {mean_time:.2f} ± {std_time:.2f} ms (median: {statistics.median(times):.2f} ms)")
    
    return {
        'nodes': n_nodes,
        'edges': len(edges),
        'mean_ms': mean_time,
        'median_ms': statistics.median(times),
        'std_ms': std_time,
        'min_ms': min(times),
        'max_ms': max(times),
        'all_times': times
    }

def main():
    print("=" * 70)
    print("Connected Components Union-Find Optimization Benchmark")
    print("=" * 70)
    
    # Test sizes
    sizes = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
    
    results = []
    for size in sizes:
        try:
            result = benchmark_connected_components(size, trials=5)
            results.append(result)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Nodes':>10} | {'Edges':>10} | {'Mean (ms)':>12} | {'Median (ms)':>12} | {'Min (ms)':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['nodes']:>10,} | {r['edges']:>10,} | {r['mean_ms']:>12.2f} | {r['median_ms']:>12.2f} | {r['min_ms']:>10.2f}")
    
    # Calculate scaling
    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        node_ratio = last['nodes'] / first['nodes']
        time_ratio = last['median_ms'] / first['median_ms']
        print(f"\nScaling: {node_ratio:.0f}x nodes → {time_ratio:.1f}x time")
        print(f"Expected for O(m α(n)): ~{node_ratio:.0f}x time")
        
        # Check if we're close to linear
        if time_ratio < node_ratio * 1.5:
            print("✅ Performance is near-linear or better!")
        else:
            print("⚠️  Performance worse than linear")
    
    # Save results
    import json
    with open('connected_components_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to connected_components_benchmark.json")

if __name__ == '__main__':
    main()
