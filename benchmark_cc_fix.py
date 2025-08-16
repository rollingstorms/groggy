#!/usr/bin/env python3

"""
Before/after comparison for connected_components fix
"""

import time
import groggy

def benchmark_connected_components():
    """Benchmark with the exact same test case"""
    
    print("Connected Components Performance - After Fix")
    print("=" * 50)
    
    # Create the same 500-node social network
    g = groggy.generators.social_network(n=500)
    print(f"Graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Run multiple times for average
    times = []
    for i in range(5):
        start_time = time.time()
        components = g.connected_components()
        end_time = time.time()
        duration = end_time - start_time
        times.append(duration)
        print(f"Run {i+1}: {duration:.6f} seconds")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage time: {avg_time:.6f} seconds")
    print(f"Components found: {len(components)}")
    
    # Compare with previous results
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"Before fix:  ~0.760000 seconds")
    print(f"After fix:   {avg_time:.6f} seconds")
    improvement = 0.760000 / avg_time
    print(f"Improvement: {improvement:.1f}x faster")
    
    if improvement > 5:
        print("✅ Significant performance improvement achieved!")
    else:
        print("⚠️  Limited improvement - may need further optimization")

if __name__ == "__main__":
    benchmark_connected_components()
