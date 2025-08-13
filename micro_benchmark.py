#!/usr/bin/env python3
"""
Micro-benchmark to isolate the exact bottleneck in query performance.
Focus on measuring individual components at a granular level.
"""

import groggy
import time
import random

def create_simple_graph(size):
    """Create a simple graph for micro-benchmarking"""
    graph = groggy.Graph()
    
    # Create nodes
    node_ids = []
    for i in range(size):
        node_id = graph.add_node()
        node_ids.append(node_id)
    
    # Add attributes
    for i, node_id in enumerate(node_ids):
        graph.set_node_attribute(node_id, "dept", groggy.AttrValue(f"dept_{i % 3}"))  # 3 departments
        graph.set_node_attribute(node_id, "score", groggy.AttrValue(random.randint(1, 100)))
    
    return graph

def micro_benchmark_components():
    """Test individual components to find the bottleneck"""
    
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\n🔬 MICRO-BENCHMARK: {size} nodes")
        print("=" * 50)
        
        # Create graph
        start_time = time.time()
        graph = create_simple_graph(size)
        creation_time = time.time() - start_time
        print(f"   Graph creation: {creation_time:.6f}s")
        
        # Test 1: Simple attribute filter (dept)
        dept_filter = groggy.NodeFilter.attribute_equals("dept", groggy.AttrValue("dept_0"))
        
        start_time = time.time()
        dept_results = graph.filter_nodes(dept_filter)
        dept_time = time.time() - start_time
        print(f"   Dept filter: {dept_time:.6f}s ({len(dept_results)} results)")
        
        # Test 2: Numeric attribute filter (score)
        score_filter = groggy.NodeFilter.attribute_filter("score", 
            groggy.AttributeFilter.greater_than(groggy.AttrValue(50)))
        
        start_time = time.time()
        score_results = graph.filter_nodes(score_filter)
        score_time = time.time() - start_time
        print(f"   Score filter: {score_time:.6f}s ({len(score_results)} results)")
        
        # Test 3: AND filter
        and_filter = groggy.NodeFilter.and_filters([dept_filter, score_filter])
        
        start_time = time.time()
        and_results = graph.filter_nodes(and_filter)
        and_time = time.time() - start_time
        print(f"   AND filter: {and_time:.6f}s ({len(and_results)} results)")
        
        # Test 4: Multiple individual calls (Python overhead test)
        start_time = time.time()
        for _ in range(10):
            _ = graph.filter_nodes(dept_filter)
        multi_time = time.time() - start_time
        print(f"   10x dept filter: {multi_time:.6f}s (avg: {multi_time/10:.6f}s)")
        
        # Calculate efficiency
        expected_and_time = dept_time + score_time
        and_efficiency = expected_and_time / and_time if and_time > 0 else 0
        print(f"   AND efficiency: {and_efficiency:.2f}x")
        
        # Per-node timing
        dept_per_node = dept_time / size * 1_000_000  # microseconds
        score_per_node = score_time / size * 1_000_000
        and_per_node = and_time / size * 1_000_000
        
        print(f"   Per-node timing:")
        print(f"     Dept: {dept_per_node:.2f} μs/node")
        print(f"     Score: {score_per_node:.2f} μs/node") 
        print(f"     AND: {and_per_node:.2f} μs/node")

def test_scaling_pattern():
    """Test if the scaling is truly O(n²) or has other patterns"""
    
    print(f"\n🔬 SCALING PATTERN ANALYSIS")
    print("=" * 50)
    
    sizes = [1000, 2000, 4000, 8000]  # Powers of 2 for clear scaling analysis
    times = []
    
    for size in sizes:
        graph = create_simple_graph(size)
        
        # Use the simple dept filter for consistency
        dept_filter = groggy.NodeFilter.attribute_equals("dept", groggy.AttrValue("dept_0"))
        
        # Time multiple runs for accuracy
        total_time = 0
        runs = 5
        for _ in range(runs):
            start_time = time.time()
            _ = graph.filter_nodes(dept_filter)
            total_time += time.time() - start_time
        
        avg_time = total_time / runs
        times.append(avg_time)
        
        per_node = avg_time / size * 1_000_000
        print(f"   {size:5d} nodes: {avg_time:.6f}s ({per_node:.2f} μs/node)")
    
    # Analyze scaling
    print(f"\n   Scaling Analysis:")
    for i in range(1, len(sizes)):
        size_ratio = sizes[i] / sizes[i-1]
        time_ratio = times[i] / times[i-1]
        complexity = time_ratio / size_ratio
        
        print(f"     {sizes[i-1]} -> {sizes[i]}: {size_ratio:.1f}x size, {time_ratio:.2f}x time")
        print(f"       Complexity factor: {complexity:.2f} (1.0=O(n), 2.0=O(n²))")

if __name__ == "__main__":
    print("🔬 GROGGY MICRO-BENCHMARK")
    print("Finding the exact bottleneck in query performance")
    print("=" * 60)
    
    micro_benchmark_components()
    test_scaling_pattern()
    
    print(f"\n🎯 ANALYSIS COMPLETE")
    print("Look for:")
    print("  - Which individual operation is slowest")
    print("  - Whether AND optimization is working") 
    print("  - What the complexity factor reveals about scaling")
