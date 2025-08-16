#!/usr/bin/env python3

"""
Test script to compare connected components performance between old and new implementations.
"""

import time
import groggy
import sys
import psutil
import gc

def create_test_graph(num_nodes=1000, num_edges=2000, num_components=5):
    """Create a test graph with specified characteristics"""
    g = groggy.Graph()
    
    # Create nodes in separate components
    nodes_per_component = num_nodes // num_components
    node_mapping = {}
    
    # Add nodes with component info for debugging
    for comp in range(num_components):
        start_node = comp * nodes_per_component
        end_node = min((comp + 1) * nodes_per_component, num_nodes)
        
        for i in range(start_node, end_node):
            node_id = g.add_node(component=comp, node_index=i)
            node_mapping[i] = node_id
    
    # Add edges within components to ensure connectivity
    edges_per_component = num_edges // num_components
    
    for comp in range(num_components):
        start_node = comp * nodes_per_component
        end_node = min((comp + 1) * nodes_per_component, num_nodes)
        
        # Create a connected component with random edges
        import random
        for _ in range(edges_per_component):
            source = random.randint(start_node, end_node - 1)
            target = random.randint(start_node, end_node - 1)
            if source != target and source in node_mapping and target in node_mapping:
                try:
                    g.add_edge(node_mapping[source], node_mapping[target])
                except:
                    pass  # Edge might already exist
    
    return g

def measure_memory():
    """Get current memory usage"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_connected_components(g, label="Test"):
    """Benchmark connected components with memory tracking"""
    print(f"\n{label}")
    print("-" * 50)
    
    # Force garbage collection before test
    gc.collect()
    mem_before = measure_memory()
    
    start_time = time.time()
    components = g.connected_components()
    end_time = time.time()
    
    mem_after = measure_memory()
    
    duration = end_time - start_time
    mem_used = mem_after - mem_before
    
    print(f"Nodes: {g.node_count()}")
    print(f"Edges: {g.edge_count()}")
    print(f"Components found: {len(components)}")
    print(f"Time: {duration:.4f} seconds")
    print(f"Memory used: {mem_used:.2f} MB")
    print(f"Memory before: {mem_before:.2f} MB")
    print(f"Memory after: {mem_after:.2f} MB")
    
    # Print component sizes
    sizes = [comp.node_count() for comp in components]
    sizes.sort(reverse=True)
    print(f"Component sizes: {sizes[:10]}")  # Show top 10
    
    return duration, mem_used, len(components)

def main():
    print("Connected Components Performance Test")
    print("=" * 50)
    
    # Test different graph sizes
    test_cases = [
        (100, 200, 3),    # Small
        (500, 1000, 5),   # Medium  
        (1000, 2000, 5),  # Large
        (2000, 4000, 10), # Very Large
    ]
    
    results = []
    
    for nodes, edges, components in test_cases:
        print(f"\nCreating test graph ({nodes} nodes, {edges} edges, {components} components)...")
        g = create_test_graph(nodes, edges, components)
        
        # Run multiple times to get average
        durations = []
        for i in range(3):
            duration, mem_used, found_components = benchmark_connected_components(
                g, f"Run {i+1}: {nodes} nodes"
            )
            durations.append(duration)
        
        avg_duration = sum(durations) / len(durations)
        results.append((nodes, edges, avg_duration, found_components))
        
        print(f"Average time: {avg_duration:.4f} seconds")
    
    # Summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"{'Nodes':<8} {'Edges':<8} {'Time (s)':<10} {'Components':<12}")
    print("-" * 50)
    for nodes, edges, duration, comp_count in results:
        print(f"{nodes:<8} {edges:<8} {duration:<10.4f} {comp_count:<12}")

if __name__ == "__main__":
    main()
