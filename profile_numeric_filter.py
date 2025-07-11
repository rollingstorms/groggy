#!/usr/bin/env python3
"""
Profile numeric filtering performance to identify bottlenecks
"""

import time
import random
import cProfile
import pstats
from typing import Dict, List, Any

# Set up local groggy import
import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')
import groggy as gr


def create_test_graph(num_nodes=10000, num_edges=5000):
    """Create a test graph with numeric attributes"""
    print(f"Creating test graph: {num_nodes} nodes, {num_edges} edges...")
    
    # Create nodes with salary attribute
    nodes_data = []
    for i in range(num_nodes):
        nodes_data.append({
            'id': f'n{i}',
            'salary': random.randint(50000, 150000),
            'age': random.randint(25, 65),
            'role': random.choice(['engineer', 'manager', 'analyst', 'designer'])
        })
    
    # Create edges with strength attribute
    edges_data = []
    for i in range(num_edges):
        source = f'n{random.randint(0, num_nodes-1)}'
        target = f'n{random.randint(0, num_nodes-1)}'
        if source != target:
            edges_data.append({
                'source': source,
                'target': target,
                'strength': random.uniform(0.1, 1.0),
                'relationship': random.choice(['friend', 'colleague', 'family'])
            })
    
    # Create Groggy graph
    start = time.time()
    graph = gr.Graph(backend='rust')
    graph.add_nodes(nodes_data)
    graph.add_edges(edges_data)
    creation_time = time.time() - start
    print(f"Graph created in {creation_time:.4f}s")
    
    return graph


def profile_numeric_filtering():
    """Profile the numeric filtering operations"""
    graph = create_test_graph(10000, 5000)
    
    print("\n=== PROFILING NUMERIC FILTERING ===")
    
    # Test different filtering scenarios
    filters = [
        ("Node salary > 100000", lambda: graph.filter_nodes({'salary': ('>', 100000)})),
        ("Node salary > 80000 AND age > 30", lambda: graph.filter_nodes({'salary': ('>', 80000), 'age': ('>', 30)})),
        ("Edge strength > 0.7", lambda: graph.filter_edges({'strength': ('>', 0.7)})),
        ("Node exact role match", lambda: graph.filter_nodes({'role': 'engineer'})),
        ("Edge exact relationship match", lambda: graph.filter_edges({'relationship': 'friend'})),
    ]
    
    for description, filter_func in filters:
        print(f"\n--- {description} ---")
        
        # Warm-up run
        filter_func()
        
        # Timed runs
        times = []
        for _ in range(5):
            start = time.time()
            result = filter_func()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"Results: {len(result)} items")
        print(f"Times: avg={avg_time:.4f}s, min={min_time:.4f}s, max={max_time:.4f}s")


def profile_rust_methods_directly():
    """Test Rust backend methods directly to isolate performance"""
    graph = create_test_graph(10000, 5000)
    
    print("\n=== PROFILING RUST METHODS DIRECTLY ===")
    
    # Test Rust methods directly
    rust_core = graph._rust_core
    
    tests = [
        ("Rust exact node filter", lambda: rust_core.filter_nodes_by_attributes({'role': 'engineer'})),
        ("Rust numeric node filter", lambda: rust_core.filter_nodes_by_numeric_comparison('salary', '>', 100000.0)),
        ("Rust exact edge filter", lambda: rust_core.filter_edges_by_attributes({'relationship': 'friend'})),
        ("Rust numeric edge filter", lambda: rust_core.filter_edges_by_numeric_comparison('strength', '>', 0.7)),
    ]
    
    for description, test_func in tests:
        print(f"\n--- {description} ---")
        
        # Warm-up
        test_func()
        
        # Timed runs
        times = []
        for _ in range(10):
            start = time.time()
            result = test_func()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        
        print(f"Results: {len(result)} items")
        print(f"Times: avg={avg_time:.4f}s, min={min_time:.4f}s")


def detailed_profile_numeric_filter():
    """Use cProfile to get detailed performance breakdown"""
    print("\n=== DETAILED PROFILING WITH cProfile ===")
    
    graph = create_test_graph(10000, 5000)
    
    def test_function():
        # Run multiple filtering operations
        for _ in range(10):
            graph.filter_nodes({'salary': ('>', 100000)})
            graph.filter_edges({'strength': ('>', 0.7)})
    
    # Profile the test function
    profiler = cProfile.Profile()
    profiler.enable()
    test_function()
    profiler.disable()
    
    # Print top time-consuming functions
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions


if __name__ == "__main__":
    profile_numeric_filtering()
    profile_rust_methods_directly()
    detailed_profile_numeric_filter()
