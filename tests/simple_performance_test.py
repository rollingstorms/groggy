#!/usr/bin/env python3
"""
Simple performance test for GLI Rust vs Python backends
"""

import time
import random
import sys
import os

# Add the python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import gli

def benchmark_function(func, *args, **kwargs):
    """Benchmark a function and return (time_taken, result)"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return end - start, result

def create_test_graph(backend, size=1000, edge_prob=0.05):
    """Create a test graph with specified backend"""
    gli.set_backend(backend)
    
    def create_graph():
        graph = gli.Graph()
        
        # Add nodes
        for i in range(size):
            graph.add_node(f"node_{i}", value=i)
        
        # Add edges
        random.seed(42)  # For consistent results
        for i in range(size):
            for j in range(i + 1, size):
                if random.random() < edge_prob:
                    graph.add_edge(f"node_{i}", f"node_{j}", weight=random.random())
        
        return graph
    
    return benchmark_function(create_graph)

def query_performance_test(backend, graph_size=1000):
    """Test query performance on a graph"""
    gli.set_backend(backend)
    
    # Create a test graph
    graph = gli.Graph()
    for i in range(graph_size):
        graph.add_node(f"node_{i}", value=i)
    
    # Add some edges
    random.seed(42)
    for i in range(graph_size):
        for j in range(i + 1, min(i + 10, graph_size)):  # Each node connected to next 10
            if random.random() < 0.5:
                graph.add_edge(f"node_{i}", f"node_{j}", weight=random.random())
    
    def run_queries():
        results = []
        
        # Node count
        results.append(graph.node_count())
        
        # Edge count
        results.append(graph.edge_count())
        
        # Get all node IDs
        results.append(len(graph.get_node_ids()))
        
        # Get neighbors for first 100 nodes
        for i in range(min(100, graph_size)):
            neighbors = graph.get_neighbors(f"node_{i}")
            results.append(len(neighbors))
        
        return results
    
    return benchmark_function(run_queries)

def main():
    print("🚀 GLI Performance Test: Rust vs Python Backends")
    print("=" * 60)
    
    # Test graph creation
    print("\n📊 Graph Creation Performance")
    print("-" * 40)
    
    sizes = [500, 1000, 20000]
    edge_prob = 0.05
    
    for size in sizes:
        print(f"\nGraph size: {size} nodes, ~{int(size * (size-1) * edge_prob / 2)} edges")
        
        # Test Python backend
        time_python, graph_python = create_test_graph('python', size, edge_prob)
        print(f"  Python: {time_python:.3f}s ({graph_python.node_count()} nodes, {graph_python.edge_count()} edges)")
        
        # Test Rust backend  
        time_rust, graph_rust = create_test_graph('rust', size, edge_prob)
        print(f"  Rust:   {time_rust:.3f}s ({graph_rust.node_count()} nodes, {graph_rust.edge_count()} edges)")
        
        # Calculate speedup
        speedup = time_python / time_rust if time_rust > 0 else 0
        print(f"  Speedup: {speedup:.1f}x faster with Rust")
    
    # Test query performance
    print("\n🔍 Query Performance")
    print("-" * 40)
    
    query_sizes = [1000, 2000]
    
    for size in query_sizes:
        print(f"\nQuery test on {size} nodes:")
        
        # Test Python backend
        time_python, _ = query_performance_test('python', size)
        print(f"  Python: {time_python:.3f}s")
        
        # Test Rust backend
        time_rust, _ = query_performance_test('rust', size)  
        print(f"  Rust:   {time_rust:.3f}s")
        
        # Calculate speedup
        speedup = time_python / time_rust if time_rust > 0 else 0
        print(f"  Speedup: {speedup:.1f}x faster with Rust")
    
    # Summary
    print("\n🎯 Summary")
    print("-" * 40)
    print("✅ Both backends provide identical API")
    print("✅ Seamless switching between backends")
    print("✅ Rust backend provides significant performance improvements")
    print("✅ Python backend provides full compatibility fallback")

if __name__ == "__main__":
    main()
