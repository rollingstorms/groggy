#!/usr/bin/env python3
"""
Performance benchmarks for GLI optimizations
"""

import time
import random
import sys
import os
from typing import List, Tuple

# Add the python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import gli


def benchmark_function(func, *args, **kwargs) -> Tuple[float, any]:
    """Benchmark a function and return (time_taken, result)"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return end - start, result


def benchmark_graph_creation():
    """Benchmark different graph creation methods"""
    print("=== Graph Creation Benchmarks ===")
    
    # Test small graphs
    sizes = [100, 500, 1000, 5000]
    edge_prob = 0.1
    
    for size in sizes:
        print(f"\nGraph size: {size} nodes")
        
        # Method 1: Sequential adds
        def sequential_creation():
            graph = Graph.empty()
            for i in range(size):
                graph = graph.add_node(f"node_{i}")
            
            # Add edges
            for i in range(size):
                for j in range(i + 1, size):
                    if random.random() < edge_prob:
                        graph = graph.add_edge(f"node_{i}", f"node_{j}")
            return graph.snapshot()
        
        # Method 2: Batch operations
        def batch_creation():
            graph = Graph.empty()
            with graph.batch_operations() as batch:
                for i in range(size):
                    batch.add_node(f"node_{i}")
                
                for i in range(size):
                    for j in range(i + 1, size):
                        if random.random() < edge_prob:
                            batch.add_edge(f"node_{i}", f"node_{j}")
            return graph.snapshot()
        
        # Method 3: Vectorized creation
        def vectorized_creation():
            return create_random_graph(size, edge_prob)
        
        # Run benchmarks
        random.seed(42)  # Consistent results
        time1, graph1 = benchmark_function(sequential_creation)
        
        random.seed(42)
        time2, graph2 = benchmark_function(batch_creation)
        
        random.seed(42)
        time3, graph3 = benchmark_function(vectorized_creation)
        
        print(f"  Sequential:  {time1*1000:.1f}ms ({len(graph1.nodes)} nodes, {len(graph1.edges)} edges)")
        print(f"  Batch:       {time2*1000:.1f}ms ({len(graph2.nodes)} nodes, {len(graph2.edges)} edges)")
        print(f"  Vectorized:  {time3*1000:.1f}ms ({len(graph3.nodes)} nodes, {len(graph3.edges)} edges)")
        print(f"  Speedup batch/seq:      {time1/time2:.1f}x")
        print(f"  Speedup vectorized/seq: {time1/time3:.1f}x")


def benchmark_state_management():
    """Benchmark state creation and reconstruction"""
    print("\n=== State Management Benchmarks ===")
    
    store = GraphStore()
    
    # Create base graph
    graph = create_random_graph(1000, 0.05)
    store.update_graph(graph, "initial")
    
    # Benchmark state creation
    def create_many_states():
        for i in range(100):
            current = store.get_current_graph()
            current = current.add_node(f"new_node_{i}", iteration=i)
            store.update_graph(current, f"add_node_{i}")
    
    time_taken, _ = benchmark_function(create_many_states)
    print(f"Creating 100 states: {time_taken*1000:.1f}ms ({time_taken*10:.1f}ms per state)")
    
    # Benchmark reconstruction
    def reconstruct_states():
        history = store.get_history()
        reconstructed = []
        for entry in history[-10:]:  # Last 10 states
            graph = store._reconstruct_graph_from_state(entry['hash'])
            reconstructed.append(graph)
        return reconstructed
    
    time_taken, graphs = benchmark_function(reconstruct_states)
    print(f"Reconstructing 10 states: {time_taken*1000:.1f}ms ({time_taken*100:.1f}ms per state)")


def benchmark_graph_operations():
    """Benchmark common graph operations"""
    print("\n=== Graph Operations Benchmarks ===")
    
    # Create test graph
    graph = create_random_graph(2000, 0.02)
    print(f"Test graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Benchmark subgraph creation
    def create_subgraph():
        return graph.create_subgraph(node_filter=lambda n: int(n.id.split('_')[1]) < 500)
    
    time_taken, subgraph = benchmark_function(create_subgraph)
    print(f"Subgraph creation: {time_taken*1000:.1f}ms ({len(subgraph.nodes)} nodes)")
    
    # Benchmark connected component
    def find_component():
        return graph.get_connected_component("node_0")
    
    time_taken, component = benchmark_function(find_component)
    print(f"Connected component: {time_taken*1000:.1f}ms ({len(component.nodes)} nodes)")
    
    # Benchmark effective data access (simulating multiple queries)
    def access_effective_data():
        results = []
        for _ in range(100):
            effective_nodes, effective_edges, _ = graph._get_effective_data()
            results.append(len(effective_nodes))
        return results
    
    time_taken, results = benchmark_function(access_effective_data)
    print(f"100 effective data accesses: {time_taken*1000:.1f}ms ({time_taken*10:.1f}ms per access)")


def benchmark_branching():
    """Benchmark branching operations"""
    print("\n=== Branching Benchmarks ===")
    
    store = GraphStore()
    base_graph = create_random_graph(500, 0.05)
    store.update_graph(base_graph, "base")
    store.commit("Base graph")
    
    # Create multiple branches
    def create_branches():
        for i in range(10):
            store.create_branch(f"branch_{i}", description=f"Test branch {i}")
    
    time_taken, _ = benchmark_function(create_branches)
    print(f"Creating 10 branches: {time_taken*1000:.1f}ms")
    
    # Switch between branches
    def switch_branches():
        branches = store.list_branches()
        for branch in branches[:5]:
            store.switch_branch(branch['name'])
    
    time_taken, _ = benchmark_function(switch_branches)
    print(f"Switching between 5 branches: {time_taken*1000:.1f}ms")


def run_memory_usage_test():
    """Test memory efficiency of content pooling"""
    print("\n=== Memory Usage Test ===")
    
    store = GraphStore()
    
    # Create many similar graphs to test deduplication
    base_graph = create_random_graph(100, 0.1)
    
    print("Creating 50 similar graphs...")
    for i in range(50):
        # Create a fresh copy of the base graph each time
        fresh_base = create_random_graph(100, 0.1)
        # Slightly modify the graph
        modified = fresh_base.add_node(f"unique_{i}", index=i)
        store.update_graph(modified, f"iteration_{i}")
    
    stats = store.get_storage_stats()
    print(f"Storage stats after 50 iterations:")
    print(f"  Total states: {stats['total_states']}")
    print(f"  Pooled nodes: {stats['pooled_nodes']}")
    print(f"  Pooled edges: {stats['pooled_edges']}")
    print(f"  Cached reconstructions: {stats['cached_reconstructions']}")
    
    # Estimate memory savings
    total_unique_content = stats['pooled_nodes'] + stats['pooled_edges']
    theoretical_without_pooling = stats['total_states'] * 100  # Rough estimate
    savings_ratio = theoretical_without_pooling / total_unique_content if total_unique_content > 0 else 1
    print(f"  Estimated memory savings: {savings_ratio:.1f}x")


if __name__ == "__main__":
    print("GLI Performance Benchmarks")
    print("==========================")
    
    benchmark_graph_creation()
    benchmark_state_management()
    benchmark_graph_operations()
    benchmark_branching()
    run_memory_usage_test()
    
    print("\n=== Summary ===")
    print("Key optimizations implemented:")
    print("1. LazyDict for zero-copy views")
    print("2. Incremental cache updates")
    print("3. Batch operation context managers")
    print("4. Vectorized graph construction")
    print("5. Content-addressed storage with deduplication")
    print("6. Performance monitoring decorators")
