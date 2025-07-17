#!/usr/bin/env python3
"""
Check Rust memory usage patterns and potential leaks
"""

import sys
import gc
import psutil
import time

# Add local development version
local_groggy_path = '/Users/michaelroth/Documents/Code/groggy/python'
sys.path.insert(0, local_groggy_path)

import groggy as gr

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def test_rust_memory_leak():
    """Test for memory leaks in Rust bindings"""
    print("Testing Rust memory leak patterns...")
    
    # Create and destroy many small graphs
    initial_memory = get_memory_usage()
    rust_memory_baseline = None
    
    for i in range(20):
        # Create graph
        graph = gr.Graph()
        
        # Add some data
        nodes = [{'id': f'n{j}', 'value': j} for j in range(100)]
        edges = [{'source': f'n{j}', 'target': f'n{j+1}', 'weight': 1.0} for j in range(99)]
        
        graph.nodes.add(nodes)
        graph.edges.add(edges)
        
        # Check Rust memory usage
        info = graph.info()
        if rust_memory_baseline is None:
            rust_memory_baseline = float(info['attributes']['memory_graph_store_mb'])
        
        current_rust_memory = float(info['attributes']['memory_graph_store_mb'])
        python_memory = get_memory_usage()
        
        print(f"Iteration {i+1}: Python={python_memory:.1f}MB, Rust={current_rust_memory:.2f}MB")
        
        # Explicitly delete and force GC
        del graph, nodes, edges
        gc.collect()
        
        # Brief pause for cleanup
        time.sleep(0.01)
    
    final_memory = get_memory_usage()
    print(f"\nMemory change: {final_memory - initial_memory:.1f}MB")
    print(f"Rust baseline: {rust_memory_baseline:.2f}MB")

def test_incremental_memory_growth():
    """Test incremental memory growth patterns"""
    print("\nTesting incremental memory growth...")
    
    graph = gr.Graph()
    initial_memory = get_memory_usage()
    
    # Add data incrementally
    for batch in range(10):
        # Add 500 nodes and edges
        nodes = [{'id': f'n{batch}_{j}', 'value': j} for j in range(500)]
        edges = [{'source': f'n{batch}_{j}', 'target': f'n{batch}_{j+1}', 'weight': 1.0} for j in range(499)]
        
        graph.nodes.add(nodes)
        graph.edges.add(edges)
        
        # Check memory
        info = graph.info()
        python_memory = get_memory_usage()
        rust_memory = float(info['attributes']['memory_graph_store_mb'])
        
        print(f"Batch {batch+1}: Nodes={len(graph.nodes)}, Python={python_memory:.1f}MB, Rust={rust_memory:.2f}MB")
        
        # Clean up batch data
        del nodes, edges
        gc.collect()
    
    final_memory = get_memory_usage()
    print(f"\nTotal memory growth: {final_memory - initial_memory:.1f}MB")
    print(f"Final graph info: {graph.info()}")

def test_attribute_memory_usage():
    """Test attribute-specific memory usage"""
    print("\nTesting attribute memory usage...")
    
    graph = gr.Graph()
    
    # Add nodes without attributes
    nodes = [{'id': f'n{j}'} for j in range(1000)]
    mem_before = get_memory_usage()
    graph.nodes.add(nodes)
    mem_after = get_memory_usage()
    
    print(f"Nodes without attributes: {mem_after - mem_before:.1f}MB")
    print(f"Rust memory: {graph.info()['attributes']['memory_graph_store_mb']}MB")
    
    # Add attributes
    attrs = {}
    for i in range(1000):
        attrs[f'n{i}'] = {
            'role': 'engineer',
            'salary': 100000,
            'active': True,
            'score': 85.5
        }
    
    mem_before = get_memory_usage()
    graph.nodes.attr.set(attrs)
    mem_after = get_memory_usage()
    
    print(f"Adding attributes: {mem_after - mem_before:.1f}MB")
    print(f"Rust memory after attrs: {graph.info()['attributes']['memory_graph_store_mb']}MB")

def main():
    print("🔍 RUST MEMORY LEAK DETECTION")
    print("=" * 50)
    
    test_rust_memory_leak()
    test_incremental_memory_growth()
    test_attribute_memory_usage()
    
    print("\n" + "=" * 50)
    print("RUST MEMORY CHECK COMPLETE")

if __name__ == "__main__":
    main()