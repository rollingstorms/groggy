#!/usr/bin/env python3
"""
Test FastGraphCore performance optimization
"""

import groggy as gr
import time
import json

def test_fast_core_basic():
    """Test basic FastGraphCore operations"""
    print("=== Testing FastGraphCore Basic Operations ===")
    
    # Create graph
    g = gr.Graph()
    
    # Test fast core node operations
    print("\n1. Testing fast_add_nodes...")
    start = time.perf_counter()
    nodes = [f"n{i}" for i in range(1000)]
    g.fast_add_nodes(nodes)
    add_time = time.perf_counter() - start
    print(f"   Added 1000 nodes in {add_time:.6f}s ({1000/add_time:.0f} nodes/sec)")
    
    # Test fast core edge operations  
    print("\n2. Testing fast_add_edges...")
    start = time.perf_counter()
    edges = [(f"n{i}", f"n{(i+1)%1000}") for i in range(1000)]
    g.fast_add_edges(edges)
    add_edge_time = time.perf_counter() - start
    print(f"   Added 1000 edges in {add_edge_time:.6f}s ({1000/add_edge_time:.0f} edges/sec)")
    
    # Test fast core attribute setting
    print("\n3. Testing fast_set_node_attr...")
    start = time.perf_counter()
    for i in range(100):
        g.fast_set_node_attr("salary", f"n{i}", json.dumps(50000 + i * 1000))
    attr_time = time.perf_counter() - start
    print(f"   Set 100 node attributes in {attr_time:.6f}s ({100/attr_time:.0f} attrs/sec)")
    
    # Test batch attribute setting
    print("\n4. Testing fast_set_node_attrs_batch...")
    start = time.perf_counter()
    batch_data = {f"n{i}": json.dumps(f"role_{i%4}") for i in range(100, 200)}
    g.fast_set_node_attrs_batch("role", batch_data)
    batch_time = time.perf_counter() - start
    print(f"   Batch set 100 node attributes in {batch_time:.6f}s ({100/batch_time:.0f} attrs/sec)")
    
    # Test getting attributes
    print("\n5. Testing fast_get_node_attr...")
    start = time.perf_counter()
    for i in range(100):
        value = g.fast_get_node_attr("salary", f"n{i}")
    get_time = time.perf_counter() - start
    print(f"   Got 100 node attributes in {get_time:.6f}s ({100/get_time:.0f} attrs/sec)")
    
    # Test ID retrieval
    print("\n6. Testing fast_node_ids and fast_edge_ids...")
    start = time.perf_counter()
    node_ids = g.fast_node_ids()
    node_time = time.perf_counter() - start
    
    start = time.perf_counter()
    edge_ids = g.fast_edge_ids()
    edge_time = time.perf_counter() - start
    
    print(f"   Retrieved {len(node_ids)} node IDs in {node_time:.6f}s")
    print(f"   Retrieved {len(edge_ids)} edge IDs in {edge_time:.6f}s")
    
    # Check memory usage
    print("\n7. Memory usage comparison:")
    info = g.info()
    fast_core_memory = float(info["attributes"].get("memory_fast_core_mb", "0"))
    graph_store_memory = float(info["attributes"].get("memory_graph_store_mb", "0"))
    content_pool_memory = float(info["attributes"].get("memory_content_pool_mb", "0"))
    
    print(f"   FastCore memory: {fast_core_memory:.2f} MB")
    print(f"   GraphStore memory: {graph_store_memory:.2f} MB")
    print(f"   ContentPool memory: {content_pool_memory:.2f} MB")
    print(f"   FastCore nodes: {info['attributes'].get('fast_core_nodes', 'N/A')}")
    print(f"   FastCore edges: {info['attributes'].get('fast_core_edges', 'N/A')}")
    
    total_time = add_time + add_edge_time + attr_time + batch_time + get_time
    print(f"\n🎯 Total FastCore operations time: {total_time:.6f}s")
    
    return {
        'add_nodes_time': add_time,
        'add_edges_time': add_edge_time, 
        'set_attr_time': attr_time,
        'batch_attr_time': batch_time,
        'get_attr_time': get_time,
        'fast_core_memory_mb': fast_core_memory,
        'total_time': total_time
    }

def test_fast_vs_regular():
    """Compare FastCore vs regular operations"""
    print("\n=== FastCore vs Regular Operations Comparison ===")
    
    # Regular operations
    print("\n1. Regular operations...")
    g1 = gr.Graph()
    
    start = time.perf_counter()
    nodes_data = [{'id': f"n{i}", 'salary': 50000 + i * 1000} for i in range(1000)]
    g1.nodes.add(nodes_data)
    regular_time = time.perf_counter() - start
    print(f"   Regular: 1000 nodes with attrs in {regular_time:.6f}s")
    
    # Fast core operations
    print("\n2. FastCore operations...")
    g2 = gr.Graph()
    
    start = time.perf_counter()
    nodes = [f"n{i}" for i in range(1000)]
    g2.fast_add_nodes(nodes)
    
    batch_data = {f"n{i}": json.dumps(50000 + i * 1000) for i in range(1000)}
    g2.fast_set_node_attrs_batch("salary", batch_data)
    fast_time = time.perf_counter() - start
    print(f"   FastCore: 1000 nodes with attrs in {fast_time:.6f}s")
    
    if fast_time > 0:
        speedup = regular_time / fast_time
        print(f"\n🚀 FastCore speedup: {speedup:.2f}x faster")
    else:
        print("\n⚡ FastCore was too fast to measure!")
    
    return {'regular_time': regular_time, 'fast_time': fast_time, 'speedup': regular_time/fast_time if fast_time > 0 else float('inf')}

if __name__ == "__main__":
    print("🔥 FastGraphCore Performance Test")
    print("=" * 50)
    
    # Test basic operations
    basic_results = test_fast_core_basic()
    
    # Test comparison
    comparison_results = test_fast_vs_regular()
    
    print(f"\n📊 Summary:")
    print(f"   FastCore total time: {basic_results['total_time']:.6f}s")
    print(f"   FastCore memory: {basic_results['fast_core_memory_mb']:.2f} MB")
    print(f"   Speedup vs regular: {comparison_results['speedup']:.2f}x")
    
    # Target check
    target_speedup = 5.0  # Aiming for 5x as first milestone toward 10x
    if comparison_results['speedup'] >= target_speedup:
        print(f"✅ SUCCESS: FastCore achieved {comparison_results['speedup']:.2f}x speedup (target: {target_speedup}x)")
    else:
        print(f"⚠️  PARTIAL: FastCore achieved {comparison_results['speedup']:.2f}x speedup (target: {target_speedup}x)")
        print("   Further optimization needed for 10x goal.")