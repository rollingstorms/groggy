#!/usr/bin/env python3
"""
Quick validation and demonstration of the optimized connected components.
Run this to verify the optimization is working correctly.
"""

import groggy as gr
from groggy import algorithms
import time

def demo():
    print("=" * 70)
    print("High-Performance Connected Components - Validation Demo")
    print("=" * 70)
    
    # Test 1: Basic correctness
    print("\n1. Testing basic correctness...")
    g = gr.Graph()
    nodes = g.add_nodes(7)
    
    # Create 3 components
    g.add_edge(nodes[0], nodes[1])  # Component 1
    g.add_edge(nodes[2], nodes[3])  # Component 2
    g.add_edge(nodes[3], nodes[4])
    g.add_edge(nodes[5], nodes[6])  # Component 3
    
    g.apply(algorithms.community.connected_components(
        mode='undirected', output_attr='cc'))
    
    cc = [g.nodes[n]['cc'] for n in nodes]
    assert cc[0] == cc[1], "Nodes 0-1 should be in same component"
    assert cc[2] == cc[3] == cc[4], "Nodes 2-4 should be in same component"
    assert cc[5] == cc[6], "Nodes 5-6 should be in same component"
    assert len(set(cc)) == 3, "Should have exactly 3 components"
    print(f"   ✓ Found 3 components correctly: {cc}")
    
    # Test 2: Performance on larger graph
    print("\n2. Testing performance on 50K node graph...")
    g_large = gr.Graph()
    n = 50000
    nodes_large = g_large.add_nodes(n)
    
    # Create 50 components of 1000 nodes each
    for i in range(0, n-1):
        if (i + 1) % 1000 != 0:  # Break into components every 1000 nodes
            g_large.add_edge(nodes_large[i], nodes_large[i + 1])
    
    start = time.time()
    g_large.apply(algorithms.community.connected_components(
        mode='undirected', output_attr='component'))
    elapsed = time.time() - start
    
    ns_per_node = (elapsed * 1e9) / n
    print(f"   ✓ Processed {n:,} nodes in {elapsed*1000:.2f}ms")
    print(f"   ✓ Performance: {ns_per_node:.2f} ns/node")
    
    if ns_per_node < 2.0:
        print(f"   ✓ EXCELLENT: Sub-2ns/node performance achieved!")
    elif ns_per_node < 5.0:
        print(f"   ✓ GOOD: Sub-5ns/node performance achieved!")
    else:
        print(f"   ! Performance slower than expected")
    
    # Test 3: Strong connectivity
    print("\n3. Testing strongly connected components...")
    g_dir = gr.Graph(directed=True)
    nodes_dir = g_dir.add_nodes(6)
    
    # Create a strong component (cycle)
    g_dir.add_edge(nodes_dir[0], nodes_dir[1])
    g_dir.add_edge(nodes_dir[1], nodes_dir[2])
    g_dir.add_edge(nodes_dir[2], nodes_dir[0])
    
    # Not strongly connected (just a path)
    g_dir.add_edge(nodes_dir[3], nodes_dir[4])
    g_dir.add_edge(nodes_dir[4], nodes_dir[5])
    
    g_dir.apply(algorithms.community.connected_components(
        mode='strong', output_attr='scc'))
    
    scc = [g_dir.nodes[n]['scc'] for n in nodes_dir]
    assert scc[0] == scc[1] == scc[2], "Cycle should be one SCC"
    assert len(set(scc[3:])) == 3, "Path nodes should each be own SCC"
    print(f"   ✓ Strong components detected correctly")
    print(f"   ✓ Cycle nodes: {scc[0:3]} (all same)")
    print(f"   ✓ Path nodes: {scc[3:]} (all different)")
    
    print("\n" + "=" * 70)
    print("All validation tests passed! ✓")
    print("Optimizations are working correctly.")
    print("=" * 70)

if __name__ == '__main__':
    try:
        demo()
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
