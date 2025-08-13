#!/usr/bin/env python3
"""
Test simple node creation
"""

import time
import groggy as gr

def test_simple_node_creation():
    """Test the simplest possible node creation"""
    
    print("Testing simple node creation...")
    
    # Test 1: Most basic - add_node() with no parameters
    print("\n=== Test 1: add_node() with no parameters ===")
    graph = gr.Graph()
    start = time.time()
    nodes = []
    for i in range(10000):
        nodes.append(graph.add_node())
    no_param_time = time.time() - start
    
    print(f"Created {len(nodes)} nodes with add_node() in {no_param_time:.4f}s")
    print(f"Rate: {len(nodes)/no_param_time:,.0f} nodes/sec")
    
    # Test 2: Bulk creation 
    print("\n=== Test 2: add_nodes(count) bulk ===")
    graph2 = gr.Graph()
    start = time.time()
    bulk_nodes = graph2.add_nodes(10000)
    bulk_time = time.time() - start
    
    print(f"Created {len(bulk_nodes)} nodes with add_nodes(count) in {bulk_time:.4f}s")
    print(f"Rate: {len(bulk_nodes)/bulk_time:,.0f} nodes/sec")
    
    print(f"\n=== Comparison ===")
    print(f"Individual add_node(): {len(nodes)/no_param_time:,.0f} nodes/sec")
    print(f"Bulk add_nodes(): {len(bulk_nodes)/bulk_time:,.0f} nodes/sec")
    
    bulk_advantage = (len(bulk_nodes)/bulk_time) / (len(nodes)/no_param_time)
    print(f"Bulk is {bulk_advantage:.1f}x faster")

if __name__ == "__main__":
    test_simple_node_creation()