#!/usr/bin/env python3
"""
Test for performance regression in basic operations
"""

import time
import groggy as gr

def test_basic_creation_performance():
    """Test that basic node/edge creation hasn't regressed"""
    print("Testing basic creation performance...")
    
    # Test 1: Basic node creation (legacy API)
    print("\n=== Test 1: Legacy node creation ===")
    graph = gr.Graph()
    
    start = time.time()
    node_ids = graph.add_nodes(10000)  # Should use fast legacy path
    creation_time = time.time() - start
    print(f"Created {len(node_ids)} nodes in {creation_time:.4f}s")
    print(f"Rate: {len(node_ids)/creation_time:,.0f} nodes/sec")
    
    # Test 2: Basic edge creation (legacy API)
    print("\n=== Test 2: Legacy edge creation ===")
    start = time.time()
    edge_specs = []
    for i in range(5000):
        source = i % len(node_ids)
        target = (i + 1) % len(node_ids)
        edge_specs.append((source, target))
    
    edge_ids = graph.add_edges(edge_specs)  # Should use fast legacy path
    edge_creation_time = time.time() - start
    print(f"Created {len(edge_ids)} edges in {edge_creation_time:.4f}s") 
    print(f"Rate: {len(edge_ids)/edge_creation_time:,.0f} edges/sec")
    
    # Test 3: Direct node addition (most basic)
    print("\n=== Test 3: Direct node addition ===")
    graph2 = gr.Graph()
    start = time.time()
    direct_nodes = []
    for i in range(10000):
        direct_nodes.append(graph2.add_node())
    direct_creation_time = time.time() - start
    print(f"Created {len(direct_nodes)} nodes directly in {direct_creation_time:.4f}s")
    print(f"Rate: {len(direct_nodes)/direct_creation_time:,.0f} nodes/sec")
    
    # Test 4: Single node with attributes (new API)
    print("\n=== Test 4: Nodes with attributes (kwargs) ===")
    graph3 = gr.Graph()
    start = time.time()
    attr_nodes = []
    for i in range(1000):  # Fewer nodes for this more expensive operation
        attr_nodes.append(graph3.add_node(id=i, name=f"node_{i}", value=i*2.5))
    attr_creation_time = time.time() - start
    print(f"Created {len(attr_nodes)} nodes with attributes in {attr_creation_time:.4f}s")
    print(f"Rate: {len(attr_nodes)/attr_creation_time:,.0f} nodes/sec")
    
    print("\n=== Performance Summary ===")
    print(f"Legacy bulk nodes: {len(node_ids)/creation_time:,.0f} nodes/sec")
    print(f"Direct single nodes: {len(direct_nodes)/direct_creation_time:,.0f} nodes/sec")
    print(f"Kwargs single nodes: {len(attr_nodes)/attr_creation_time:,.0f} nodes/sec")
    print(f"Bulk edges: {len(edge_ids)/edge_creation_time:,.0f} edges/sec")
    
    # Ratio comparison
    if direct_creation_time > 0 and creation_time > 0:
        bulk_advantage = (len(node_ids)/creation_time) / (len(direct_nodes)/direct_creation_time)
        print(f"\nBulk vs Direct: {bulk_advantage:.1f}x faster")
    
    if attr_creation_time > 0 and direct_creation_time > 0:
        attr_overhead = (len(direct_nodes)/direct_creation_time) / (len(attr_nodes)/attr_creation_time)
        print(f"Direct vs Kwargs: {attr_overhead:.1f}x faster")

if __name__ == "__main__":
    test_basic_creation_performance()