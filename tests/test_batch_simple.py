#!/usr/bin/env python3
"""
Simple test for batch operations
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/gli')

# Force reload of the module
import importlib
import gli
importlib.reload(gli.graph)
import gli

def test_batch_operations():
    print("Testing batch operations...")
    
    gli.set_backend('rust')
    g = gli.Graph.empty()
    
    print("Creating batch context...")
    with g.batch_operations() as batch:
        print("Adding nodes to batch...")
        node1 = batch.add_node(name="Test1", value=1)
        node2 = batch.add_node(name="Test2", value=2)
        
        print("Adding edge to batch...")
        edge1 = batch.add_edge(node1, node2, weight=0.5)
        
        print(f"Batch nodes: {list(batch.batch_nodes.keys())}")
        print(f"Batch edges: {list(batch.batch_edges.keys())}")
    
    print(f"Graph after batch: {g.node_count()} nodes, {g.edge_count()} edges")
    
    for node_id in g.nodes:
        node = g.get_node(node_id)
        print(f"  Node: {node.id} - {dict(node.attributes)}")

if __name__ == "__main__":
    test_batch_operations()
