#!/usr/bin/env python3
"""
Simple test to check NaN issue in tables
"""
import groggy

def test_nan_issue():
    print("Creating test graph...")
    g = groggy.Graph()
    
    # Add nodes and edges with some missing attributes
    n1 = g.add_node(name='Alice', age=25)
    n2 = g.add_node(name='Bob')  # Missing age
    n3 = g.add_node(age=35)      # Missing name
    
    e1 = g.add_edge(n1, n2, strength=5)        # Missing type
    e2 = g.add_edge(n2, n3, type='colleague')  # Missing strength
    
    # Test nodes table
    print("\n=== Nodes Table ===")
    nodes_table = g.table().nodes
    print(nodes_table)
    
    print("\n=== Edges Table ===")
    edges_table = g.table().edges
    print(edges_table)
    
    # Test meta nodes (should be empty but properly formatted)
    print("\n=== Meta Nodes Table ===")
    try:
        meta_nodes = g.nodes.meta.table()
        print(f"Meta nodes shape: {meta_nodes.shape()}")
        print(meta_nodes)
    except Exception as e:
        print(f"Meta nodes error: {e}")
    
    # Test meta edges (should be empty but properly formatted)  
    print("\n=== Meta Edges Table ===")
    try:
        meta_edges = g.edges.meta.table()
        print(f"Meta edges shape: {meta_edges.shape()}")
        print(meta_edges)
    except Exception as e:
        print(f"Meta edges error: {e}")

if __name__ == '__main__':
    test_nan_issue()
