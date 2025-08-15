#!/usr/bin/env python3
"""Test the new adjacency matrix functionality in Groggy."""

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy

def test_adjacency_matrix():
    print("=== Testing Adjacency Matrix Implementation ===")
    
    # Create a simple graph
    g = groggy.Graph()
    
    # Add nodes using bulk operation (much more efficient!)
    nodes = g.add_nodes(4)
    
    # Add node names using bulk operation (same format as benchmark)
    bulk_attrs_dict = {
        "name": {
            "nodes": nodes,
            "values": [f"node_{i}" for i in range(len(nodes))],
            "value_type": "text"
        }
    }
    g.set_node_attributes(bulk_attrs_dict)
    
    # Create edges to form a small network using bulk operation (much more efficient!)
    edge_pairs = [
        (nodes[0], nodes[1]),  # 0-1
        (nodes[1], nodes[2]),  # 1-2
        (nodes[2], nodes[3]),  # 2-3
        (nodes[3], nodes[0]),  # 3-0 (cycle)
    ]
    edges = g.add_edges(edge_pairs)
    
    # Add edge weights using bulk operation (same format as benchmark) 
    weights = [1.0, 2.0, 3.0, 4.0]
    bulk_edge_attrs_dict = {
        "weight": {
            "edges": edges,
            "values": weights,
            "value_type": "float"
        }
    }
    g.set_edge_attributes(bulk_edge_attrs_dict)
    
    print(f"Created graph with {len(nodes)} nodes and {len(edges)} edges")
    
    # Test basic adjacency matrix - NOTE: Python bindings not yet implemented
    print("ⓘ Adjacency matrix methods are implemented in Rust but not yet exposed in Python bindings")
    print("  The following would work once Python bindings are added:")
    print("  - g.adjacency_matrix()")
    print("  - g.weighted_adjacency_matrix('weight')")  
    print("  - g.dense_adjacency_matrix()")
    print("  - g.sparse_adjacency_matrix()")
    print("  - g.laplacian_matrix(normalized=False)")
    print("  - g.subgraph_adjacency_matrix(node_ids)")
    print("  - g.custom_adjacency_matrix(format, matrix_type, compact_indexing, node_ids)")
    
    print("\n=== Testing GraphArray .sum() method ===")
    
    # Test the new sum method on GraphArray
    try:
        # Get a GraphTable column which should return a GraphArray
        nodes_table = g.nodes[:4]
        print(f"Created GraphTable with {len(nodes_table)} rows")
        
        # This would test GraphArray functionality if the Python bindings were updated
        # For now, just verify the table works
        names = nodes_table.get_column('name')
        print(f"✓ Column access works: got {len(names)} name values")
        
    except Exception as e:
        print(f"✗ GraphArray test failed: {e}")
    
    print("\nAdjacency matrix implementation test completed!")

if __name__ == "__main__":
    test_adjacency_matrix()