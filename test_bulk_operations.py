#!/usr/bin/env python3
"""Test bulk operations using the direct Rust binding format that works."""

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy

def test_bulk_operations():
    print("=== Testing Bulk Operations (Direct Rust Binding Format) ===")
    
    # Create a simple graph
    g = groggy.Graph()
    
    # Add nodes using bulk operation
    nodes = g.add_nodes(4)
    print(f"Created {len(nodes)} nodes: {nodes}")
    
    # Create edges using bulk operation
    edge_pairs = [
        (nodes[0], nodes[1]),  # 0-1
        (nodes[1], nodes[2]),  # 1-2
        (nodes[2], nodes[3]),  # 2-3
        (nodes[3], nodes[0]),  # 3-0 (cycle)
    ]
    edges = g.add_edges(edge_pairs)
    print(f"Created {len(edges)} edges: {edges}")
    
    # Test bulk node attributes using the direct Rust format (like in benchmark)
    try:
        # This is the format that the Rust binding expects
        rust_node_attrs = {
            "name": {
                "nodes": nodes,
                "values": [f"node_{i}" for i in range(len(nodes))],
                "value_type": "text"
            }
        }
        g._rust_graph.set_node_attributes(rust_node_attrs)
        print("✓ Bulk node attributes set successfully using direct Rust format")
        
        # Verify by getting the attributes back
        for i, node in enumerate(nodes):
            attr = g.get_node_attribute(node, "name")
            print(f"  Node {node}: name = {attr}")
            
    except Exception as e:
        print(f"✗ Bulk node attributes failed: {e}")
    
    # Test bulk edge attributes using the direct Rust format
    try:
        rust_edge_attrs = {
            "weight": {
                "edges": edges,
                "values": [1.0 + i * 0.5 for i in range(len(edges))],
                "value_type": "float"
            }
        }
        g._rust_graph.set_edge_attributes(rust_edge_attrs)
        print("✓ Bulk edge attributes set successfully using direct Rust format")
        
        # Verify by getting the attributes back
        for i, edge in enumerate(edges):
            attr = g.get_edge_attribute(edge, "weight")
            print(f"  Edge {edge}: weight = {attr}")
            
    except Exception as e:
        print(f"✗ Bulk edge attributes failed: {e}")
    
    print("\n=== Testing Python Wrapper Format (Expected to Fail) ===")
    
    # Test the Python wrapper format that should work but currently doesn't
    try:
        # This format should work according to the Python wrapper docs
        name_attrs = [(node, groggy.AttrValue(f"node_{i}_v2")) for i, node in enumerate(nodes)]
        g.set_node_attributes({"name_v2": name_attrs})
        print("✓ Python wrapper format worked!")
    except Exception as e:
        print(f"✗ Python wrapper format failed: {e}")
        print("  This confirms there's a mismatch between Python wrapper and Rust binding")
    
    print("\nBulk operations test completed!")

if __name__ == "__main__":
    test_bulk_operations()