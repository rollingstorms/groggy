#!/usr/bin/env python3
"""
Test script to verify that g.nodes.meta.table() only shows meta nodes 
and that NaN/None values are properly handled in table display.
"""

import sys
import os

# Add the groggy module to the path
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

try:
    import groggy
    print("✓ Successfully imported groggy")
except ImportError as e:
    print(f"✗ Failed to import groggy: {e}")
    print("Make sure the python-groggy package is built and installed.")
    sys.exit(1)

def test_nodes_meta_table():
    """Test that g.nodes.meta.table() shows only meta nodes with proper NaN handling."""
    print("\n=== Testing g.nodes.meta.table() ===")
    
    # Create a graph
    g = groggy.Graph()
    
    # Add regular nodes
    node1 = g.add_node(name="Alice", age=30, role="Engineer")
    node2 = g.add_node(name="Bob", age=25, role="Designer")  
    node3 = g.add_node(name="Charlie", age=35)  # Missing role attribute
    
    print(f"Added regular nodes: {node1}, {node2}, {node3}")
    
    # Add meta-nodes (nodes with entity_type="meta")
    meta_node1 = g.add_node(entity_type="meta", name="Team A", size=10, budget=50000)
    meta_node2 = g.add_node(entity_type="meta", name="Team B", size=5)  # Missing budget
    
    print(f"Added meta-nodes: {meta_node1}, {meta_node2}")
    
    # Test 1: All nodes table should show all nodes
    print("\n--- All nodes table ---")
    all_nodes_table = g.nodes.table()
    print(f"All nodes table shape: {all_nodes_table.shape()}")
    print(f"All nodes columns: {all_nodes_table.column_names()}")
    print("All nodes table:")
    print(all_nodes_table)
    
    # Test 2: Meta nodes table should show only meta nodes
    print("\n--- Meta nodes table ---")
    meta_nodes_table = g.nodes.meta.table()
    print(f"Meta nodes table shape: {meta_nodes_table.shape()}")
    print(f"Meta nodes columns: {meta_nodes_table.column_names()}")
    print("Meta nodes table:")
    print(meta_nodes_table)
    
    # Test 3: Base nodes table should show only non-meta nodes
    print("\n--- Base nodes table ---") 
    base_nodes_table = g.nodes.base.table()
    print(f"Base nodes table shape: {base_nodes_table.shape()}")
    print(f"Base nodes columns: {base_nodes_table.column_names()}")
    print("Base nodes table:")
    print(base_nodes_table)
    
    # Verify meta nodes count
    expected_meta_count = 2
    actual_meta_count = meta_nodes_table.nrows()
    if actual_meta_count == expected_meta_count:
        print(f"✓ Meta nodes table has correct count: {actual_meta_count}")
    else:
        print(f"✗ Meta nodes table has wrong count: {actual_meta_count}, expected: {expected_meta_count}")
        
    # Verify base nodes count  
    expected_base_count = 3
    actual_base_count = base_nodes_table.nrows()
    if actual_base_count == expected_base_count:
        print(f"✓ Base nodes table has correct count: {actual_base_count}")
    else:
        print(f"✗ Base nodes table has wrong count: {actual_base_count}, expected: {expected_base_count}")

def test_edges_meta_table():
    """Test that g.edges.meta.table() works properly."""
    print("\n=== Testing g.edges.meta.table() ===")
    
    # Create a graph with edges
    g = groggy.Graph()
    
    # Add nodes
    node1 = g.add_node(name="Alice")
    node2 = g.add_node(name="Bob")
    node3 = g.add_node(name="Charlie")
    
    # Add regular edges
    edge1 = g.add_edge(node1, node2, weight=0.8, type="friendship")
    edge2 = g.add_edge(node2, node3, weight=0.6)  # Missing type
    
    # Add meta-edges  
    meta_edge1 = g.add_edge(node1, node3, entity_type="meta", weight=0.9, capacity=100)
    
    print(f"Added edges: {edge1}, {edge2}, {meta_edge1}")
    
    # Test all edges table
    print("\n--- All edges table ---")
    all_edges_table = g.edges.table()
    print(f"All edges table shape: {all_edges_table.shape()}")
    print(f"All edges columns: {all_edges_table.column_names()}")
    print("All edges table:")
    print(all_edges_table)
    
    # Test meta edges table
    print("\n--- Meta edges table ---")
    meta_edges_table = g.edges.meta.table()
    print(f"Meta edges table shape: {meta_edges_table.shape()}")
    print(f"Meta edges columns: {meta_edges_table.column_names()}")
    print("Meta edges table:")
    print(meta_edges_table)
    
    # Test base edges table
    print("\n--- Base edges table ---")
    base_edges_table = g.edges.base.table()
    print(f"Base edges table shape: {base_edges_table.shape()}")
    print(f"Base edges columns: {base_edges_table.column_names()}")
    print("Base edges table:")
    print(base_edges_table)

if __name__ == "__main__":
    test_nodes_meta_table()
    test_edges_meta_table()
    print("\n=== Test completed ===")
