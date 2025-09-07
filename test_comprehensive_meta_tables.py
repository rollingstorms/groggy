#!/usr/bin/env python3
"""
Test script to verify the fixes for both nodes and edges table functionality:
1. g.nodes.meta.table() only shows meta nodes
2. g.edges.meta.table() only shows meta edges  
3. NaN/None values are properly handled in table display
4. Auto-slicing removes empty columns
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
    sys.exit(1)

def test_nodes_meta_table_comprehensive():
    """Comprehensive test for nodes meta table functionality."""
    print("\n" + "="*60)
    print("TESTING NODES META TABLE")
    print("="*60)
    
    # Create a graph
    g = groggy.Graph()
    
    # Add regular nodes with some having missing attributes
    node1 = g.add_node(name="Alice", age=30, role="Engineer", salary=100000)
    node2 = g.add_node(name="Bob", age=25, role="Designer")  # Missing salary
    node3 = g.add_node(name="Charlie", age=35)  # Missing role and salary
    
    # Add meta-nodes with different attribute patterns
    meta_node1 = g.add_node(entity_type="meta", name="Team A", size=10, budget=50000, active=True)
    meta_node2 = g.add_node(entity_type="meta", name="Team B", size=5, active=False)  # Missing budget  
    meta_node3 = g.add_node(entity_type="meta", name="Temp Team")  # Missing size, budget, active
    
    print(f"Added regular nodes: {node1}, {node2}, {node3}")
    print(f"Added meta-nodes: {meta_node1}, {meta_node2}, {meta_node3}")
    
    # Test 1: All nodes table (should show all nodes)
    print(f"\n{'-'*20} ALL NODES TABLE {'-'*20}")
    all_nodes_table = g.nodes.table()
    print(f"Shape: {all_nodes_table.shape()}")
    print(f"Columns: {all_nodes_table.column_names()}")
    print(all_nodes_table)
    
    # Test 2: Meta nodes table (should show only meta nodes)
    print(f"\n{'-'*20} META NODES TABLE {'-'*20}")
    meta_nodes_table = g.nodes.meta.table()
    print(f"Shape: {meta_nodes_table.shape()}")
    print(f"Columns: {meta_nodes_table.column_names()}")
    print(meta_nodes_table)
    
    # Test 3: Base nodes table (should show only non-meta nodes)
    print(f"\n{'-'*20} BASE NODES TABLE {'-'*20}")
    base_nodes_table = g.nodes.base.table()
    print(f"Shape: {base_nodes_table.shape()}")
    print(f"Columns: {base_nodes_table.column_names()}")
    print(base_nodes_table)
    
    # Verify counts
    print(f"\n{'-'*20} VALIDATION {'-'*20}")
    expected_total = 6
    expected_meta = 3
    expected_base = 3
    
    actual_total = all_nodes_table.nrows()
    actual_meta = meta_nodes_table.nrows()
    actual_base = base_nodes_table.nrows()
    
    print(f"Total nodes: {actual_total} (expected {expected_total}) {'✓' if actual_total == expected_total else '✗'}")
    print(f"Meta nodes: {actual_meta} (expected {expected_meta}) {'✓' if actual_meta == expected_meta else '✗'}")
    print(f"Base nodes: {actual_base} (expected {expected_base}) {'✓' if actual_base == expected_base else '✗'}")
    
    # Verify meta nodes table doesn't contain base-only attributes (auto-slicing test)
    meta_columns = set(meta_nodes_table.column_names())
    print(f"Meta table columns: {sorted(meta_columns)}")
    
    # Should not contain base-only attributes like 'role' or 'salary'
    has_base_only_attrs = bool({'role', 'salary'} & meta_columns)
    print(f"Contains base-only attrs: {'✗' if has_base_only_attrs else '✓'}")
    
    # Should contain meta-specific attributes like 'size', 'budget'  
    has_meta_attrs = bool({'size'} & meta_columns)
    print(f"Contains meta attrs: {'✓' if has_meta_attrs else '✗'}")

def test_edges_meta_table_comprehensive():
    """Comprehensive test for edges meta table functionality."""
    print("\n" + "="*60)
    print("TESTING EDGES META TABLE") 
    print("="*60)
    
    # Create a graph with nodes and edges
    g = groggy.Graph()
    
    # Add nodes
    node1 = g.add_node(name="Alice")
    node2 = g.add_node(name="Bob") 
    node3 = g.add_node(name="Charlie")
    node4 = g.add_node(name="Diana")
    
    print(f"Added nodes: {node1}, {node2}, {node3}, {node4}")
    
    # Add regular edges with some missing attributes
    edge1 = g.add_edge(node1, node2, weight=0.8, type="friendship", strength="strong")
    edge2 = g.add_edge(node2, node3, weight=0.6, type="colleague")  # Missing strength
    edge3 = g.add_edge(node3, node4, weight=0.4)  # Missing type and strength
    
    # Add meta-edges with different attribute patterns
    meta_edge1 = g.add_edge(node1, node3, entity_type="meta", capacity=100, flow=80, active=True)
    meta_edge2 = g.add_edge(node2, node4, entity_type="meta", capacity=50, active=False)  # Missing flow
    meta_edge3 = g.add_edge(node1, node4, entity_type="meta", capacity=75)  # Missing flow and active
    
    print(f"Added regular edges: {edge1}, {edge2}, {edge3}")
    print(f"Added meta-edges: {meta_edge1}, {meta_edge2}, {meta_edge3}")
    
    # Test 1: All edges table (should show all edges)
    print(f"\n{'-'*20} ALL EDGES TABLE {'-'*20}")
    all_edges_table = g.edges.table()
    print(f"Shape: {all_edges_table.shape()}")
    print(f"Columns: {all_edges_table.column_names()}")
    print(all_edges_table)
    
    # Test 2: Meta edges table (should show only meta edges)
    print(f"\n{'-'*20} META EDGES TABLE {'-'*20}")
    meta_edges_table = g.edges.meta.table()
    print(f"Shape: {meta_edges_table.shape()}")
    print(f"Columns: {meta_edges_table.column_names()}")
    print(meta_edges_table)
    
    # Test 3: Base edges table (should show only non-meta edges)
    print(f"\n{'-'*20} BASE EDGES TABLE {'-'*20}")
    base_edges_table = g.edges.base.table()
    print(f"Shape: {base_edges_table.shape()}")
    print(f"Columns: {base_edges_table.column_names()}")
    print(base_edges_table)
    
    # Verify counts
    print(f"\n{'-'*20} VALIDATION {'-'*20}")
    expected_total = 6
    expected_meta = 3
    expected_base = 3
    
    actual_total = all_edges_table.nrows()
    actual_meta = meta_edges_table.nrows()
    actual_base = base_edges_table.nrows()
    
    print(f"Total edges: {actual_total} (expected {expected_total}) {'✓' if actual_total == expected_total else '✗'}")
    print(f"Meta edges: {actual_meta} (expected {expected_meta}) {'✓' if actual_meta == expected_meta else '✗'}")
    print(f"Base edges: {actual_base} (expected {expected_base}) {'✓' if actual_base == expected_base else '✗'}")
    
    # Verify meta edges table doesn't contain base-only attributes (auto-slicing test)
    meta_columns = set(meta_edges_table.column_names())
    print(f"Meta table columns: {sorted(meta_columns)}")
    
    # Should not contain base-only attributes like 'type' or 'strength'
    has_base_only_attrs = bool({'type', 'strength'} & meta_columns)
    print(f"Contains base-only attrs: {'✗' if has_base_only_attrs else '✓'}")
    
    # Should contain meta-specific attributes like 'capacity'
    has_meta_attrs = bool({'capacity'} & meta_columns)
    print(f"Contains meta attrs: {'✓' if has_meta_attrs else '✗'}")
    
    # Verify required edge columns are always present
    required_cols = {'edge_id', 'source', 'target'}
    has_required_cols = required_cols.issubset(meta_columns)
    print(f"Has required edge columns: {'✓' if has_required_cols else '✗'}")

def test_nan_handling():
    """Test that NaN/None values are properly displayed in tables."""
    print("\n" + "="*60)
    print("TESTING NaN/None HANDLING")
    print("="*60)
    
    g = groggy.Graph()
    
    # Create nodes where some have attributes and others don't
    node1 = g.add_node(name="Complete", age=30, salary=50000, active=True)
    node2 = g.add_node(name="Partial", age=25)  # Missing salary, active
    node3 = g.add_node(name="Minimal")  # Missing age, salary, active
    
    # Create meta-nodes with mixed attributes 
    meta1 = g.add_node(entity_type="meta", name="MetaComplete", size=10, budget=1000)
    meta2 = g.add_node(entity_type="meta", name="MetaPartial", size=5)  # Missing budget
    
    print("Added nodes with mixed attribute patterns")
    
    # Test all nodes table - should show NaN/None for missing values
    print(f"\n{'-'*20} ALL NODES - NaN HANDLING {'-'*20}")
    all_table = g.nodes.table()
    print(f"Shape: {all_table.shape()}")
    print(all_table)
    
    # Test meta nodes table - should only show relevant columns
    print(f"\n{'-'*20} META NODES - FILTERED {'-'*20}")
    meta_table = g.nodes.meta.table()
    print(f"Shape: {meta_table.shape()}")
    print(meta_table)
    
    # Test base nodes table - should only show relevant columns  
    print(f"\n{'-'*20} BASE NODES - FILTERED {'-'*20}")
    base_table = g.nodes.base.table()
    print(f"Shape: {base_table.shape()}")
    print(base_table)

if __name__ == "__main__":
    test_nodes_meta_table_comprehensive()
    test_edges_meta_table_comprehensive()
    test_nan_handling()
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
