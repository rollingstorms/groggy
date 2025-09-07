#!/usr/bin/env python3
"""
Quick test to see what methods are available on the graph for creating meta nodes.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

try:
    import groggy
    g = groggy.Graph()
    
    print("Available methods on Graph:")
    methods = [method for method in dir(g) if not method.startswith('_')]
    for method in sorted(methods):
        print(f"  {method}")
        
    print(f"\nTrying to add a regular node...")
    node1 = g.add_node(name="Alice", age=30)
    print(f"Added node: {node1}")
    
    # Let me see if I can manually set entity_type after creation
    print(f"\nTrying to manually set entity_type...")
    node2 = g.add_node(name="Bob")
    
    # Check what attributes can be set
    print(f"Available methods on node2: {type(node2)}")
    
    # Try to create table and see the structure
    print(f"\nNodes table structure:")
    nodes_table = g.nodes.table()
    print(f"Shape: {nodes_table.shape()}")
    print(f"Columns: {nodes_table.column_names()}")
    print(nodes_table)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
