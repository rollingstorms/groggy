#!/usr/bin/env python3
"""
Test if we can set entity_type attribute directly and if the filtering works.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

try:
    import groggy
    g = groggy.Graph()
    
    print("Creating nodes...")
    
    # Create regular nodes
    node1 = g.add_node(name="Alice", age=30)
    node2 = g.add_node(name="Bob", age=25)
    
    print(f"Created regular nodes: {node1}, {node2}")
    
    # Try to set entity_type after creation using set_node_attr
    print(f"\nTrying to set entity_type via set_node_attr...")
    
    try:
        # Create a node that we'll make into a meta-node
        node3 = g.add_node(name="Team A", size=10)
        g.set_node_attr(node3, "entity_type", "meta")
        print(f"✓ Successfully set entity_type=meta for node {node3}")
        
        node4 = g.add_node(name="Team B", size=5) 
        g.set_node_attr(node4, "entity_type", "meta")
        print(f"✓ Successfully set entity_type=meta for node {node4}")
        
    except Exception as e:
        print(f"✗ Failed to set entity_type: {e}")
    
    # Test the tables
    print(f"\n{'='*50}")
    print("TESTING TABLE FILTERING")
    print("="*50)
    
    # All nodes table
    print(f"\n--- All Nodes Table ---")
    all_nodes = g.nodes.table()
    print(f"Shape: {all_nodes.shape()}")
    print(f"Columns: {all_nodes.column_names()}")
    print(all_nodes)
    
    # Test meta nodes accessor
    print(f"\n--- Meta Nodes Length ---")
    meta_accessor = g.nodes.meta
    print(f"Meta nodes count: {len(meta_accessor)}")
    
    # Test meta nodes table
    print(f"\n--- Meta Nodes Table ---")
    try:
        meta_table = g.nodes.meta.table()
        print(f"Meta table shape: {meta_table.shape()}")
        print(f"Meta table columns: {meta_table.column_names()}")
        print(meta_table)
    except Exception as e:
        print(f"Error getting meta table: {e}")
        import traceback
        traceback.print_exc()
    
    # Test base nodes table  
    print(f"\n--- Base Nodes Table ---")
    try:
        base_table = g.nodes.base.table()
        print(f"Base table shape: {base_table.shape()}")
        print(f"Base table columns: {base_table.column_names()}")
        print(base_table)
    except Exception as e:
        print(f"Error getting base table: {e}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"Overall error: {e}")
    import traceback
    traceback.print_exc()
