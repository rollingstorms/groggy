#!/usr/bin/env python3
"""
Debug the basic add_node + set_attribute flow.
"""

import groggy

def debug_add_node_set_attr():
    g = groggy.Graph()
    
    print("Testing basic node creation and attribute setting...")
    
    # Test 1: Add a regular node and set an attribute directly
    try:
        node0 = g.add_node(name="Alice", age=25)
        print(f"✅ Created node {node0} with attributes during creation")
        
        # Check if the attribute was set correctly
        all_nodes = g.nodes[list(g.node_ids)]
        name = all_nodes.get_node_attribute(node0, "name")
        age = all_nodes.get_node_attribute(node0, "age")
        print(f"   Retrieved: name={name}, age={age}")
        
    except Exception as e:
        print(f"❌ Error with node creation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Create a node without attributes, then set them manually
    try:
        node1 = g.add_node()
        print(f"✅ Created empty node {node1}")
        
        # Now try to set attributes manually - this should work the same way as collapse_to_node
        # The issue is that we don't have a direct Python API for setting attributes
        # But we can use the subgraph approach
        single_node_subgraph = g.nodes[[node1]]
        
        # Check if we can access this node
        contains_check = all_nodes.get_node_attribute(node1, "name")  # Should be None
        print(f"   Node {node1} initial name attribute: {contains_check}")
        
    except Exception as e:
        print(f"❌ Error with manual attribute setting: {e}")
        import traceback
        traceback.print_exc()
        
    # Test 3: Try to replicate the exact same pattern as collapse_to_node
    print(f"\nGraph nodes after tests: {list(g.node_ids)}")
    
    # Check which node has the contained_subgraph attribute (from previous tests)
    print("\nChecking for any contained_subgraph attributes:")
    all_nodes = g.nodes[list(g.node_ids)]
    for node_id in g.node_ids:
        contained = all_nodes.get_node_attribute(node_id, "contained_subgraph")
        if contained is not None:
            print(f"  Node {node_id}: contained_subgraph = {contained}")

if __name__ == "__main__":
    debug_add_node_set_attr()