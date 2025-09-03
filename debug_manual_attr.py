#!/usr/bin/env python3
"""
Debug manual attribute setting on meta-node.
"""

import groggy

def debug_manual_attr():
    g = groggy.Graph()
    
    node0 = g.add_node(name="Alice", age=25, salary=50000)
    node1 = g.add_node(name="Bob", age=30, salary=60000)  
    node2 = g.add_node(name="Charlie", age=35, salary=70000)
    
    print(f"Added nodes: {node0}, {node1}, {node2}")
    
    # Add a regular node to test attribute setting
    regular_node = g.add_node()
    print(f"Added regular node: {regular_node}")
    
    # Create a subgraph and get the meta-node
    subgraph = g.nodes[[0, 1, 2]]
    meta_node_id = subgraph.collapse_to_node({})  # Empty aggregation first
    print(f"Created meta-node: {meta_node_id}")
    
    # Now try to manually set attributes on both the regular node and meta-node
    print("\nTesting manual attribute setting...")
    
    # Set attribute on regular node via graph  
    try:
        # Try different ways to set attributes
        g.nodes[regular_node].test_attr = "regular_value"  # Try direct assignment
        print(f"✅ Set attribute on regular node {regular_node}")
    except Exception as e:
        print(f"❌ Failed to set attribute on regular node: {e}")
    
    # Set attribute on meta-node via graph
    try:
        g.nodes[meta_node_id].test_attr = "meta_value"  # Try direct assignment
        print(f"✅ Set attribute on meta-node {meta_node_id}")
    except Exception as e:
        print(f"❌ Failed to set attribute on meta-node: {e}")
    
    # Check if we can read them back
    print("\nReading attributes back...")
    all_nodes = g.nodes[list(g.node_ids)]
    
    regular_value = all_nodes.get_node_attribute(regular_node, "test_attr")
    meta_value = all_nodes.get_node_attribute(meta_node_id, "test_attr")
    
    print(f"Regular node test_attr: {regular_value}")
    print(f"Meta-node test_attr: {meta_value}")

if __name__ == "__main__":
    debug_manual_attr()