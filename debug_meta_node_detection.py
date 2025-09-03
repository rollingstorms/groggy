#!/usr/bin/env python3
"""
Debug MetaNode Detection 🔍
"""

import groggy

def debug_meta_node_detection():
    print("🔍 Debugging MetaNode Detection...")
    
    g = groggy.Graph()
    
    # Create test data
    node1 = g.add_node(name="Alice", age=25)
    node2 = g.add_node(name="Bob", age=30)
    
    subgraph = g.nodes[[node1, node2]]
    
    # Create meta-node
    meta_node = subgraph.add_to_graph({"avg_age": ("mean", "age")})
    meta_node_id = meta_node.node_id
    
    print(f"Created meta-node with ID: {meta_node_id}")
    
    # Check what entity_type attribute is actually stored
    all_nodes = g.nodes[list(g.node_ids)]
    
    print(f"\nAll node IDs: {list(g.node_ids)}")
    
    for node_id in g.node_ids:
        try:
            entity_type = all_nodes.get_node_attribute(node_id, "entity_type")
            print(f"Node {node_id}: entity_type = '{entity_type}' (type: {type(entity_type)})")
        except Exception as e:
            print(f"Node {node_id}: entity_type access failed: {e}")
    
    # Test the accessor method
    print(f"\nTesting get_meta_node({meta_node_id}):")
    result = g.nodes.get_meta_node(meta_node_id)
    print(f"Result: {result}")

if __name__ == "__main__":
    debug_meta_node_detection()