#!/usr/bin/env python3
"""
Debug sum aggregation properly.
"""

import groggy

def debug_sum_aggregation():
    g = groggy.Graph()
    
    # Create nodes with age attributes
    node1 = g.add_node(name="Alice", age=25)
    node2 = g.add_node(name="Bob", age=30)
    
    print(f"Setup: nodes {node1}, {node2}")
    
    # Check what attributes these nodes actually have
    all_nodes = g.nodes[list(g.node_ids)]
    print("\nNode attributes before collapse:")
    for node_id in [node1, node2]:
        name = all_nodes.get_node_attribute(node_id, "name")
        age = all_nodes.get_node_attribute(node_id, "age")
        print(f"  Node {node_id}: name={name}, age={age}")
    
    # Create subgraph
    subgraph = g.nodes[[node1, node2]]
    print(f"\nSubgraph nodes: {list(subgraph.node_ids)}")
    
    # Test sum aggregation on age - the key is the new attribute name, value is the function
    print("\nTesting sum aggregation on age...")
    try:
        meta_node_id = subgraph.collapse_to_node({"total_age": "sum"})
        print(f"Meta-node created: {meta_node_id}")
        
        # Check the total_age attribute
        all_nodes = g.nodes[list(g.node_ids)]
        total_age = all_nodes.get_node_attribute(meta_node_id, "total_age")
        print(f"total_age on meta-node {meta_node_id}: {total_age}")
        
    except Exception as e:
        print(f"❌ Error with sum aggregation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_sum_aggregation()
