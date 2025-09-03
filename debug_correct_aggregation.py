#!/usr/bin/env python3
"""
Test aggregation with correct syntax.
"""

import groggy

def debug_correct_aggregation():
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
    
    # Test sum aggregation - sum the 'age' attribute into a new 'age' attribute on meta-node
    print("\nTesting sum aggregation on age...")
    try:
        meta_node_id = subgraph.collapse_to_node({"age": "sum"})
        print(f"Meta-node created: {meta_node_id}")
        
        # Check the age attribute (should be 25 + 30 = 55)
        all_nodes = g.nodes[list(g.node_ids)]
        age_sum = all_nodes.get_node_attribute(meta_node_id, "age")
        contained_subgraph = all_nodes.get_node_attribute(meta_node_id, "contained_subgraph")
        print(f"age sum on meta-node {meta_node_id}: {age_sum}")
        print(f"contained_subgraph on meta-node {meta_node_id}: {contained_subgraph}")
        
    except Exception as e:
        print(f"❌ Error with sum aggregation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with different attribute names - create a custom attribute name
    print(f"\n\nTesting with count aggregation...")
    try:
        # Create fresh nodes
        node3 = g.add_node(name="Charlie", age=35)
        node4 = g.add_node(name="David", age=40)
        
        subgraph2 = g.nodes[[node3, node4]]
        print(f"Fresh subgraph nodes: {list(subgraph2.node_ids)}")
        
        # Count aggregation doesn't need existing attributes
        meta_node_id2 = subgraph2.collapse_to_node({"person_count": "count"})
        print(f"Second meta-node created: {meta_node_id2}")
        
        all_nodes = g.nodes[list(g.node_ids)]
        person_count = all_nodes.get_node_attribute(meta_node_id2, "person_count")
        print(f"person_count on meta-node {meta_node_id2}: {person_count}")
        
    except Exception as e:
        print(f"❌ Error with count aggregation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_correct_aggregation()
