#!/usr/bin/env python3
"""
Debug the aggregation function step by step.
"""

import groggy

def debug_aggregation():
    g = groggy.Graph()
    
    # Create nodes with explicit attributes to aggregate
    node3 = g.add_node(name="David", age=40, salary=80000)
    node4 = g.add_node(name="Eve", age=28, salary=55000)
    
    print(f"Setup: nodes {node3}, {node4}")
    
    # Let's check what attributes these nodes actually have
    all_nodes = g.nodes[list(g.node_ids)]
    print("\nNode attributes before collapse:")
    for node_id in [node3, node4]:
        name = all_nodes.get_node_attribute(node_id, "name")
        age = all_nodes.get_node_attribute(node_id, "age")
        salary = all_nodes.get_node_attribute(node_id, "salary")
        print(f"  Node {node_id}: name={name}, age={age}, salary={salary}")
    
    # Create subgraph and test different aggregation functions
    subgraph = g.nodes[[node3, node4]]
    print(f"\nSubgraph nodes: {list(subgraph.node_ids)}")
    
    # Test count aggregation (should work even if no specific attribute)
    print("\nTesting count aggregation...")
    try:
        meta_node_id = subgraph.collapse_to_node({"person_count": "count"})
        print(f"Meta-node created: {meta_node_id}")
        
        # Check the person_count attribute
        all_nodes = g.nodes[list(g.node_ids)]
        person_count = all_nodes.get_node_attribute(meta_node_id, "person_count")
        print(f"person_count on meta-node {meta_node_id}: {person_count}")
        
    except Exception as e:
        print(f"❌ Error with count aggregation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test sum aggregation on age
    print(f"\n\nTesting sum aggregation on age...")
    try:
        # Create fresh nodes
        node5 = g.add_node(name="Frank", age=50, salary=90000)
        node6 = g.add_node(name="Grace", age=32, salary=75000)
        
        subgraph2 = g.nodes[[node5, node6]]
        print(f"Fresh subgraph nodes: {list(subgraph2.node_ids)}")
        
        meta_node_id2 = subgraph2.collapse_to_node({"total_age": "sum", "attr": "age"})
        print(f"Second meta-node created: {meta_node_id2}")
        
        all_nodes = g.nodes[list(g.node_ids)]
        total_age = all_nodes.get_node_attribute(meta_node_id2, "total_age")
        print(f"total_age on meta-node {meta_node_id2}: {total_age}")
        
    except Exception as e:
        print(f"❌ Error with sum aggregation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_aggregation()
