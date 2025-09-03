#!/usr/bin/env python3
"""
Debug collapse_to_node step by step.
"""

import groggy

def debug_step_by_step():
    g = groggy.Graph()
    
    node0 = g.add_node(name="Alice", age=25, salary=50000)
    node1 = g.add_node(name="Bob", age=30, salary=60000)  
    node2 = g.add_node(name="Charlie", age=35, salary=70000)
    
    print(f"Added nodes: {node0}, {node1}, {node2}")
    
    subgraph = g.nodes[[0, 1, 2]]
    
    # Check that we can retrieve the attributes from the original nodes
    print("Checking original node attributes:")
    for node_id in [0, 1, 2]:
        name = subgraph.get_node_attribute(node_id, "name")
        age = subgraph.get_node_attribute(node_id, "age")  
        salary = subgraph.get_node_attribute(node_id, "salary")
        print(f"  Node {node_id}: name={name}, age={age}, salary={salary}")
    
    # Create a simple meta-node without aggregation first
    print("\nTesting collapse_to_node with empty aggregation functions...")
    try:
        meta_node_id = subgraph.collapse_to_node({})
        print(f"Meta-node ID with no aggregation: {meta_node_id}")
        
        # Check if the contained_subgraph attribute was at least set
        contained_subgraph = subgraph.get_node_attribute(meta_node_id, "contained_subgraph")
        print(f"  contained_subgraph: {contained_subgraph}")
        
    except Exception as e:
        print(f"Error with empty aggregation: {e}")
        import traceback
        traceback.print_exc()
    
    # Try with one simple aggregation 
    print("\nTesting collapse_to_node with one aggregation function...")
    try:
        # Create a new subgraph to avoid conflicts
        subgraph2 = g.nodes[[0, 1, 2]]
        meta_node_id2 = subgraph2.collapse_to_node({
            "person_count": "count"
        })
        print(f"Meta-node ID with count aggregation: {meta_node_id2}")
        
        # Check if both attributes were set
        contained_subgraph2 = subgraph2.get_node_attribute(meta_node_id2, "contained_subgraph")
        person_count = subgraph2.get_node_attribute(meta_node_id2, "person_count")
        
        print(f"  contained_subgraph: {contained_subgraph2}")
        print(f"  person_count: {person_count}")
        
    except Exception as e:
        print(f"Error with count aggregation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_step_by_step()