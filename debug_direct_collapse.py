#!/usr/bin/env python3
"""
Debug collapse_to_node directly and check attributes immediately.
"""

import groggy

def debug_direct_collapse():
    g = groggy.Graph()
    
    node0 = g.add_node(name="Alice", age=25, salary=50000)
    node1 = g.add_node(name="Bob", age=30, salary=60000)  
    node2 = g.add_node(name="Charlie", age=35, salary=70000)
    
    print(f"Added nodes: {node0}, {node1}, {node2}")
    
    subgraph = g.nodes[[0, 1, 2]]
    
    # Call collapse_to_node directly
    print("Calling collapse_to_node...")
    meta_node_id = subgraph.collapse_to_node({
        "total_salary": "sum",
        "avg_age": "mean",
        "person_count": "count"
    })
    
    print(f"Meta-node ID: {meta_node_id}")
    
    # Now try to access the attributes using subgraph.get_node_attribute()
    print("Checking attributes via subgraph.get_node_attribute():")
    
    try:
        total_salary = subgraph.get_node_attribute(meta_node_id, "total_salary")
        print(f"  total_salary: {total_salary}")
        
        avg_age = subgraph.get_node_attribute(meta_node_id, "avg_age")
        print(f"  avg_age: {avg_age}")
        
        person_count = subgraph.get_node_attribute(meta_node_id, "person_count")
        print(f"  person_count: {person_count}")
        
        # Also check the contained_subgraph attribute
        contained_subgraph = subgraph.get_node_attribute(meta_node_id, "contained_subgraph")
        print(f"  contained_subgraph: {contained_subgraph}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Try the same with the graph directly  
    print("\nCreating new subgraph that includes the meta-node:")
    try:
        # Create a subgraph that includes the meta-node
        full_subgraph = g.nodes[[0, 1, 2, meta_node_id]]
        
        total_salary2 = full_subgraph.get_node_attribute(meta_node_id, "total_salary")
        print(f"  total_salary via full subgraph: {total_salary2}")
        
        avg_age2 = full_subgraph.get_node_attribute(meta_node_id, "avg_age")
        print(f"  avg_age via full subgraph: {avg_age2}")
        
        person_count2 = full_subgraph.get_node_attribute(meta_node_id, "person_count")
        print(f"  person_count via full subgraph: {person_count2}")
        
    except Exception as e:
        print(f"Error with full subgraph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_direct_collapse()