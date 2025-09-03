#!/usr/bin/env python3
"""
Debug the collapse_to_node method step by step.
"""

import groggy

def debug_collapse_steps():
    g = groggy.Graph()
    
    # Create the same initial setup as our tests
    node0 = g.add_node(name="Alice", age=25, salary=50000)
    node1 = g.add_node(name="Bob", age=30, salary=60000)  
    node2 = g.add_node(name="Charlie", age=35, salary=70000)
    
    print(f"Setup: nodes {node0}, {node1}, {node2}")
    print(f"All graph nodes: {list(g.node_ids)}")
    
    # Step 1: Create a subgraph (this should work fine)
    subgraph = g.nodes[[0, 1, 2]]
    print(f"Created subgraph with nodes: {list(subgraph.node_ids)}")
    
    # Step 2: Test calling collapse_to_node with NO aggregation functions
    # This should only create the meta-node and set the contained_subgraph attribute
    print("\nTesting collapse_to_node with empty aggregation...")
    
    try:
        meta_node_id = subgraph.collapse_to_node({})
        print(f"Meta-node created with ID: {meta_node_id}")
        print(f"All graph nodes after collapse: {list(g.node_ids)}")
        
        # Check immediately after collapse where the contained_subgraph attribute went
        all_nodes = g.nodes[list(g.node_ids)]
        print("\nChecking contained_subgraph attribute on all nodes:")
        
        for node_id in g.node_ids:
            contained = all_nodes.get_node_attribute(node_id, "contained_subgraph")
            print(f"  Node {node_id}: contained_subgraph = {contained}")
            
        # This should tell us exactly which node got the attribute
        
    except Exception as e:
        print(f"❌ Error with collapse_to_node: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Test with ONE simple aggregation function to see if that changes anything
    print(f"\n\nTesting with a fresh subgraph and one aggregation function...")
    try:
        # Create fresh nodes to avoid conflicts
        node3 = g.add_node(name="David", age=40, salary=80000)
        node4 = g.add_node(name="Eve", age=28, salary=55000)
        
        subgraph2 = g.nodes[[node3, node4]]
        print(f"Fresh subgraph with nodes: {list(subgraph2.node_ids)}")
        
        meta_node_id2 = subgraph2.collapse_to_node({"person_count": "count"})
        print(f"Second meta-node created with ID: {meta_node_id2}")
        print(f"All graph nodes after second collapse: {list(g.node_ids)}")
        
        # Check where both the contained_subgraph AND person_count attributes went
        all_nodes = g.nodes[list(g.node_ids)]
        print("\nChecking ALL attributes on all nodes:")
        
        for node_id in g.node_ids:
            contained = all_nodes.get_node_attribute(node_id, "contained_subgraph")
            person_count = all_nodes.get_node_attribute(node_id, "person_count")
            if contained is not None or person_count is not None:
                print(f"  Node {node_id}: contained_subgraph = {contained}, person_count = {person_count}")
        
    except Exception as e:
        print(f"❌ Error with second collapse: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_collapse_steps()