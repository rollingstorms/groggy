#!/usr/bin/env python3
"""
Debug contained_subgraph attribute specifically.
"""

import groggy

def debug_contained_subgraph():
    g = groggy.Graph()
    
    node0 = g.add_node(name="Alice", age=25, salary=50000)
    node1 = g.add_node(name="Bob", age=30, salary=60000)  
    node2 = g.add_node(name="Charlie", age=35, salary=70000)
    
    print(f"Added nodes: {node0}, {node1}, {node2}")
    
    # Create subgraph and collapse
    subgraph = g.nodes[[0, 1, 2]]
    meta_node_id = subgraph.collapse_to_node({"person_count": "count"})
    print(f"Created meta-node: {meta_node_id}")
    
    print("\nChecking all nodes for contained_subgraph attribute...")
    all_nodes = g.nodes[list(g.node_ids)]
    
    # Check each node for the contained_subgraph attribute
    for node_id in g.node_ids:
        contained_subgraph = all_nodes.get_node_attribute(node_id, "contained_subgraph")
        print(f"  Node {node_id}: contained_subgraph = {contained_subgraph}")
    
    print("\nChecking meta-node for aggregated attribute...")
    person_count = all_nodes.get_node_attribute(meta_node_id, "person_count")
    print(f"  Meta-node {meta_node_id}: person_count = {person_count}")
    
    # Let's also test accessing via direct attribute access
    print("\nTesting direct attribute access...")
    try:
        contained_via_direct = g.nodes[meta_node_id].contained_subgraph
        print(f"  Direct access contained_subgraph: {contained_via_direct}")
    except Exception as e:
        print(f"  Direct access failed: {e}")
        
    try:
        count_via_direct = g.nodes[meta_node_id].person_count
        print(f"  Direct access person_count: {count_via_direct}")
    except Exception as e:
        print(f"  Direct access failed: {e}")

if __name__ == "__main__":
    debug_contained_subgraph()