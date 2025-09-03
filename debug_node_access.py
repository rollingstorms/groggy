#!/usr/bin/env python3
"""
Debug node access issue with meta-nodes.
"""

import groggy

def debug_node_access():
    g = groggy.Graph()
    
    node0 = g.add_node(name="Alice", age=25, salary=50000)
    node1 = g.add_node(name="Bob", age=30, salary=60000)  
    node2 = g.add_node(name="Charlie", age=35, salary=70000)
    
    print(f"Added nodes: {node0}, {node1}, {node2}")
    
    subgraph = g.nodes[[0, 1, 2]]
    print(f"Original subgraph contains nodes: {list(subgraph.node_ids)}")
    
    # Call collapse_to_node
    print("\nCollapsing to meta-node...")
    meta_node_id = subgraph.collapse_to_node({"person_count": "count"})
    print(f"Meta-node ID: {meta_node_id}")
    
    # The key insight: the meta-node is added to the main graph, not to the subgraph
    print(f"All graph nodes after collapse: {list(g.node_ids)}")
    
    # Try accessing the meta-node attribute via the main graph instead
    print("\nTrying to access meta-node attributes via main graph...")
    
    # Create a new subgraph that includes ALL nodes including the meta-node
    all_node_ids = list(g.node_ids)
    all_nodes_subgraph = g.nodes[all_node_ids]
    print(f"All-nodes subgraph contains: {list(all_nodes_subgraph.node_ids)}")
    
    # Now try to access the attributes
    contained_subgraph = all_nodes_subgraph.get_node_attribute(meta_node_id, "contained_subgraph")
    person_count = all_nodes_subgraph.get_node_attribute(meta_node_id, "person_count")
    
    print(f"  contained_subgraph: {contained_subgraph}")
    print(f"  person_count: {person_count}")
    
    # Also test accessing other attributes to make sure they exist
    print(f"  Node 0 name via all-nodes subgraph: {all_nodes_subgraph.get_node_attribute(0, 'name')}")

if __name__ == "__main__":
    debug_node_access()