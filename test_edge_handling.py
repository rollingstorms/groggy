#!/usr/bin/env python3
"""
Test what happens to edges when collapsing subgraphs.
"""

import groggy

def test_edge_handling():
    g = groggy.Graph()
    
    # Create a triangle of nodes with edges
    node1 = g.add_node(name="Alice")
    node2 = g.add_node(name="Bob") 
    node3 = g.add_node(name="Charlie")
    
    # Add edges within the group
    edge1 = g.add_edge(node1, node2)
    edge2 = g.add_edge(node2, node3)
    edge3 = g.add_edge(node1, node3)
    
    # Add an external node and edge
    external_node = g.add_node(name="David")
    external_edge = g.add_edge(node1, external_node)  # Edge from subgraph to external
    
    print(f"Before collapse:")
    print(f"  Nodes: {list(g.node_ids)}")
    print(f"  Edges: {list(g.edge_ids)}")
    print(f"  Internal edges: {[edge1, edge2, edge3]}")
    print(f"  External edge: {external_edge}")
    
    # Collapse the triangle
    subgraph = g.nodes[[node1, node2, node3]]
    meta_node = subgraph.collapse_to_node({"count": "count"})
    
    print(f"\nAfter collapse:")
    print(f"  Nodes: {list(g.node_ids)}")
    print(f"  Edges: {list(g.edge_ids)}")
    print(f"  Meta-node: {meta_node}")
    
    # Check if external edge still exists and what it connects to
    print(f"\nEdge connectivity:")
    for edge_id in g.edge_ids:
        # Get edge endpoints - need to check the edge structure
        print(f"  Edge {edge_id} still exists")
        # TODO: Check what nodes this edge connects

if __name__ == "__main__":
    test_edge_handling()
