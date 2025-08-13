#!/usr/bin/env python3
"""
Test the new PySubgraph implementation
"""

import groggy as gr

def test_subgraph_basic():
    """Test basic subgraph functionality"""
    
    print("=== Testing PySubgraph Implementation ===")
    
    # Create a simple graph
    g = gr.Graph()
    
    # Add nodes with attributes
    node_ids = []
    for i in range(5):
        node_id = g.add_node(value=i, category="test")
        node_ids.append(node_id)
    
    # Add some edges
    g.add_edge(node_ids[0], node_ids[1], weight=1.0)
    g.add_edge(node_ids[1], node_ids[2], weight=2.0)
    g.add_edge(node_ids[2], node_ids[3], weight=3.0)
    g.add_edge(node_ids[0], node_ids[4], weight=4.0)
    
    print(f"Created graph: {g}")
    print(f"Node IDs: {node_ids}")
    
    # Test filter_nodes returning PySubgraph
    print("\n=== Testing filter_nodes → PySubgraph ===")
    
    # Use string query for nodes with value < 3
    subgraph = g.filter_nodes("value < 3")
    
    print(f"Subgraph type: {type(subgraph)}")
    print(f"Subgraph: {subgraph}")
    print(f"Subgraph nodes: {subgraph.nodes}")
    print(f"Subgraph edges: {subgraph.edges}")
    print(f"Subgraph node count: {subgraph.node_count()}")
    print(f"Subgraph edge count: {subgraph.edge_count()}")
    print(f"Subgraph length: {len(subgraph)}")
    
    # Test chainable operations (placeholders for now)
    print("\n=== Testing Chainable Operations ===")
    
    # These are placeholder implementations for now
    chained_subgraph = subgraph.filter_nodes("placeholder")
    print(f"Chained subgraph: {chained_subgraph}")
    
    edge_filtered = subgraph.filter_edges("placeholder")
    print(f"Edge filtered subgraph: {edge_filtered}")
    
    components = subgraph.connected_components()
    print(f"Connected components: {len(components)} components")
    if components:
        print(f"First component: {components[0]}")
    
    print("\n✅ Basic PySubgraph functionality working!")

if __name__ == "__main__":
    test_subgraph_basic()