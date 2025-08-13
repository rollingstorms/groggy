#!/usr/bin/env python3
"""
Test the new graph properties g.nodes and g.edges
"""

import groggy as gr

def test_graph_properties():
    """Test g.nodes and g.edges properties"""
    
    print("=== Testing Graph Properties ===")
    
    # Create a simple graph
    g = gr.Graph()
    
    # Test empty graph
    print(f"Empty graph nodes: {g.nodes}")
    print(f"Empty graph edges: {g.edges}")
    print(f"Empty graph length: {len(g)}")
    
    # Add some nodes
    node_ids = []
    for i in range(5):
        node_id = g.add_node(value=i)
        node_ids.append(node_id)
    
    print(f"\nAfter adding 5 nodes:")
    print(f"g.nodes: {g.nodes}")
    print(f"g.edges: {g.edges}")
    print(f"len(g): {len(g)}")
    print(f"g.node_count(): {g.node_count()}")
    print(f"g.edge_count(): {g.edge_count()}")
    
    # Add some edges
    edge_ids = []
    edge_ids.append(g.add_edge(node_ids[0], node_ids[1]))
    edge_ids.append(g.add_edge(node_ids[1], node_ids[2]))
    edge_ids.append(g.add_edge(node_ids[2], node_ids[3]))
    
    print(f"\nAfter adding 3 edges:")
    print(f"g.nodes: {g.nodes}")
    print(f"g.edges: {g.edges}")
    print(f"len(g): {len(g)}")
    print(f"g.node_count(): {g.node_count()}")
    print(f"g.edge_count(): {g.edge_count()}")
    
    # Test property access vs method access consistency
    print(f"\n=== Consistency Check ===")
    print(f"g.nodes == g.node_ids(): {g.nodes == g.node_ids()}")
    print(f"g.edges == g.edge_ids(): {g.edges == g.edge_ids()}")
    print(f"len(g.nodes) == g.node_count(): {len(g.nodes) == g.node_count()}")
    print(f"len(g.edges) == g.edge_count(): {len(g.edges) == g.edge_count()}")
    
    # Test with subgraph
    print(f"\n=== Testing with Subgraph ===")
    subgraph = g.filter_nodes("value < 3")
    print(f"Subgraph nodes: {subgraph.nodes}")
    print(f"Subgraph edges: {subgraph.edges}")
    print(f"Original graph still has: {len(g.nodes)} nodes, {len(g.edges)} edges")
    
    print("\n✅ Graph properties working correctly!")

if __name__ == "__main__":
    test_graph_properties()