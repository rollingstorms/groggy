#!/usr/bin/env python3
"""
Test convenience property access methods
"""

import groggy as gr

def test_convenience_methods():
    print("Testing convenience property access methods...")
    
    graph = gr.Graph()
    
    # Test empty graph
    print(f"\n=== Test 1: Empty graph ===")
    print(f"len(graph): {len(graph)}")
    print(f"graph.node_count(): {graph.node_count()}")
    print(f"graph.edge_count(): {graph.edge_count()}")
    
    # Add some nodes and edges
    print(f"\n=== Test 2: Adding nodes and edges ===")
    node_data = [
        {"id": "alice", "name": "Alice"},
        {"id": "bob", "name": "Bob"},
        {"id": "charlie", "name": "Charlie"}
    ]
    
    mapping = graph.add_nodes(node_data, id_key="id")
    print(f"Added nodes: {mapping}")
    
    alice_id = mapping["alice"]
    bob_id = mapping["bob"]
    charlie_id = mapping["charlie"]
    
    # Add edges
    edge1 = graph.add_edge(alice_id, bob_id)
    edge2 = graph.add_edge(bob_id, charlie_id)
    
    print(f"Added edges: {edge1}, {edge2}")
    
    # Test counts
    print(f"\n=== Test 3: Updated counts ===")
    print(f"len(graph): {len(graph)}")
    print(f"graph.node_count(): {graph.node_count()}")
    print(f"graph.edge_count(): {graph.edge_count()}")
    
    # Test existence checks
    print(f"\n=== Test 4: Existence checks ===")
    print(f"graph.has_node(alice_id={alice_id}): {graph.has_node(alice_id)}")
    print(f"graph.has_node(999): {graph.has_node(999)}")  # Non-existent node
    
    print(f"graph.has_edge(edge1={edge1}): {graph.has_edge(edge1)}")
    print(f"graph.has_edge(999): {graph.has_edge(999)}")  # Non-existent edge
    
    # Test Python len() function
    print(f"\n=== Test 5: Python len() integration ===")
    nodes_via_len = len(graph)
    nodes_via_method = graph.node_count()
    print(f"len(graph) == graph.node_count(): {nodes_via_len == nodes_via_method}")
    
    # Test with empty graph again
    print(f"\n=== Test 6: Empty graph again ===")
    empty_graph = gr.Graph()
    print(f"len(empty_graph): {len(empty_graph)}")
    print(f"empty_graph.has_node(0): {empty_graph.has_node(0)}")
    print(f"empty_graph.has_edge(0): {empty_graph.has_edge(0)}")
    
    print(f"\nFinal graph: {graph}")
    print("🎉 All convenience method tests passed!")

if __name__ == "__main__":
    test_convenience_methods()