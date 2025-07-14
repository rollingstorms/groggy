#!/usr/bin/env python3

"""
Test the unified filtering implementation to ensure all methods work correctly.
"""

import groggy

def test_unified_filtering():
    """Test the unified filtering approach."""
    print("Testing unified filtering implementation...")
    
    # Create a graph
    graph = groggy.FastGraph(directed=False)
    
    # Add nodes with attributes
    graph.add_node("node1", {"age": 25, "name": "Alice", "active": True})
    graph.add_node("node2", {"age": 30, "name": "Bob", "active": False})
    graph.add_node("node3", {"age": 35, "name": "Charlie", "active": True})
    
    # Add edges with attributes
    graph.add_edge("node1", "node2", {"weight": 1.5, "type": "friend"})
    graph.add_edge("node2", "node3", {"weight": 2.0, "type": "colleague"})
    graph.add_edge("node1", "node3", {"weight": 0.8, "type": "friend"})
    
    print(f"Created graph with {graph.node_count()} nodes and {graph.edge_count()} edges")
    
    # Test node filtering by attributes
    print("\n=== Testing Node Filtering ===")
    
    # Filter by exact attribute match
    active_nodes = graph.filter_nodes_by_attributes({"active": True})
    print(f"Active nodes: {active_nodes}")
    assert set(active_nodes) == {"node1", "node3"}
    
    # Filter by numeric comparison
    young_nodes = graph.filter_nodes_by_numeric_comparison("age", "<", 30)
    print(f"Young nodes (age < 30): {young_nodes}")
    assert set(young_nodes) == {"node1"}
    
    # Filter by string comparison
    charlie_nodes = graph.filter_nodes_by_string_comparison("name", "==", "Charlie")
    print(f"Nodes named Charlie: {charlie_nodes}")
    assert set(charlie_nodes) == {"node3"}
    
    # Test multi-criteria filtering
    criteria_nodes = graph.filter_nodes_multi_criteria(
        exact_matches={"active": "true"},
        numeric_comparisons=[("age", ">=", 30)],
        string_comparisons=[]
    )
    print(f"Active nodes with age >= 30: {criteria_nodes}")
    assert set(criteria_nodes) == {"node3"}
    
    # Test sparse filtering
    sparse_nodes = graph.filter_nodes_by_attributes_sparse({"name": "Alice"})
    print(f"Sparse filter for Alice: {sparse_nodes}")
    assert set(sparse_nodes) == {"node1"}
    
    # Test edge filtering by attributes
    print("\n=== Testing Edge Filtering ===")
    
    # Filter by exact attribute match
    friend_edges = graph.filter_edges_by_attributes({"type": "friend"})
    print(f"Friend edges: {friend_edges}")
    assert len(friend_edges) == 2
    assert set(friend_edges) == {"node1->node2", "node1->node3"}
    
    # Filter by numeric comparison
    heavy_edges = graph.filter_edges_by_numeric_comparison("weight", ">", 1.0)
    print(f"Heavy edges (weight > 1.0): {heavy_edges}")
    assert len(heavy_edges) == 2
    
    # Filter by string comparison
    colleague_edges = graph.filter_edges_by_string_comparison("type", "==", "colleague")
    print(f"Colleague edges: {colleague_edges}")
    assert len(colleague_edges) == 1
    
    # Test multi-criteria edge filtering
    criteria_edges = graph.filter_edges_multi_criteria(
        exact_matches={"type": "friend"},
        numeric_comparisons=[("weight", "<", 1.0)],
        string_comparisons=[]
    )
    print(f"Friend edges with weight < 1.0: {criteria_edges}")
    assert len(criteria_edges) == 1
    
    print("\n✅ All unified filtering tests passed!")
    print("🎉 Code duplication successfully reduced - same filtering logic works for both nodes and edges!")

if __name__ == "__main__":
    test_unified_filtering()
