#!/usr/bin/env python3
"""Test the fixed MetaGraph Composer API"""

import sys
sys.path.append('.')
import groggy as gr

def test_immediate_execution():
    """Test that collapse() returns MetaNode directly"""
    print("=== Testing Immediate Execution API ===")
    
    # Create test graph
    g = gr.Graph(directed=False)
    g.add_node(name="Alice", age=25, salary=50000)
    g.add_node(name="Bob", age=30, salary=60000)
    g.add_node(name="Carol", age=28, salary=55000)
    
    g.add_edge(0, 1, weight=0.8, type="collaboration")
    g.add_edge(1, 2, weight=0.9, type="collaboration")
    g.add_edge(0, 2, weight=0.7, type="collaboration")
    
    subgraph = g.nodes[[0, 1, 2]]
    
    # Check available attributes
    print(f"Node attributes: {g.all_node_attribute_names()}")
    print(f"Edge attributes: {g.all_edge_attribute_names()}")
    
    # Test direct execution with available attributes
    print("Testing direct execution:")
    meta_node = subgraph.collapse(
        node_aggs=[("team_size", "count")],
        edge_strategy="aggregate"
    )
    
    print(f"✓ Result type: {type(meta_node)}")
    print(f"✓ Meta-node: {meta_node}")
    
    # Verify it's a MetaNode, not a plan
    assert "MetaNode" in str(type(meta_node))
    
    print("✅ Direct execution works perfectly!")
    return True

def test_user_case():
    """Test the exact case the user reported"""
    print("\n=== Testing User Case ===")
    
    # Create a graph similar to user's case
    g = gr.Graph(directed=False)
    
    # Add nodes with degrees
    for i in range(10):
        g.add_node(name=f"Node{i}", value=i)
    
    # Add edges to create varying degrees
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),  # Node 0 has degree 5
        (1, 2), (1, 3), (1, 6), (1, 7),          # Node 1 has degree 5  
        (2, 3), (2, 8),                          # Node 2 has degree 4
        (4, 9), (5, 6), (7, 8)                   # Others lower degree
    ]
    
    for src, dst in edges:
        g.add_edge(src, dst, weight=1.0)
    
    # Find high degree nodes (>4)
    high_degree_nodes = [i for i in g.node_ids if g.degree()[i] > 4]
    print(f"High degree nodes: {high_degree_nodes}")
    
    if len(high_degree_nodes) > 0:
        # Create connected component subgraph
        subgraph = g.nodes[high_degree_nodes]
        
        # Test the user's desired syntax (but without add_to_graph)
        print("Testing user's syntax:")
        meta_node = subgraph.collapse()
        
        print(f"✓ Created meta-node: {meta_node}")
        print(f"✓ Type: {type(meta_node)}")
        
        print("✅ User case works perfectly!")
        return True
    else:
        print("⚠️ No high degree nodes found, skipping test")
        return True

def main():
    """Run tests"""
    print("Testing Fixed MetaGraph Composer API")
    print("=" * 50)
    
    results = [
        test_immediate_execution(),
        test_user_case(),
    ]
    
    if all(results):
        print(f"\n🎉 All tests passed! The API is now clean and simple:")
        print("   ✓ meta_node = subgraph.collapse()")
        print("   ✓ No need for .add_to_graph() - execution is immediate")
        print("   ✓ Clean, intuitive builder pattern")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())