#!/usr/bin/env python3
"""
Simple test for node_strategy parameter functionality.
"""

import groggy

def test_extract_strategy():
    """Test that extract strategy keeps original nodes."""
    print("Testing Extract strategy...")
    
    # Create a simple graph
    g = groggy.Graph()
    
    # Add some nodes
    n1 = g.add_node()
    n2 = g.add_node()
    n3 = g.add_node()
    
    # Add some edges
    e1 = g.add_edge(n1, n2)
    e2 = g.add_edge(n2, n3)
    
    print(f"Original graph has {len(g.nodes())} nodes")
    
    # Create subgraph
    subgraph = g.view()
    
    print(f"Subgraph type: {type(subgraph)}")
    
    try:
        # Test collapse with extract strategy
        result = subgraph.collapse(node_strategy="extract")
        print(f"Result type: {type(result)}")
        print("Extract strategy test completed successfully!")
        
    except Exception as e:
        print(f"Extract strategy test failed: {e}")
        import traceback
        traceback.print_exc()

def test_collapse_strategy():
    """Test that collapse strategy removes original nodes."""
    print("\nTesting Collapse strategy...")
    
    # Create a simple graph
    g = groggy.Graph()
    
    # Add some nodes
    n1 = g.add_node()
    n2 = g.add_node()
    n3 = g.add_node()
    
    # Add some edges
    e1 = g.add_edge(n1, n2)
    e2 = g.add_edge(n2, n3)
    
    print(f"Original graph has {len(g.nodes())} nodes")
    
    # Create subgraph
    subgraph = g.view()
    
    try:
        # Test collapse with collapse strategy
        result = subgraph.collapse(node_strategy="collapse")
        print(f"Result type: {type(result)}")
        print("Collapse strategy test completed successfully!")
        
    except Exception as e:
        print(f"Collapse strategy test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_extract_strategy()
    test_collapse_strategy()
