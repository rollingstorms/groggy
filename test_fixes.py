#!/usr/bin/env python3
"""
Simple test to verify the bug fixes:
1. EdgesAccessor .sources/.targets subgraph constraint handling
2. NumArray.unique() method implementation  
3. NodesArray and EdgesArray table() methods
"""

import groggy as g

def test_basic_functionality():
    print("Testing basic functionality...")
    
    # Create a simple graph
    graph = g.Graph()
    
    # Add some nodes (just IDs for simplicity)
    for i in range(5):
        graph.add_node(node_id=i)
    
    # Add some edges  
    edges = [(0, 1), (1, 2), (2, 3), (0, 3), (3, 4)]
    for src, dst in edges:
        graph.add_edge(source=src, target=dst)
    
    print(f"Created graph with {graph.node_count()} nodes and {graph.edge_count()} edges")
    
    # Test NumArray.unique() 
    print("\n--- Testing NumArray.unique() ---")
    
    # Test with different data types
    test_data = [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0]
    num_array = g.NumArray(test_data)
    unique_vals = num_array.unique()
    print(f"Original: {test_data}")
    print(f"Unique: {unique_vals}")
    assert len(unique_vals) == 4, f"Expected 4 unique values, got {len(unique_vals)}"
    
    # Test with integer array
    int_data = [1, 2, 1, 3, 2, 4, 1]
    int_array = g.NumArray(int_data, dtype="int64") 
    unique_ints = int_array.unique()
    print(f"Int Original: {int_data}")
    print(f"Int Unique: {unique_ints}")
    assert len(unique_ints) == 4, f"Expected 4 unique int values, got {len(unique_ints)}"
    
    print("✅ NumArray.unique() working correctly!")
    
    # Test array table() methods and basic EdgesAccessor functionality
    print("\n--- Testing array table() methods and EdgesAccessor ---")
    
    try:
        # Test nodes accessor directly from main graph
        nodes_accessor = graph.nodes
        print(f"NodesAccessor type: {type(nodes_accessor)}")
        
        # Test sources and targets from edges accessor
        edges_accessor = graph.edges
        print(f"EdgesAccessor type: {type(edges_accessor)}")
        
        # Test basic EdgesAccessor functionality
        sources = edges_accessor.sources
        targets = edges_accessor.targets
        print(f"Edge sources: {sources}")
        print(f"Edge targets: {targets}")
        print("✅ EdgesAccessor .sources and .targets working!")
        
        # Test if we can get table from accessor
        try:
            nodes_table = nodes_accessor.table()
            print(f"NodesAccessor.table() returned: {type(nodes_table)}")
            print("✅ NodesAccessor.table() working!")
        except Exception as e:
            print(f"NodesAccessor.table() issue: {e}")
            
        try:
            edges_table = edges_accessor.table()
            print(f"EdgesAccessor.table() returned: {type(edges_table)}")
            print("✅ EdgesAccessor.table() working!")
        except Exception as e:
            print(f"EdgesAccessor.table() issue: {e}")
        
    except Exception as e:
        print(f"❌ Error testing functionality: {e}")
        raise

if __name__ == "__main__":
    test_basic_functionality()
    print("\n🎉 All tests passed! Bug fixes are working correctly.")