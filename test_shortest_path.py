#!/usr/bin/env python3
"""
Test the shortest_path method with Subgraph return and inplace support
"""

import groggy as gr

def test_shortest_path():
    """Test shortest_path method"""
    
    print("🛤️ === TESTING SHORTEST PATH ===")
    
    # Create a test graph with a clear path
    g = gr.Graph()
    
    # Create a linear path: 0 -> 1 -> 2 -> 3 -> 4
    for i in range(5):
        g.add_node(name=f"Node_{i}")
    
    for i in range(4):
        g.add_edge(i, i+1, weight=1.0)
    
    # Add a branch: 2 -> 5
    g.add_node(name="Node_5")
    g.add_edge(2, 5, weight=2.0)
    
    print(f"✅ Created test graph: {g}")
    print(f"   Nodes: {g.node_count()}, Edges: {g.edge_count()}")
    
    # Test 1: Basic shortest path (without inplace)
    print(f"\n🔍 === Test 1: Basic Shortest Path ===")
    try:
        path = g.shortest_path(0, 4)
        if path:
            print(f"✅ shortest_path(0, 4) returned: {path}")
            print(f"   Type: {type(path)}")
            print(f"   Nodes in path: {len(path.nodes)}")
            print(f"   Edges in path: {len(path.edges)}")
        else:
            print(f"❌ No path found between 0 and 4")
        
    except Exception as e:
        print(f"❌ Basic shortest path error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Shortest path with inplace attribute setting
    print(f"\n📝 === Test 2: Shortest Path with inplace ===")
    try:
        path_inplace = g.shortest_path(0, 3, inplace=True, attr_name="path_distance")
        if path_inplace:
            print(f"✅ shortest_path(0, 3, inplace=True) returned: {path_inplace}")
            
            # Verify inplace worked
            for i in range(4):  # Path should be 0->1->2->3
                distance = g.nodes[i]["path_distance"]
                print(f"   Node {i} path_distance: {distance}")
        else:
            print(f"❌ No path found between 0 and 3")
        
    except Exception as e:
        print(f"❌ Inplace shortest path error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: No path case
    print(f"\n🚫 === Test 3: No Path Case ===")
    try:
        # Add isolated node
        g.add_node(name="Isolated")
        isolated_id = g.node_count() - 1
        
        no_path = g.shortest_path(0, isolated_id)
        if no_path is None:
            print(f"✅ No path found between 0 and {isolated_id} (expected)")
        else:
            print(f"❌ Unexpected path found: {no_path}")
        
    except Exception as e:
        print(f"❌ No path test error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Weighted path
    print(f"\n⚖️ === Test 4: Weighted Path ===")
    try:
        # This should find the path using weight attribute
        weighted_path = g.shortest_path(0, 2, weight_attribute="weight")
        if weighted_path:
            print(f"✅ Weighted shortest_path(0, 2) returned: {weighted_path}")
            print(f"   Path nodes: {len(weighted_path.nodes)}")
        else:
            print(f"❌ No weighted path found")
        
    except Exception as e:
        print(f"❌ Weighted path error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Shortest path testing complete!")

if __name__ == "__main__":
    test_shortest_path()