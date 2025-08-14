#!/usr/bin/env python3
"""
Test the updated algorithm methods to verify they return Subgraph objects
"""

import groggy as gr

def test_algorithm_returns():
    """Test that algorithms return Subgraph objects with inplace support"""
    
    print("🧪 === TESTING ALGORITHM RETURN TYPES ===")
    
    # Create a test graph with multiple components
    g = gr.Graph()
    
    # Component 1
    g.add_node(name="Alice", age=30)
    g.add_node(name="Bob", age=25)  
    g.add_edge(0, 1, weight=0.8)
    
    # Component 2
    g.add_node(name="Carol", age=35)
    g.add_node(name="Dave", age=28)
    g.add_edge(2, 3, weight=0.9)
    
    # Component 3 (single node)
    g.add_node(name="Eve", age=32)
    
    print(f"✅ Created test graph: {g}")
    print(f"   Nodes: {g.node_count()}, Edges: {g.edge_count()}")
    
    # Test 1: Connected Components
    print(f"\n🔗 === Test 1: Connected Components ===")
    try:
        # Test without inplace
        components = g.connected_components()
        print(f"✅ connected_components() returned: {len(components)} components")
        print(f"   Type: {type(components)}")
        if components:
            print(f"   First component type: {type(components[0])}")
            print(f"   First component: {components[0]}")
        
        # Test with inplace
        components_inplace = g.connected_components(inplace=True, attr_name="component_id")
        print(f"✅ connected_components(inplace=True) returned: {len(components_inplace)} components")
        
        # Verify inplace worked
        comp_id = g.nodes[0]["component_id"]
        print(f"✅ Component ID set: node 0 has component_id={comp_id}")
        
    except Exception as e:
        print(f"❌ Connected components error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: BFS
    print(f"\n🌐 === Test 2: BFS ===")
    try:
        # Test without inplace
        bfs_result = g.bfs(0, max_depth=2)
        print(f"✅ bfs() returned: {bfs_result}")
        print(f"   Type: {type(bfs_result)}")
        
        # Test with inplace
        bfs_inplace = g.bfs(0, max_depth=2, inplace=True, attr_name="bfs_distance")
        print(f"✅ bfs(inplace=True) returned: {bfs_inplace}")
        
        # Verify inplace worked
        bfs_dist = g.nodes[0]["bfs_distance"]
        print(f"✅ BFS distance set: node 0 has bfs_distance={bfs_dist}")
        
    except Exception as e:
        print(f"❌ BFS error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: DFS
    print(f"\n🌳 === Test 3: DFS ===")
    try:
        # Test without inplace
        dfs_result = g.dfs(0, max_depth=2)
        print(f"✅ dfs() returned: {dfs_result}")
        print(f"   Type: {type(dfs_result)}")
        
        # Test with inplace (for both node and edge attributes)
        dfs_inplace = g.dfs(0, max_depth=2, inplace=True, node_attr="dfs_visited", edge_attr="dfs_traversed")
        print(f"✅ dfs(inplace=True) returned: {dfs_inplace}")
        
        # Verify inplace worked
        dfs_visited = g.nodes[0]["dfs_visited"]
        print(f"✅ DFS visited set: node 0 has dfs_visited={dfs_visited}")
        
    except Exception as e:
        print(f"❌ DFS error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Algorithm return type testing complete!")

if __name__ == "__main__":
    test_algorithm_returns()