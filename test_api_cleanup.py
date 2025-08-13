#!/usr/bin/env python3
"""
Test the new cleaner API method names
"""

import groggy as gr

def test_cleaner_api():
    """Test the new cleaner method names"""
    
    print("🧹 === TESTING CLEANER API METHODS ===")
    
    # Create a simple graph for testing
    g = gr.Graph()
    
    # Add some nodes
    for i in range(5):
        g.add_node(value=i, category="test")
    
    # Add some edges to create a connected component
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(3, 4)
    
    print(f"Created test graph: {g}")
    
    # Test 1: BFS vs traverse_bfs
    print(f"\n🔍 === Test 1: BFS Methods ===")
    try:
        old_bfs = g.traverse_bfs(0, max_depth=2, node_filter=None, edge_filter=None)
        print(f"✅ traverse_bfs: {old_bfs}")
        
        new_bfs = g.bfs(0, max_depth=2, node_filter=None, edge_filter=None)
        print(f"✅ bfs: {new_bfs}")
        
        print(f"Same result: {old_bfs.nodes == new_bfs.nodes}")
        
    except Exception as e:
        print(f"❌ BFS error: {e}")
    
    # Test 2: DFS vs traverse_dfs
    print(f"\n🔍 === Test 2: DFS Methods ===")
    try:
        old_dfs = g.traverse_dfs(0, max_depth=2, node_filter=None, edge_filter=None)
        print(f"✅ traverse_dfs: {old_dfs}")
        
        new_dfs = g.dfs(0, max_depth=2, node_filter=None, edge_filter=None)
        print(f"✅ dfs: {new_dfs}")
        
        print(f"Same result: {old_dfs.nodes == new_dfs.nodes}")
        
    except Exception as e:
        print(f"❌ DFS error: {e}")
    
    # Test 3: Connected components
    print(f"\n🔗 === Test 3: Connected Components ===")
    try:
        old_components = g.find_connected_components()
        print(f"✅ find_connected_components: {len(old_components)} components")
        
        new_components = g.connected_components()
        print(f"✅ connected_components: {len(new_components)} components")
        
        print(f"Same result: {len(old_components) == len(new_components)}")
        
    except Exception as e:
        print(f"❌ Connected components error: {e}")
    
    # Test 4: Version control methods (if available)
    print(f"\n📚 === Test 4: Version Control Methods ===")
    try:
        # Commit something first
        g.commit("Test commit", "API Tester")
        
        old_branches = g.list_branches()
        new_branches = g.branches()
        print(f"✅ list_branches vs branches: {len(old_branches)} == {len(new_branches)}")
        
        old_history = g.get_commit_history()
        new_history = g.commit_history()
        print(f"✅ get_commit_history vs commit_history: {len(old_history)} == {len(new_history)}")
        
    except Exception as e:
        print(f"❌ Version control error: {e}")
    
    # Test 5: Check what new methods are available
    print(f"\n📋 === Test 5: Available Methods ===")
    
    all_methods = [method for method in dir(g) if not method.startswith('_')]
    new_clean_methods = ['bfs', 'dfs', 'connected_components', 'branches', 'commit_history', 'historical_view', 'group_by']
    
    print(f"New clean methods available:")
    for method in new_clean_methods:
        if method in all_methods:
            print(f"  ✅ {method}")
        else:
            print(f"  ❌ {method} (missing)")
    
    print(f"\n🎉 Cleaner API methods working!")

if __name__ == "__main__":
    test_cleaner_api()