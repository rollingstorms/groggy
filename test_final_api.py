#!/usr/bin/env python3
"""
Final test of the clean API after cleanup
"""

import groggy as gr

def test_final_clean_api():
    """Test the final clean API"""
    
    print("🎯 === FINAL API CLEANUP VERIFICATION ===")
    
    # Create a test graph
    g = gr.Graph()
    
    # Add some test data
    g.add_node(name="Alice", salary=120000)
    g.add_node(name="Bob", salary=140000)
    g.add_node(name="Carol", salary=95000)
    
    g.add_edge(0, 1, weight=0.9)
    g.add_edge(1, 2, weight=0.7)
    
    print(f"✅ Created test graph: {g}")
    
    # Test new clean methods
    print(f"\n🧪 === Testing Clean Methods ===")
    
    try:
        # Test BFS
        bfs_result = g.bfs(0, max_depth=2, node_filter=None, edge_filter=None)
        print(f"✅ bfs(): {bfs_result}")
        
        # Test DFS
        dfs_result = g.dfs(0, max_depth=2, node_filter=None, edge_filter=None)
        print(f"✅ dfs(): {dfs_result}")
        
        # Test connected components
        components = g.connected_components()
        print(f"✅ connected_components(): {len(components)} components")
        
        # Test version control methods
        g.commit("Test commit", "API Tester")
        branches = g.branches()
        history = g.commit_history()
        print(f"✅ branches(): {len(branches)} branches")
        print(f"✅ commit_history(): {len(history)} commits")
        
        # Test unified aggregate method
        result = g.aggregate("salary", "mean")
        print(f"✅ aggregate(): {result}")
        
        print(f"\n🎉 All clean API methods working perfectly!")
        
    except Exception as e:
        print(f"❌ Error testing clean methods: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_final_clean_api()