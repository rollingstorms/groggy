#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy

def test_bfs_timing():
    print("🚀 Testing BFS timing with debug markers...")
    
    # Create a small graph for testing
    g = groggy.Graph()
    
    # Add nodes using count (simpler approach)
    g.add_nodes(20)  # Add 20 nodes
    
    # Add edges to create a connected component  
    for i in range(19):
        g.add_edge(i, i+1)  # Connect node i to node i+1
    
    print(f"Graph has {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Test BFS - this should trigger our debug output
    print("\n🔍 Starting BFS test...")
    result = g.bfs(0, max_depth=5)
    print(f"BFS result: {len(result.nodes)} nodes")
    
    # Test view().bfs() for comparison
    print("\n🔍 Testing g.view().bfs() for comparison...")
    view_result = g.view().bfs(0, max_depth=5)
    print(f"View BFS result: {len(view_result.nodes)} nodes")

if __name__ == "__main__":
    test_bfs_timing()