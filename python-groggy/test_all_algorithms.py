#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy
import time

def test_all_algorithms():
    print("🚀 Testing timing for ALL algorithms (50,000 nodes)...")
    
    # Create a larger graph that matches benchmark size
    g = groggy.Graph()
    
    # Add 50,000 nodes 
    g.add_nodes(50000)
    
    # Add edges to create connected components
    for i in range(49999):
        g.add_edge(i, i+1)
    
    print(f"Graph has {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Test BFS
    print("\n🔍 Testing BFS...")
    start_time = time.time()
    bfs_result = g.view().bfs(0, max_depth=10)
    bfs_time = time.time() - start_time
    print(f"BFS: {bfs_time*1000:.1f}ms, {len(bfs_result.nodes)} nodes")
    
    # Test DFS
    print("\n🔍 Testing DFS...")
    start_time = time.time()
    dfs_result = g.view().dfs(0, max_depth=10)
    dfs_time = time.time() - start_time
    print(f"DFS: {dfs_time*1000:.1f}ms, {len(dfs_result.nodes)} nodes")
    
    # Test connected_components
    print("\n🔍 Testing connected_components...")
    start_time = time.time()
    cc_result = g.view().connected_components()
    cc_time = time.time() - start_time
    print(f"Connected components: {cc_time*1000:.1f}ms, {len(cc_result)} components")
    
    # Test second round to see cached performance
    print("\n🔍 Testing second round (cached)...")
    
    start_time = time.time()
    g.view().bfs(0, max_depth=10)
    print(f"BFS (2nd): {(time.time() - start_time)*1000:.1f}ms")
    
    start_time = time.time()
    g.view().dfs(0, max_depth=10)
    print(f"DFS (2nd): {(time.time() - start_time)*1000:.1f}ms")
    
    start_time = time.time()
    g.view().connected_components()
    print(f"Connected components (2nd): {(time.time() - start_time)*1000:.1f}ms")

if __name__ == "__main__":
    test_all_algorithms()