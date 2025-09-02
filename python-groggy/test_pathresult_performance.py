#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy
import time

def test_pathresult_performance():
    print("🚀 Testing PathResult BFS/DFS performance...")
    
    # Create a test graph
    g = groggy.Graph()
    
    # Add nodes and edges for testing
    node_ids = []
    for i in range(100):  # Smaller test for clarity
        node_id = g.add_node()
        node_ids.append(node_id)
    
    for i in range(99):
        g.add_edge(node_ids[i], node_ids[i + 1])
    
    print(f"Created test graph with {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Test BFS performance directly on graph view
    print("\n=== Testing Graph View BFS Performance ===")
    
    # Time the view creation (this was the bottleneck before)
    start = time.time()
    view = g.view()
    view_time = time.time() - start
    print(f"View creation: {view_time*1000:.2f}ms")
    
    # Time BFS (should return PathResult now)
    start = time.time()
    bfs_result = view.bfs(node_ids[0], max_depth=3)
    bfs_time = time.time() - start
    
    print(f"BFS took: {bfs_time*1000:.2f}ms")
    print(f"BFS result type: {type(bfs_result)}")
    print(f"BFS result: {bfs_result}")
    
    # Test PathResult methods
    try:
        print(f"BFS found {bfs_result.node_count} nodes")
        print(f"BFS found {bfs_result.edge_count} edges") 
        print(f"BFS result type: {bfs_result.result_type}")
    except Exception as e:
        print(f"PathResult method test error: {e}")
    
    # Test DFS performance
    print("\n=== Testing Graph View DFS Performance ===")
    
    start = time.time()
    dfs_result = view.dfs(node_ids[0], max_depth=3)
    dfs_time = time.time() - start
    
    print(f"DFS took: {dfs_time*1000:.2f}ms")
    print(f"DFS result type: {type(dfs_result)}")
    print(f"DFS result: {dfs_result}")
    
    # Test PathResult methods  
    try:
        print(f"DFS found {dfs_result.node_count} nodes")
        print(f"DFS found {dfs_result.edge_count} edges")
        print(f"DFS result type: {dfs_result.result_type}")
    except Exception as e:
        print(f"PathResult method test error: {e}")

    # Test performance comparison
    print("\n=== Performance Summary ===")
    print(f"View creation: {view_time*1000:.2f}ms")
    print(f"BFS execution: {bfs_time*1000:.2f}ms")
    print(f"DFS execution: {dfs_time*1000:.2f}ms")
    print(f"Total BFS (view + bfs): {(view_time + bfs_time)*1000:.2f}ms")
    print(f"Total DFS (view + dfs): {(view_time + dfs_time)*1000:.2f}ms")

if __name__ == "__main__":
    test_pathresult_performance()