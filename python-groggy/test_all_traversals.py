#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy
import time

def test_all_traversal_performance():
    print("🚀 Testing all basic traversal methods performance...")
    
    # Create a test graph with more complex structure
    g = groggy.Graph()
    
    # Create a more interesting graph structure for testing
    node_ids = []
    for i in range(20):
        node_id = g.add_node()
        node_ids.append(node_id)
        g.set_node_attr(node_id, 'value', i)
    
    # Create a connected graph with multiple paths
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Main path
        (0, 5), (5, 6), (6, 4),          # Alternative path  
        (4, 7), (7, 8), (8, 9),          # Extension
        (1, 10), (10, 11), (11, 3),      # Another alternative
        (9, 12), (12, 13), (13, 14),     # Branch
        (14, 15), (15, 16), (16, 17),    # Extension
        (17, 18), (18, 19)               # Final branch
    ]
    
    for source_idx, target_idx in edges:
        g.add_edge(node_ids[source_idx], node_ids[target_idx])
    
    print(f"Created test graph with {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Prime the cache
    print("\n=== Priming snapshot cache ===")
    start = time.time()
    _ = g.bfs(node_ids[0], max_depth=1)
    prime_time = time.time() - start
    print(f"Cache priming took: {prime_time*1000:.2f}ms")
    
    # Test all basic traversal methods
    print("\n=== Testing Basic Traversal Methods ===")
    
    # Test BFS
    start = time.time()
    bfs_result = g.bfs(node_ids[0], max_depth=3)
    bfs_time = time.time() - start
    print(f"BFS: {bfs_time*1000:.2f}ms -> {bfs_result}")
    
    # Test DFS  
    start = time.time()
    dfs_result = g.dfs(node_ids[0], max_depth=3)
    dfs_time = time.time() - start
    print(f"DFS: {dfs_time*1000:.2f}ms -> {dfs_result}")
    
    # Test shortest_path_fast
    start = time.time()
    path_result = g.shortest_path_fast(node_ids[0], node_ids[4])
    path_time = time.time() - start
    print(f"Shortest path: {path_time*1000:.2f}ms -> {path_result}")
    
    # Test has_path
    start = time.time()
    has_path1 = g.has_path(node_ids[0], node_ids[4])  # Should be True
    has_path_time1 = time.time() - start
    
    start = time.time()
    has_path2 = g.has_path(node_ids[0], node_ids[19])  # Should be True (connected)
    has_path_time2 = time.time() - start
    
    print(f"Has path (0->4): {has_path_time1*1000:.2f}ms -> {has_path1}")
    print(f"Has path (0->19): {has_path_time2*1000:.2f}ms -> {has_path2}")
    
    # Test multiple calls to show caching benefits
    print("\n=== Testing Repeated Calls (Cache Benefits) ===")
    
    start = time.time()
    bfs_result2 = g.bfs(node_ids[5], max_depth=2)
    bfs_time2 = time.time() - start
    print(f"BFS (2nd call): {bfs_time2*1000:.2f}ms -> {bfs_result2}")
    
    start = time.time()
    path_result2 = g.shortest_path_fast(node_ids[1], node_ids[8])
    path_time2 = time.time() - start
    print(f"Shortest path (2nd call): {path_time2*1000:.2f}ms -> {path_result2}")
    
    # Performance summary
    print(f"\n🎯 Performance Summary:")
    print(f"BFS (1st): {bfs_time*1000:.2f}ms, (2nd): {bfs_time2*1000:.2f}ms")
    print(f"DFS: {dfs_time*1000:.2f}ms")
    print(f"Shortest path (1st): {path_time*1000:.2f}ms, (2nd): {path_time2*1000:.2f}ms")
    print(f"Has path: {has_path_time1*1000:.2f}ms, {has_path_time2*1000:.2f}ms")
    
    # Verify PathResult functionality
    if path_result and hasattr(path_result, 'node_count'):
        print(f"\n✅ PathResult working: {path_result.node_count} nodes, {path_result.edge_count} edges")
    
    avg_traversal_time = (bfs_time + dfs_time + path_time + has_path_time1) / 4
    print(f"📊 Average traversal time: {avg_traversal_time*1000:.2f}ms")
    
    if avg_traversal_time < 0.002:  # Less than 2ms
        print("🚀 EXCELLENT: All traversals are sub-2ms!")
    elif avg_traversal_time < 0.005:  # Less than 5ms  
        print("✅ GOOD: All traversals are sub-5ms")
    else:
        print("⚠️  Could be optimized further")

if __name__ == "__main__":
    test_all_traversal_performance()