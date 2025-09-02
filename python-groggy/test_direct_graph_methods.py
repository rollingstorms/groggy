#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy
import time

def test_direct_graph_performance():
    print("🚀 Testing direct Graph method performance...")
    
    # Create a test graph
    g = groggy.Graph()
    
    # Add test nodes with attributes  
    node_ids = []
    for i in range(1000):
        node_id = g.add_node()
        node_ids.append(node_id)
        g.set_node_attr(node_id, 'value', i)
        g.set_node_attr(node_id, 'category', 'even' if i % 2 == 0 else 'odd')
    
    # Add edges
    for i in range(999):
        g.add_edge(node_ids[i], node_ids[i + 1])
    
    print(f"Created test graph with {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Prime the snapshot cache with a small operation first
    print("\n=== Priming snapshot cache ===")
    start = time.time()
    _ = g.bfs(node_ids[0], max_depth=1)  # Small operation to build snapshot cache
    prime_time = time.time() - start
    print(f"Cache priming took: {prime_time*1000:.2f}ms")
    
    # Test DIRECT g.filter_nodes() performance (should be fast now)
    print("\n=== Testing DIRECT g.filter_nodes() (after cache prime) ===")
    
    # First call (direct method - should be fast now that cache is primed)
    start = time.time()
    filtered1 = g.filter_nodes("value > 500")
    first_time = time.time() - start
    print(f"First filter_nodes call: {first_time*1000:.2f}ms")
    print(f"Result: {filtered1.node_count()} nodes, {filtered1.edge_count()} edges")
    
    # Second call (direct method - should be fast)  
    start = time.time()
    filtered2 = g.filter_nodes("value < 100")
    second_time = time.time() - start
    print(f"Second filter_nodes call: {second_time*1000:.2f}ms")
    print(f"Result: {filtered2.node_count()} nodes, {filtered2.edge_count()} edges")
    
    # Test DIRECT BFS/DFS performance (should return PathResult)
    print("\n=== Testing DIRECT g.bfs() and g.dfs() ===")
    
    start = time.time()
    bfs_result = g.bfs(node_ids[0], max_depth=3)
    bfs_time = time.time() - start
    print(f"BFS call: {bfs_time*1000:.2f}ms")
    print(f"BFS result type: {type(bfs_result)}")
    print(f"BFS result: {bfs_result}")
    
    start = time.time()
    dfs_result = g.dfs(node_ids[0], max_depth=3)
    dfs_time = time.time() - start
    print(f"DFS call: {dfs_time*1000:.2f}ms")
    print(f"DFS result type: {type(dfs_result)}")
    print(f"DFS result: {dfs_result}")
    
    # Compare with cached view approach
    print("\n=== Comparing with cached view approach ===")
    
    start = time.time()
    view_filtered = g.view().filter_nodes("value > 750")
    view_time = time.time() - start
    print(f"g.view().filter_nodes(): {view_time*1000:.2f}ms")
    
    print(f"\n🎯 Performance Comparison:")
    print(f"Direct g.filter_nodes(): {first_time*1000:.2f}ms")
    print(f"Cached g.view().filter_nodes(): {view_time*1000:.2f}ms")
    
    if first_time < view_time:
        speedup = view_time / first_time
        print(f"✅ Direct method is {speedup:.1f}x FASTER!")
    else:
        slowdown = first_time / view_time
        print(f"❌ Direct method is {slowdown:.1f}x slower")

if __name__ == "__main__":
    test_direct_graph_performance()