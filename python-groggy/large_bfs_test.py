#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy
import time

def test_large_bfs_timing():
    print("🚀 Testing BFS timing on larger graph (50,000 nodes)...")
    
    # Create a larger graph that matches benchmark size
    g = groggy.Graph()
    
    # Add 50,000 nodes 
    print("Adding 50,000 nodes...")
    start_time = time.time()
    g.add_nodes(50000)
    print(f"Nodes added in {time.time() - start_time:.3f}s")
    
    # Add edges to create connected components
    print("Adding 49,999 edges...")
    start_time = time.time()
    for i in range(49999):
        g.add_edge(i, i+1)
    print(f"Edges added in {time.time() - start_time:.3f}s")
    
    print(f"Graph has {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Test BFS on large graph - this should show the real bottleneck
    print("\n🔍 Starting BFS test on large graph...")
    start_time = time.time()
    result = g.view().bfs(0, max_depth=10)  # Use view().bfs() since g.bfs() doesn't exist
    bfs_time = time.time() - start_time
    print(f"BFS completed in {bfs_time*1000:.1f}ms")
    print(f"BFS result: {len(result.nodes)} nodes")
    
    # Compare with multiple runs for consistency
    print("\n🔍 Running multiple BFS tests for consistency...")
    times = []
    for i in range(5):
        start_time = time.time()
        result = g.view().bfs(0, max_depth=10)
        times.append((time.time() - start_time) * 1000)
        print(f"Run {i+1}: {times[-1]:.1f}ms")
    
    print(f"Average BFS time: {sum(times)/len(times):.1f}ms")
    print(f"This should match the benchmark results!")

if __name__ == "__main__":
    test_large_bfs_timing()