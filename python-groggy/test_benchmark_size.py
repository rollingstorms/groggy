#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy
import time

def test_benchmark_scale():
    print("🚀 Testing DFS timing at benchmark scale (50,000 nodes)...")
    
    # Create graph exactly like benchmark
    g = groggy.Graph()
    
    print("Creating benchmark-scale graph...")
    start_time = time.time()
    g.add_nodes(50000)
    
    # Use bulk edge addition like benchmark
    edge_pairs = [(i, (i + 1) % 50000) for i in range(50000)]  # Circular graph
    g.add_edges(edge_pairs)
    
    creation_time = time.time() - start_time
    print(f"Graph created in {creation_time:.2f}s: {g.node_count()} nodes, {g.edge_count()} edges")
    
    print("\n=== BENCHMARK-SCALE TIMING ===")
    
    print("\n1️⃣ BFS call:")
    start_time = time.time()
    bfs_result = g.view().bfs(0, max_depth=3)
    bfs_time = time.time() - start_time
    print(f"BFS: {bfs_time*1000:.1f}ms, {len(bfs_result.nodes)} nodes")
    
    print("\n2️⃣ DFS call:")
    start_time = time.time()
    dfs_result = g.view().dfs(0, max_depth=3)  
    dfs_time = time.time() - start_time
    print(f"DFS: {dfs_time*1000:.1f}ms, {len(dfs_result.nodes)} nodes")
    
    print("\n3️⃣ Second DFS call:")
    start_time = time.time()
    dfs_result2 = g.view().dfs(100, max_depth=3)
    dfs_time2 = time.time() - start_time  
    print(f"DFS (2nd): {dfs_time2*1000:.1f}ms, {len(dfs_result2.nodes)} nodes")

if __name__ == "__main__":
    test_benchmark_scale()