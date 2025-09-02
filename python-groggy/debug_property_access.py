#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy
import time

def debug_property_access():
    print("🚀 Testing when properties are accessed...")
    
    # Create benchmark-scale graph
    g = groggy.Graph()
    g.add_nodes(50000)
    edge_pairs = [(i, i+1) for i in range(49999)]
    g.add_edges(edge_pairs)
    
    print("\n=== STEP BY STEP ANALYSIS ===")
    
    print("\n1️⃣ Calling g.view().bfs() (should trigger core BFS + PySubgraph creation):")
    start_time = time.time()
    result = g.view().bfs(0, max_depth=3)
    bfs_call_time = time.time() - start_time
    print(f"BFS call took: {bfs_call_time*1000:.1f}ms")
    
    print(f"\n2️⃣ Accessing result.nodes (should trigger nodes property getter):")
    start_time = time.time()
    nodes = result.nodes
    nodes_access_time = time.time() - start_time
    print(f"Nodes property access took: {nodes_access_time*1000:.1f}ms")
    
    print(f"\n3️⃣ Calling len(nodes) (should be fast):")
    start_time = time.time()
    node_count = len(nodes)
    len_call_time = time.time() - start_time
    print(f"len(nodes) took: {len_call_time*1000:.1f}ms, count: {node_count}")
    
    print(f"\n4️⃣ Total time breakdown:")
    total_time = bfs_call_time + nodes_access_time + len_call_time
    print(f"BFS call: {bfs_call_time*1000:.1f}ms ({bfs_call_time/total_time*100:.1f}%)")
    print(f"Nodes access: {nodes_access_time*1000:.1f}ms ({nodes_access_time/total_time*100:.1f}%)")
    print(f"len() call: {len_call_time*1000:.1f}ms ({len_call_time/total_time*100:.1f}%)")
    print(f"Total: {total_time*1000:.1f}ms")

if __name__ == "__main__":
    debug_property_access()