#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy

def test_cache_sequence():
    print("🚀 Testing cache sequence across multiple algorithm calls...")
    
    # Create graph with bulk operations
    g = groggy.Graph()
    
    print("Adding nodes and edges...")
    g.add_nodes(10000)  # Smaller graph for cleaner output
    
    # Add edges in bulk to avoid individual cache invalidations
    edge_pairs = [(i, i+1) for i in range(9999)]
    g.add_edges(edge_pairs)
    
    print(f"Graph has {g.node_count()} nodes and {g.edge_count()} edges")
    
    print("\n=== ALGORITHM SEQUENCE TEST ===")
    
    print("\n1️⃣ First BFS call (should be CACHE MISS):")
    bfs1 = g.view().bfs(0, max_depth=5)
    print(f"BFS result: {len(bfs1.nodes)} nodes")
    
    print("\n2️⃣ Second BFS call (should be CACHE HIT):")
    bfs2 = g.view().bfs(100, max_depth=5)
    print(f"BFS result: {len(bfs2.nodes)} nodes")
    
    print("\n3️⃣ DFS call (should be CACHE HIT):")
    dfs1 = g.view().dfs(0, max_depth=5)
    print(f"DFS result: {len(dfs1.nodes)} nodes")
    
    print("\n4️⃣ Connected Components call (should be CACHE HIT):")
    cc1 = g.view().connected_components()
    print(f"Connected components: {len(cc1)} components")

if __name__ == "__main__":
    test_cache_sequence()