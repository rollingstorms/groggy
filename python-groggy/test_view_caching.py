#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy
import time

def test_view_caching_potential():
    print("🚀 Testing view caching potential...")
    
    # Create benchmark-scale graph
    g = groggy.Graph()
    g.add_nodes(50000)
    edge_pairs = [(i, i+1) for i in range(49999)]
    g.add_edges(edge_pairs)
    
    print(f"Graph created: {g.node_count()} nodes, {g.edge_count()} edges")
    
    print("\n=== CURRENT BEHAVIOR (No Caching) ===")
    
    # Test multiple view() calls to see repeated overhead
    times = []
    for i in range(3):
        print(f"\n📊 View call #{i+1}:")
        start_time = time.time()
        view = g.view()
        view_time = time.time() - start_time
        times.append(view_time * 1000)
        print(f"g.view() took: {view_time*1000:.1f}ms")
        
        # Quick BFS on the view
        start_time = time.time()
        result = view.bfs(0, max_depth=3)
        bfs_time = time.time() - start_time
        print(f"view.bfs() took: {bfs_time*1000:.1f}ms")
        print(f"Total: {(view_time + bfs_time)*1000:.1f}ms")
    
    print(f"\n📈 View() timing consistency:")
    for i, t in enumerate(times):
        print(f"  Call {i+1}: {t:.1f}ms")
    
    print(f"\n💡 With caching, calls 2-3 would be ~0ms!")
    print(f"💡 This would make BFS: ~6ms → ~0.1ms after warmup")

if __name__ == "__main__":
    test_view_caching_potential()