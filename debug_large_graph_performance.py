#!/usr/bin/env python3
"""
Debug Performance with Larger Graphs 🐌⚡
"""

import groggy
import time

def test_large_graph_performance():
    print("🐌⚡ TESTING LARGE GRAPH PERFORMANCE...")
    
    # Test different sizes to find where bottleneck occurs
    sizes = [10, 50, 100, 200, 500]
    
    for size in sizes:
        print(f"\n📋 Testing graph with {size} nodes...")
        
        start_time = time.time()
        g = groggy.Graph()
        
        # Add nodes
        node_start = time.time()
        node_ids = []
        for i in range(size):
            node_id = g.add_node(name=f"node_{i}", index=i, category="test")
            node_ids.append(node_id)
        node_time = time.time() - node_start
        
        # Add edges (create a connected graph)
        edge_start = time.time()
        edge_count = 0
        for i in range(min(size-1, 100)):  # Limit edges to avoid explosion
            source = node_ids[i]
            target = node_ids[(i + 1) % size]
            g.add_edge(source, target, weight=1.0, edge_type="connection")
            edge_count += 1
        edge_time = time.time() - edge_start
        
        total_time = time.time() - start_time
        
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Nodes time: {node_time:.4f}s ({node_time/size:.6f}s per node)")
        print(f"  Edges time: {edge_time:.4f}s ({edge_time/edge_count:.6f}s per edge)")
        print(f"  Final node count: {len(g.node_ids)}")
        print(f"  Final edge count: {len(g.edge_ids)}")
        
        # Check if performance degraded significantly
        if node_time/size > 0.001:  # More than 1ms per node
            print(f"  🚨 SLOW NODE CREATION: {node_time/size:.6f}s per node")
            
        if edge_count > 0 and edge_time/edge_count > 0.001:  # More than 1ms per edge
            print(f"  🚨 SLOW EDGE CREATION: {edge_time/edge_count:.6f}s per edge")
        
        # Test some operations that might be affected by our changes
        print(f"  Testing entity type filtering...")
        filter_start = time.time()
        base_nodes = g.nodes.base
        base_count = len(base_nodes)
        filter_time = time.time() - filter_start
        print(f"  Base nodes filter time: {filter_time:.4f}s (found {base_count} nodes)")
        
        if filter_time > 0.1:  # More than 100ms for filtering
            print(f"  🚨 SLOW FILTERING: {filter_time:.4f}s for {size} nodes")

def test_entity_type_overhead():
    print(f"\n🔍 TESTING ENTITY TYPE SYSTEM OVERHEAD...")
    
    # Test if our Worf safety system is causing overhead
    g = groggy.Graph()
    
    # Time many node additions to see if there's cumulative overhead
    print("Adding 1000 nodes with timing...")
    times = []
    
    start_total = time.time()
    for i in range(1000):
        node_start = time.time()
        g.add_node(name=f"speed_test_{i}")
        node_time = time.time() - node_start
        times.append(node_time)
        
        if i % 100 == 0:
            recent_avg = sum(times[-10:]) / min(len(times), 10)
            print(f"  Node {i}: {node_time:.6f}s (recent avg: {recent_avg:.6f}s)")
    
    total_time = time.time() - start_total
    avg_time = sum(times) / len(times)
    
    print(f"1000 nodes total time: {total_time:.4f}s")
    print(f"Average per node: {avg_time:.6f}s")
    
    # Check if times are getting worse (indicating cumulative overhead)
    first_100_avg = sum(times[:100]) / 100
    last_100_avg = sum(times[-100:]) / 100
    
    print(f"First 100 nodes avg: {first_100_avg:.6f}s")
    print(f"Last 100 nodes avg: {last_100_avg:.6f}s")
    
    if last_100_avg > first_100_avg * 2:
        print(f"🚨 PERFORMANCE DEGRADATION: Later nodes are {last_100_avg/first_100_avg:.1f}x slower!")
        print("   This suggests cumulative overhead in our implementation")
    elif last_100_avg > first_100_avg * 1.5:
        print(f"⚠️ MILD DEGRADATION: Later nodes are {last_100_avg/first_100_avg:.1f}x slower")
    else:
        print(f"✅ STABLE PERFORMANCE: No significant degradation")

if __name__ == "__main__":
    test_large_graph_performance()
    test_entity_type_overhead()