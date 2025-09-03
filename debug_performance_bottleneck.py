#!/usr/bin/env python3
"""
Debug Performance Bottleneck in Graph Creation 🐌⚡
"""

import groggy
import time

def test_performance_bottleneck():
    print("🐌⚡ DEBUGGING PERFORMANCE BOTTLENECK...")
    
    # Test 1: Basic graph creation timing
    print("\n📋 Test 1: Basic graph creation")
    
    start_time = time.time()
    g = groggy.Graph()
    creation_time = time.time() - start_time
    print(f"Graph creation time: {creation_time:.4f}s")
    
    # Test 2: Node addition timing
    print("\n📋 Test 2: Node addition timing")
    
    node_times = []
    for i in range(10):
        start_time = time.time()
        node_id = g.add_node(name=f"node_{i}", index=i)
        node_time = time.time() - start_time
        node_times.append(node_time)
        print(f"Node {i} creation time: {node_time:.6f}s (ID: {node_id})")
        
        if node_time > 0.01:  # If it takes more than 10ms, something is wrong
            print(f"⚠️ SLOW NODE CREATION: {node_time:.6f}s")
    
    avg_node_time = sum(node_times) / len(node_times)
    print(f"Average node creation time: {avg_node_time:.6f}s")
    
    # Test 3: Edge addition timing
    print("\n📋 Test 3: Edge addition timing")
    
    edge_times = []
    node_ids = list(g.node_ids)
    
    for i in range(min(5, len(node_ids)-1)):
        start_time = time.time()
        edge_id = g.add_edge(node_ids[i], node_ids[i+1], weight=1.0)
        edge_time = time.time() - start_time
        edge_times.append(edge_time)
        print(f"Edge {i} creation time: {edge_time:.6f}s (ID: {edge_id})")
        
        if edge_time > 0.01:  # If it takes more than 10ms, something is wrong
            print(f"⚠️ SLOW EDGE CREATION: {edge_time:.6f}s")
    
    if edge_times:
        avg_edge_time = sum(edge_times) / len(edge_times)
        print(f"Average edge creation time: {avg_edge_time:.6f}s")
    
    # Test 4: Compare with batch creation vs individual
    print("\n📋 Test 4: Batch vs individual operations")
    
    # Individual creation
    start_time = time.time()
    g_individual = groggy.Graph()
    for i in range(50):
        g_individual.add_node(name=f"batch_node_{i}")
    individual_time = time.time() - start_time
    
    print(f"50 individual nodes: {individual_time:.4f}s")
    print(f"Per node (individual): {individual_time/50:.6f}s")
    
    # Test 5: Check if the issue is with entity_type setting
    print("\n📋 Test 5: Profile entity_type operations")
    
    g_test = groggy.Graph()
    
    # Time adding node without checking entity_type
    start_time = time.time()
    test_node = g_test.add_node(name="test")
    add_time = time.time() - start_time
    print(f"Add node time: {add_time:.6f}s")
    
    # Time checking entity_type
    start_time = time.time()
    all_nodes = g_test.nodes[list(g_test.node_ids)]
    try:
        entity_type = all_nodes.get_node_attribute(test_node, "entity_type")
        entity_check_time = time.time() - start_time
        print(f"Entity type check time: {entity_check_time:.6f}s")
        print(f"Entity type value: {entity_type}")
    except Exception as e:
        entity_check_time = time.time() - start_time
        print(f"Entity type check failed in {entity_check_time:.6f}s: {e}")
    
    print(f"\n🐌⚡ PERFORMANCE ANALYSIS COMPLETED!")
    
    # Identify potential issues
    if avg_node_time > 0.001:  # More than 1ms per node
        print(f"🚨 PERFORMANCE ISSUE: Node creation is slow ({avg_node_time:.6f}s per node)")
        print("   This suggests an issue with the add_node implementation")
        
    if edge_times and sum(edge_times)/len(edge_times) > 0.001:
        print(f"🚨 PERFORMANCE ISSUE: Edge creation is slow")
        print("   This suggests an issue with the add_edge implementation")
        
    if entity_check_time > 0.001:
        print(f"🚨 PERFORMANCE ISSUE: Entity type checking is slow")
        print("   This might be related to our Worf safety system")

if __name__ == "__main__":
    test_performance_bottleneck()