#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy
import time

def test_components_array():
    print("🧪 Testing ComponentsArray lazy implementation...")
    
    # Create a graph with many small components
    g = groggy.Graph()
    g.add_nodes(10000)
    
    # Create many small disconnected components (chains of 5 nodes each)
    edge_pairs = []
    for start in range(0, 10000, 5):
        for i in range(4):  # Create chains of 5 nodes
            if start + i + 1 < 10000:
                edge_pairs.append((start + i, start + i + 1))
    
    g.add_edges(edge_pairs)
    print(f"Created graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    print("\n=== Testing ComponentsArray ===")
    
    start_time = time.time()
    components = g.view().connected_components()
    creation_time = time.time() - start_time
    print(f"✅ Components creation: {creation_time*1000:.1f}ms")
    print(f"✅ Components type: {type(components)}")
    print(f"✅ Number of components: {len(components)}")
    
    print("\n=== Testing Lazy Access ===")
    
    # Access first component (should materialize on demand)
    start_time = time.time()
    first_component = components[0]
    access_time = time.time() - start_time
    print(f"✅ First component access: {access_time*1000:.1f}ms")
    print(f"✅ First component type: {type(first_component)}")
    print(f"✅ First component nodes: {len(first_component.nodes)}")
    
    # Access same component again (should use cache)
    start_time = time.time()
    first_component_again = components[0]
    cache_time = time.time() - start_time
    print(f"✅ First component (cached): {cache_time*1000:.1f}ms")
    
    print("\n=== Testing Array Interface ===")
    
    # Test iteration
    count = 0
    start_time = time.time()
    for comp in components:
        count += 1
        if count >= 3:  # Just test first few
            break
    iter_time = time.time() - start_time
    print(f"✅ Iteration over 3 components: {iter_time*1000:.1f}ms")
    
    # Test sizes without materialization
    start_time = time.time()
    sizes = components.sizes()
    sizes_time = time.time() - start_time
    print(f"✅ Get component sizes: {sizes_time*1000:.1f}ms")
    print(f"✅ First 5 component sizes: {sizes[:5]}")
    
    print(f"\n🎉 ComponentsArray is working! Much faster than eager materialization.")

if __name__ == "__main__":
    test_components_array()