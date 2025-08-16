#!/usr/bin/env python3

"""
Test connected components performance using groggy generators
"""

import time
import groggy

def test_connected_components_performance():
    """Test connected components with different graph sizes"""
    
    print("Testing connected_components performance...")
    print("=" * 50)
    
    # Test different sizes
    sizes = [50, 100, 200, 500, 1000]
    
    for n in sizes:
        print(f"\nTesting with {n} nodes...")
        
        # Create social network graph
        g = groggy.generators.social_network(n=n)
        print(f"Graph created: {g.node_count()} nodes, {g.edge_count()} edges")
        
        # Measure connected components performance
        start_time = time.time()
        components = g.connected_components()
        end_time = time.time()
        
        duration = end_time - start_time
        
        print(f"Time: {duration:.6f} seconds")
        print(f"Components found: {len(components)}")
        
        # Show component sizes
        if len(components) <= 10:
            sizes = [comp.node_count() for comp in components]
            sizes.sort(reverse=True)
            print(f"Component sizes: {sizes}")
        else:
            sizes = [comp.node_count() for comp in components]
            sizes.sort(reverse=True)
            print(f"Top 5 component sizes: {sizes[:5]}")
        
        # If taking too long, note it
        if duration > 1.0:
            print(f"⚠️  WARNING: Took {duration:.3f}s - this seems slow!")
            
        if duration > 5.0:
            print("❌ VERY SLOW - stopping test")
            break

def test_specific_slow_case():
    """Test a specific case that might be slow"""
    print("\n" + "=" * 50)
    print("Testing specific potentially slow case...")
    
    # Create a graph with many small components (worst case for some algorithms)
    g = groggy.Graph()
    
    # Add 1000 nodes with no edges (1000 components of size 1)
    node_ids = []
    for i in range(1000):
        node_id = g.add_node(index=i)
        node_ids.append(node_id)
    
    print(f"Created disconnected graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    start_time = time.time()
    components = g.connected_components()
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Time for disconnected graph: {duration:.6f} seconds")
    print(f"Components found: {len(components)}")
    
    if duration > 0.1:
        print(f"⚠️  This seems slow for a simple disconnected graph!")

def compare_with_simple_operations():
    """Compare connected_components with other graph operations"""
    print("\n" + "=" * 50)
    print("Comparing with other operations...")
    
    g = groggy.generators.social_network(n=500)
    print(f"Test graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Test node_count (should be very fast)
    start = time.time()
    count = g.node_count()
    node_count_time = time.time() - start
    print(f"node_count(): {node_count_time:.6f}s")
    
    # Test edge_count (should be very fast)
    start = time.time()
    count = g.edge_count()
    edge_count_time = time.time() - start
    print(f"edge_count(): {edge_count_time:.6f}s")
    
    # Test filter_nodes (should be reasonably fast)
    start = time.time()
    try:
        # Try to filter nodes (this might fail if there are no attributes)
        nodes = g.nodes
        filter_time = time.time() - start
        print(f"nodes access: {filter_time:.6f}s")
    except Exception as e:
        filter_time = time.time() - start
        print(f"nodes access: {filter_time:.6f}s (with error: {e})")
    
    # Test connected_components
    start = time.time()
    components = g.connected_components()
    cc_time = time.time() - start
    print(f"connected_components(): {cc_time:.6f}s")
    
    # Show comparison
    print(f"\nPerformance comparison:")
    print(f"  node_count: {node_count_time:.6f}s (baseline)")
    print(f"  edge_count: {edge_count_time:.6f}s ({edge_count_time/node_count_time:.1f}x)")
    print(f"  nodes access: {filter_time:.6f}s ({filter_time/node_count_time:.1f}x)")
    print(f"  connected_components: {cc_time:.6f}s ({cc_time/node_count_time:.1f}x)")
    
    if cc_time > 0.1:
        print(f"⚠️  connected_components seems unusually slow!")

def main():
    test_connected_components_performance()
    test_specific_slow_case()
    compare_with_simple_operations()

if __name__ == "__main__":
    main()
