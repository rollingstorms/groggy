#!/usr/bin/env python3

"""
Test connected components scaling to identify bottlenecks
"""

import time
import groggy as gr
import gc

def test_scaling():
    """Test different sized graphs to see where it breaks"""
    sizes = [1000, 2000, 5000, 10000, 20000]
    
    for size in sizes:
        print(f"\nTesting {size} nodes...")
        
        try:
            # Force garbage collection
            gc.collect()
            
            # Create test graph
            print(f"  Creating graph...")
            start_time = time.time()
            g = gr.generators.social_network(n=size)
            creation_time = time.time() - start_time
            print(f"  Graph created in {creation_time:.3f}s: {g.node_count()} nodes, {g.edge_count()} edges")
            
            # Test connected components with timeout
            print(f"  Running connected_components...")
            start_time = time.time()
            
            # Set a reasonable timeout for large graphs
            if size >= 20000:
                print(f"  Skipping {size} nodes - likely too slow")
                continue
                
            components = g.connected_components()
            cc_time = time.time() - start_time
            
            print(f"  Connected components: {len(components)} found in {cc_time:.3f}s")
            print(f"  Rate: {g.node_count() / cc_time:.0f} nodes/sec, {g.edge_count() / cc_time:.0f} edges/sec")
            
            # Check if it's getting too slow
            if cc_time > 10.0:  # More than 10 seconds
                print(f"  Breaking - too slow at {size} nodes")
                break
                
        except KeyboardInterrupt:
            print(f"  Interrupted at {size} nodes")
            break
        except Exception as e:
            print(f"  Error at {size} nodes: {e}")
            break

def test_disconnected_large():
    """Test with a large disconnected graph to see if that's faster"""
    print("\nTesting large disconnected graph...")
    
    g = gr.Graph()
    
    # Create 10,000 disconnected nodes (no edges)
    nodes = []
    for i in range(10000):
        node_id = g.add_node(index=i)
        nodes.append(node_id)
    
    print(f"Created {g.node_count()} nodes, {g.edge_count()} edges")
    
    start_time = time.time()
    components = g.connected_components()
    cc_time = time.time() - start_time
    
    print(f"Connected components: {len(components)} found in {cc_time:.3f}s")
    print(f"Rate: {g.node_count() / cc_time:.0f} nodes/sec")

if __name__ == "__main__":
    print("Connected Components Scaling Analysis")
    print("=" * 50)
    
    test_scaling()
    test_disconnected_large()
