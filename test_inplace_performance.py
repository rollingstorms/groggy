#!/usr/bin/env python3

"""
Test the inplace=True performance specifically
"""

import time
import groggy as gr

def test_inplace_performance():
    print("Testing connected_components with inplace=True...")
    
    # Create test graphs of different sizes
    sizes = [100, 200, 500]
    
    for size in sizes:
        print(f"\nTesting {size} nodes...")
        g = gr.generators.social_network(n=size)
        
        # Test without inplace
        start_time = time.time()
        components = g.connected_components(inplace=False)
        time_without_inplace = time.time() - start_time
        
        # Test with inplace
        start_time = time.time()
        components_inplace = g.connected_components(inplace=True)
        time_with_inplace = time.time() - start_time
        
        print(f"Without inplace: {time_without_inplace:.6f}s")
        print(f"With inplace: {time_with_inplace:.6f}s")
        print(f"Overhead: {(time_with_inplace - time_without_inplace):.6f}s")
        print(f"Relative overhead: {(time_with_inplace / time_without_inplace):.1f}x")
        
        # Verify the component_id attribute was set
        if hasattr(g.nodes, '__getitem__'):
            try:
                component_ids = g.nodes['component_id']
                print(f"Component IDs set: {len(component_ids)} nodes")
            except:
                print("Could not access component_id attribute")

if __name__ == "__main__":
    test_inplace_performance()
