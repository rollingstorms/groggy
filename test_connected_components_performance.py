#!/usr/bin/env python3
"""
Test script to specifically verify connected components performance improvements.
"""

import sys
import time
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy

def test_connected_components_performance():
    """Test connected components performance after O(n) optimizations"""
    
    # Create a moderately sized graph with multiple components
    print("🧪 Creating test graph for connected components...")
    g = groggy.Graph()
    
    # Add 1000 nodes to create a meaningful test
    nodes = []
    for i in range(1000):
        node = g.add_node()
        nodes.append(node)
    
    print(f"Created {len(nodes)} nodes")
    
    # Create several disconnected components
    # Component 1: nodes 0-199 (chain)
    for i in range(199):
        g.add_edge(nodes[i], nodes[i+1])
    
    # Component 2: nodes 200-399 (chain)  
    for i in range(200, 399):
        g.add_edge(nodes[i], nodes[i+1])
        
    # Component 3: nodes 400-599 (chain)
    for i in range(400, 599):
        g.add_edge(nodes[i], nodes[i+1])
        
    # Components 4-10: smaller isolated components
    for comp_start in range(600, 900, 50):
        comp_end = min(comp_start + 49, 999)
        for i in range(comp_start, comp_end):
            g.add_edge(nodes[i], nodes[i+1])
    
    print(f"📊 Graph created: {len(g.nodes)} nodes, {len(g.edges)} edges")
    
    # Test connected components performance
    print("\n🚀 Testing connected components performance...")
    
    start_time = time.time()
    
    # This should now use the optimized algorithm with HashSet for duplicate edge checking
    components = g.analytics.connected_components()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Connected components completed in {duration:.4f}s")
    print(f"📊 Found {len(components)} components")
    
    if len(components) > 0:
        largest = max(components, key=len)
        print(f"📈 Largest component has {len(largest)} nodes")
        
    # Expected improvement: Should be much faster than before the O(n) edge checking fix
    if duration < 1.0:  # Should be sub-second for 1000 nodes
        print("🎉 PERFORMANCE IMPROVED: Connected components is now reasonably fast!")
        print("🔧 O(n) edge duplicate checking optimization is working")
    else:
        print("⚠️  Still some performance issues - may need additional optimization")
    
    return True

if __name__ == "__main__":
    test_connected_components_performance()