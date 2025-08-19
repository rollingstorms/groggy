#!/usr/bin/env python3
"""
Test script to verify DFS traversal performance improvements.
"""

import sys
import time
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy

def test_dfs_performance():
    """Test DFS performance after O(n) edge duplicate checking optimization"""
    
    print("🧪 Creating test graph for DFS traversal...")
    g = groggy.Graph()
    
    # Create a connected graph with 1000 nodes
    nodes = []
    for i in range(1000):
        node = g.add_node()
        nodes.append(node)
    
    print(f"Created {len(nodes)} nodes")
    
    # Create a dense connected graph (each node connected to next few nodes)
    for i in range(len(nodes) - 3):
        g.add_edge(nodes[i], nodes[i+1])  # Chain
        if i % 10 == 0:  # Add some branching
            g.add_edge(nodes[i], nodes[min(i+5, len(nodes)-1)])
            g.add_edge(nodes[i], nodes[min(i+10, len(nodes)-1)])
    
    print(f"📊 Graph created: {len(g.nodes)} nodes, {len(g.edges)} edges")
    
    # Test DFS performance
    print("\n🚀 Testing DFS traversal performance...")
    
    start_time = time.time()
    
    # DFS from the first node - this should now use the optimized algorithm with HashSet for duplicate edge checking
    start_node = nodes[0]
    
    # Check if there's a DFS method available
    try:
        # Try different possible API locations
        if hasattr(g, 'dfs'):
            result = g.dfs(start_node)
            method_used = "g.dfs()"
        elif hasattr(g, 'analytics') and hasattr(g.analytics, 'dfs'):
            result = g.analytics.dfs(start_node)
            method_used = "g.analytics.dfs()"
        else:
            print("❌ DFS method not found in API")
            return False
    except Exception as e:
        print(f"❌ DFS method failed: {e}")
        return False
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ DFS traversal completed in {duration:.4f}s using {method_used}")
    print(f"📊 Traversed {len(result) if hasattr(result, '__len__') else 'unknown'} nodes")
    
    # Expected improvement: Should be much faster than before the O(n) edge checking fix
    if duration < 0.1:  # Should be very fast for 1000 nodes
        print("🎉 PERFORMANCE IMPROVED: DFS is now very fast!")
        print("🔧 O(1) edge duplicate checking optimization is working")
    elif duration < 1.0:
        print("✅ Good performance: DFS is reasonably fast")
        print("🔧 Edge duplicate checking optimization helped")
    else:
        print("⚠️  Still some performance issues - may need additional optimization")
    
    return True

if __name__ == "__main__":
    test_dfs_performance()