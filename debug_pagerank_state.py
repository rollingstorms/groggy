#!/usr/bin/env python
"""Minimal script to debug PageRank state leakage."""
import sys
sys.path.insert(0, 'python-groggy/python')

import groggy as gg
from benchmark_builder_vs_native import build_pagerank_algorithm

def test_pagerank_twice():
    """Run PageRank twice on the same graph and check for differences."""
    
    # Create test graph: A -> B -> C (directed chain)
    g = gg.Graph(directed=True)
    a = g.add_node()
    b = g.add_node()
    c = g.add_node()
    g.add_edge(a, b)
    g.add_edge(b, c)
    
    print("=== First run ===")
    algo1 = build_pagerank_algorithm(n=3, max_iter=20, damping=0.85)
    result1 = g.view().apply(algo1)
    pr1 = {node_obj.id: node_obj.pagerank for node_obj in result1.nodes}
    print(f"PageRank (run 1): {pr1}")
    
    print("\n=== Second run (same graph instance) ===")
    algo2 = build_pagerank_algorithm(n=3, max_iter=20, damping=0.85)
    result2 = g.view().apply(algo2)
    pr2 = {node_obj.id: node_obj.pagerank for node_obj in result2.nodes}
    print(f"PageRank (run 2): {pr2}")
    
    print("\n=== Third run (fresh graph) ===")
    g3 = gg.Graph(directed=True)
    a3 = g3.add_node()
    b3 = g3.add_node()
    c3 = g3.add_node()
    g3.add_edge(a3, b3)
    g3.add_edge(b3, c3)
    
    algo3 = build_pagerank_algorithm(n=3, max_iter=20, damping=0.85)
    result3 = g3.view().apply(algo3)
    pr3 = {node_obj.id: node_obj.pagerank for node_obj in result3.nodes}
    print(f"PageRank (run 3): {pr3}")
    
    # Check differences
    common_keys = sorted(pr1.keys())
    max_diff_12 = max(abs(pr1[k] - pr2[k]) for k in common_keys)
    max_diff_13 = max(abs(pr1[common_keys[i]] - pr3[sorted(pr3.keys())[i]]) for i in range(3))
    
    print(f"\nMax diff run1 vs run2: {max_diff_12:.2e}")
    print(f"Max diff run1 vs run3: {max_diff_13:.2e}")
    
    if max_diff_12 > 1e-6:
        print("⚠️  WARNING: Results differ between run1 and run2 on same graph!")
    if max_diff_13 > 1e-6:
        print("⚠️  WARNING: Results differ between run1 and run3 on fresh graphs!")
    
    # Native comparison
    print("\n=== Native PageRank ===")
    from groggy.algorithms import centrality
    result_native = g.view().apply(centrality.pagerank(max_iter=20, damping=0.85), persist=True)
    pr_native_vals = {node_obj.id: result_native.get_node_attribute(node_obj.id, "pagerank") 
                      for node_obj in result_native.nodes}
    print(f"Native PageRank: {pr_native_vals}")
    
    max_diff_native = max(abs(pr2[k] - pr_native_vals[k]) for k in common_keys)
    print(f"Max diff builder vs native: {max_diff_native:.2e}")

if __name__ == "__main__":
    test_pagerank_twice()
