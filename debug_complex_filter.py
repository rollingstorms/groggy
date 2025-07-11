#!/usr/bin/env python3
"""
Debug the complex filtering performance issue
"""

import time
import random
import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')
import groggy as gr


def test_complex_filter_issue():
    """Debug why complex filters are slow"""
    
    # Create a smaller test graph for debugging
    nodes_data = []
    for i in range(1000):
        nodes_data.append({
            'id': f'n{i}',
            'salary': random.randint(50000, 150000),
            'age': random.randint(25, 65),
            'role': random.choice(['engineer', 'manager', 'analyst'])
        })
    
    graph = gr.Graph(backend='rust')
    graph.add_nodes(nodes_data)
    
    print("=== Testing different filtering approaches ===")
    
    # Test 1: Single numeric filter
    start = time.time()
    result1 = graph.filter_nodes({'salary': ('>', 80000)})
    time1 = time.time() - start
    print(f"Single filter (salary > 80000): {time1:.4f}s, {len(result1)} results")
    
    # Test 2: Another single numeric filter
    start = time.time()
    result2 = graph.filter_nodes({'age': ('>', 30)})
    time2 = time.time() - start
    print(f"Single filter (age > 30): {time2:.4f}s, {len(result2)} results")
    
    # Test 3: Complex filter (the problematic one)
    start = time.time()
    result3 = graph.filter_nodes({'salary': ('>', 80000), 'age': ('>', 30)})
    time3 = time.time() - start
    print(f"Complex filter (both): {time3:.4f}s, {len(result3)} results")
    
    # Test 4: Manual intersection for comparison
    start = time.time()
    result4 = list(set(result1).intersection(set(result2)))
    time4 = time.time() - start
    print(f"Manual intersection: {time4:.4f}s, {len(result4)} results")
    
    # Test 5: Exact match filter for reference
    start = time.time()
    result5 = graph.filter_nodes({'role': 'engineer'})
    time5 = time.time() - start
    print(f"Exact filter (role): {time5:.4f}s, {len(result5)} results")
    
    print(f"\nResults match: {sorted(result3) == sorted(result4)}")


if __name__ == "__main__":
    test_complex_filter_issue()
