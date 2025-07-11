#!/usr/bin/env python3
"""
Debug the complex filtering performance issue in detail
"""

import time
import random
import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')
import groggy as gr


def profile_each_step():
    """Profile each step of the complex filtering process"""
    
    # Create test graph
    nodes_data = []
    for i in range(5000):  # Medium size for clearer profiling
        nodes_data.append({
            'id': f'n{i}',
            'salary': random.randint(50000, 150000),
            'age': random.randint(25, 65),
            'role': random.choice(['engineer', 'manager', 'analyst'])
        })
    
    graph = gr.Graph(backend='rust')
    graph.add_nodes(nodes_data)
    
    print("=== Profiling each step ===")
    
    # Step 1: First numeric filter
    start = time.time()
    result1 = graph._rust_core.filter_nodes_by_numeric_comparison('salary', '>', 80000.0)
    time1 = time.time() - start
    print(f"Rust salary filter: {time1:.6f}s, {len(result1)} results")
    
    # Step 2: Second numeric filter
    start = time.time()
    result2 = graph._rust_core.filter_nodes_by_numeric_comparison('age', '>', 30.0)
    time2 = time.time() - start
    print(f"Rust age filter: {time2:.6f}s, {len(result2)} results")
    
    # Step 3: Manual set intersection
    start = time.time()
    result_set1 = set(result1)
    result_set2 = set(result2)
    intersection = result_set1.intersection(result_set2)
    time3 = time.time() - start
    print(f"Manual intersection: {time3:.6f}s, {len(intersection)} results")
    
    # Step 4: Test the complex filter through the Python wrapper
    start = time.time()
    result4 = graph.filter_nodes({'salary': ('>', 80000), 'age': ('>', 30)})
    time4 = time.time() - start
    print(f"Python wrapper complex: {time4:.6f}s, {len(result4)} results")
    
    # Step 5: Break down the Python wrapper execution
    start = time.time()
    filter_dict = {'salary': ('>', 80000), 'age': ('>', 30)}
    
    # Check if all filters are exact matches
    all_exact = all(not isinstance(value, tuple) for value in filter_dict.values())
    check_time = time.time() - start
    print(f"Check exact matches: {check_time:.6f}s, all_exact={all_exact}")
    
    # Separate exact matches from tuple comparisons
    start = time.time()
    exact_matches = {}
    tuple_filters = []
    
    for attr_name, value in filter_dict.items():
        if isinstance(value, tuple) and len(value) == 2:
            operator, comparison_value = value
            tuple_filters.append((attr_name, operator, comparison_value))
        else:
            exact_matches[attr_name] = value
    
    separation_time = time.time() - start
    print(f"Separate filters: {separation_time:.6f}s, exact={len(exact_matches)}, tuple={len(tuple_filters)}")
    
    # Apply each tuple filter
    result_set = None
    for i, (attr_name, operator, value) in enumerate(tuple_filters):
        start = time.time()
        
        if isinstance(value, (int, float)):
            filtered_ids = graph._rust_core.filter_nodes_by_numeric_comparison(attr_name, operator, float(value))
        else:
            filtered_ids = graph._rust_core.filter_nodes_by_string_comparison(attr_name, operator, str(value))
        
        rust_time = time.time() - start
        
        start = time.time()
        if result_set is None:
            result_set = set(filtered_ids)
        else:
            if len(filtered_ids) < len(result_set):
                result_set = {node_id for node_id in filtered_ids if node_id in result_set}
            else:
                result_set = {node_id for node_id in result_set if node_id in set(filtered_ids)}
        
        intersection_time = time.time() - start
        print(f"Filter {i+1} ({attr_name} {operator} {value}): rust={rust_time:.6f}s, intersection={intersection_time:.6f}s, result_size={len(result_set) if result_set else 0}")
    
    print(f"\nFinal results match: {sorted(list(result_set)) == sorted(result4)}")


if __name__ == "__main__":
    profile_each_step()
