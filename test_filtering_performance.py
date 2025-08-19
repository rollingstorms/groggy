#!/usr/bin/env python3
"""
Test filtering performance after the iterator optimization
"""

import groggy
import time
import random

# Create large graph
g = groggy.Graph()

print("Creating large graph with 250k nodes...")
start = time.time()
node_ids = []
for i in range(250000):
    node_id = g.add_node()
    node_ids.append(node_id)
    
    # Add age attribute to ~10% of nodes (sparse attribute)
    if random.random() < 0.1:
        g.set_node_attribute(node_id, "age", groggy.AttrValue(random.randint(18, 80)))
        
    if i % 50000 == 0:
        print(f"  Created {i} nodes...")

print(f"Graph created in {time.time() - start:.2f}s")
print(f"Nodes: {g.node_count()}, Edges: {g.edge_count()}")

# Test filtering performance
print("\nTesting attribute filtering performance...")

# Test case: find nodes with age >= 50 (should be ~5% of total nodes)
filter_tests = [
    ("age >= 50", lambda age: age >= 50),
    ("age >= 30", lambda age: age >= 30),
    ("age >= 65", lambda age: age >= 65),
]

for test_name, filter_condition in filter_tests:
    print(f"\nTesting filter: {test_name}")
    
    # Time the filtering operation
    start_time = time.time()
    
    # Use the graph's filtering method
    from groggy import NodeFilter, AttributeFilter
    # Direct filtering for simpler testing
    matching_nodes = g.find_nodes_with_attribute("age")
    
    # Simple approach - just measure the time for nodes with age >= value
    if "50" in test_name:
        matching_nodes = g.find_nodes_with_attribute("age")
        matching_count = 0
        for node_id in matching_nodes:
            age = g.get_node_attribute(node_id, "age")
            if age is not None and age >= 50:
                matching_count += 1
    elif "30" in test_name:
        matching_nodes = g.find_nodes_with_attribute("age")
        matching_count = 0
        for node_id in matching_nodes:
            age = g.get_node_attribute(node_id, "age")
            if age is not None and age >= 30:
                matching_count += 1
    else:  # 65
        matching_nodes = g.find_nodes_with_attribute("age")
        matching_count = 0
        for node_id in matching_nodes:
            age = g.get_node_attribute(node_id, "age")
            if age is not None and age >= 65:
                matching_count += 1
    
    elapsed = time.time() - start_time
    print(f"  Found {matching_count} matching nodes in {elapsed*1000:.2f}ms")
    
    # Calculate performance metrics
    nodes_per_ms = g.node_count() / (elapsed * 1000)
    print(f"  Performance: {nodes_per_ms:.0f} nodes/ms")

print("\nFiltering performance test completed!")