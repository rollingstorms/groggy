#!/usr/bin/env python3
"""
Simple test of filtering performance after optimization
"""

import groggy
import time
import random

# Create smaller test for quick validation
g = groggy.Graph()

print("Creating test graph...")
start = time.time()

# Create nodes with sparse attribute
nodes_with_age = 0
total_nodes = 50000  # Smaller test

node_ids = []
for i in range(total_nodes):
    node_id = g.add_node()
    node_ids.append(node_id)
    
    # Add age attribute to 10% of nodes 
    if random.random() < 0.1:
        age_value = random.randint(18, 80)
        g.set_node_attribute(node_id, "age", groggy.AttrValue(age_value))
        nodes_with_age += 1

elapsed = time.time() - start        
print(f"Created {total_nodes} nodes ({nodes_with_age} with age) in {elapsed:.3f}s")

# Test filtering by iterating through all nodes
print("\nTesting manual filtering (iterate all nodes)...")
start = time.time()

matches = 0
for node_id in node_ids:
    age = g.get_node_attribute(node_id, "age")
    if age is not None and age.value >= 50:
        matches += 1

manual_time = time.time() - start
print(f"Manual filtering: Found {matches} nodes >= 50 in {manual_time*1000:.2f}ms")
print(f"Performance: {total_nodes/(manual_time*1000):.0f} nodes/ms")

# Test using the query system if available
print("\nTesting optimized filtering...")
try:
    from groggy import NodeFilter, AttributeFilter
    
    start = time.time()
    
    # Create filter for age >= 50
    age_filter = groggy.NodeFilter.attribute_filter(
        "age", 
        groggy.AttributeFilter.greater_than_or_equal(groggy.AttrValue(50))
    )
    
    # Apply filter
    filtered_nodes = g.filter_nodes(age_filter)
    
    optimized_time = time.time() - start
    print(f"Optimized filtering: Found {len(filtered_nodes)} nodes >= 50 in {optimized_time*1000:.2f}ms")
    print(f"Performance: {total_nodes/(optimized_time*1000):.0f} nodes/ms")
    
    speedup = manual_time / optimized_time
    print(f"Speedup: {speedup:.1f}x")
    
except Exception as e:
    print(f"Optimized filtering failed: {e}")

print("\nTest completed!")