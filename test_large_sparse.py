#!/usr/bin/env python3
"""
Test large sparse attribute filtering performance
"""

import groggy
import time
import random

# Test the 250k node case mentioned by user
g = groggy.Graph()

print("Creating large graph with sparse attributes...")
start = time.time()

total_nodes = 250000
nodes_with_age = 0

# Create nodes with very sparse attributes (1% have age)
for i in range(total_nodes):
    node_id = g.add_node()
    
    # Only 1% of nodes get the age attribute 
    if random.random() < 0.01:
        age_value = random.randint(18, 80)
        g.set_node_attribute(node_id, "age", groggy.AttrValue(age_value))
        nodes_with_age += 1
    
    if i % 50000 == 0:
        print(f"  Created {i:,} nodes...")

elapsed = time.time() - start
print(f"\nCreated {total_nodes:,} nodes ({nodes_with_age:,} with age - {nodes_with_age/total_nodes*100:.1f}%) in {elapsed:.2f}s")

# Test sparse filtering performance
print(f"\nTesting sparse attribute filtering (age >= 50)...")
start = time.time()

try:
    age_filter = groggy.NodeFilter.attribute_filter(
        "age", 
        groggy.AttributeFilter.greater_than_or_equal(groggy.AttrValue(50))
    )
    
    filtered_nodes = g.filter_nodes(age_filter)
    
    elapsed = time.time() - start
    
    print(f"Found {len(filtered_nodes):,} nodes >= 50 in {elapsed*1000:.2f}ms")
    print(f"Performance: {total_nodes/(elapsed*1000):.0f} nodes/ms")
    
    # Expected: ~50% of nodes with age should match (age >= 50)
    expected_matches = nodes_with_age * 0.5
    print(f"Expected ~{expected_matches:.0f} matches, got {len(filtered_nodes)} ({len(filtered_nodes)/expected_matches*100:.1f}% of expected)")
    
    if elapsed < 0.1:  # Under 100ms is excellent
        print("✅ EXCELLENT: Sub-100ms performance for 250k nodes!")
    elif elapsed < 0.2:  # Under 200ms is very good  
        print("✅ VERY GOOD: Sub-200ms performance for 250k nodes!")
    else:
        print("⚠️  Performance could be better...")
    
except Exception as e:
    print(f"Filtering failed: {e}")
    import traceback
    traceback.print_exc()

print("\nLarge sparse attribute test completed!")