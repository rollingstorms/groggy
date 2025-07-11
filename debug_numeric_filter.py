#!/usr/bin/env python3
"""Debug the numeric filtering issue"""

import sys
import os

# Import local groggy
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')
import groggy as gr

# Create a simple test graph
graph = gr.Graph(directed=False)
graph.use_rust = True

# Add some test nodes with salary data
test_nodes = [
    {'id': 'n1', 'salary': 50000},
    {'id': 'n2', 'salary': 75000},
    {'id': 'n3', 'salary': 100000},
    {'id': 'n4', 'salary': 125000},
    {'id': 'n5', 'salary': 150000},
]

graph.add_nodes(test_nodes)

print("Test nodes added:")
for node_id in graph.get_node_ids():
    node = graph.get_node(node_id)
    print(f"  {node_id}: salary = {node.attributes.get('salary', 'N/A')}")

print("\nTesting salary > 100000:")
result1 = graph.filter_nodes({'salary': ('>', 100000)})
print(f"  Result: {result1}")

print("\nTesting salary >= 100000:")
result2 = graph.filter_nodes({'salary': ('>=', 100000)})
print(f"  Result: {result2}")

print("\nTesting salary == 100000:")
result3 = graph.filter_nodes({'salary': ('==', 100000)})
print(f"  Result: {result3}")

print("\nTesting exact match (no tuple):")
result4 = graph.filter_nodes({'salary': 100000})
print(f"  Result: {result4}")

print("\nTesting with float values:")
result5 = graph.filter_nodes({'salary': ('>', 100000.0)})
print(f"  Result: {result5}")
