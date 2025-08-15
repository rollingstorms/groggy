#!/usr/bin/env python3

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy

# Create graph
g = groggy.Graph()
nodes = [g.add_node() for _ in range(3)]

print("Setting attributes...")
for i, node in enumerate(nodes):
    print(f"Setting node {node}: value = {i * 10}")
    g.set_node_attribute(node, 'value', groggy.AttrValue(i * 10))

print("\nChecking attributes directly...")
for node in nodes:
    try:
        value = g.get_node_attribute(node, 'value')
        print(f"Node {node}: value = {value}")
    except Exception as e:
        print(f"Node {node}: error = {e}")

print("\nCreating table...")
table = g.table()
print(f"Table shape: {table.shape}")
print(f"Table columns: {table.columns}")

# Check what actual data is in the table
print(f"Sample row: {table[0]}")