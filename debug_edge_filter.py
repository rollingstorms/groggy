#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')

import groggy as gr
import random

# Create a small test graph
g = gr.Graph()

# Add nodes and edges with strength attribute
nodes_data = [
    {'id': 'n1', 'name': 'Node1'},
    {'id': 'n2', 'name': 'Node2'},
    {'id': 'n3', 'name': 'Node3'}
]

edges_data = [
    {'source': 'n1', 'target': 'n2', 'strength': 0.8},
    {'source': 'n2', 'target': 'n3', 'strength': 0.5},
    {'source': 'n1', 'target': 'n3', 'strength': 0.9}
]

g.add_nodes(nodes_data)
g.add_edges(edges_data)

print("All edges:")
for edge_id in g.edges.keys():
    edge = g.edges[edge_id]
    print(f"  {edge_id}: {edge.source} -> {edge.target}, attributes: {edge.attributes}")

print("\nFiltering edges with strength > 0.7:")
filtered = g.filter_edges({'strength': ('>', 0.7)})
print(f"Filtered edges: {filtered}")

print("\nTesting Rust backend methods directly:")
print(f"Available strength values from Rust:")
for edge_id in g.edges.keys():
    edge = g.edges[edge_id]
    print(f"  {edge_id}: strength = {edge.attributes.get('strength')}")

# Test the Rust method directly
if hasattr(g._rust_core, 'filter_edges_by_numeric_comparison'):
    print("\nTesting Rust filter_edges_by_numeric_comparison directly:")
    rust_result = g._rust_core.filter_edges_by_numeric_comparison('strength', '>', 0.7)
    print(f"Rust result: {rust_result}")
else:
    print("Rust filter_edges_by_numeric_comparison not available")
