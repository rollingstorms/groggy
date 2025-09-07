#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

import groggy as gr

# Create a simple graph
graph = gr.Graph()

# Add some nodes
node1 = graph.add_node()
node2 = graph.add_node()
node3 = graph.add_node()

# Set some attributes using basic Python types
graph.nodes[node1]["name"] = "Alice"
graph.nodes[node2]["name"] = "Bob"
graph.nodes[node3]["name"] = "Charlie"

graph.nodes[node1]["age"] = 25
graph.nodes[node2]["age"] = 30
graph.nodes[node3]["age"] = 35

# Test GraphArray with index column by getting the node_id values as an array
print("=== Testing NodesTable (our table display) ===")
nodes_table = graph.nodes.table()
print("Type:", type(nodes_table))
print("Rich display:")
print(nodes_table.rich_display())

# Now let's test if we can get the actual array functionality
# Look for array-like methods
print("\n=== Available methods ===")
methods = [method for method in dir(nodes_table) if not method.startswith('_')]
print("Methods:", methods)

# Test with a simple array of values from the table
print("\n=== Testing table __str__ method ===")
print(nodes_table)

print("\n=== Testing HTML representation ===")
html_repr = nodes_table._repr_html_()
print("HTML repr length:", len(html_repr))
print("First 200 chars:", html_repr[:200])