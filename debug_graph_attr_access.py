#!/usr/bin/env python3
"""
Debug Graph Attribute Access 🔍
"""

import groggy

# Add this to test what graph.get_node_attr returns vs accessor
g = groggy.Graph()
node1 = g.add_node(name="Alice", age=25)
node2 = g.add_node(name="Bob", age=30)

subgraph = g.nodes[[node1, node2]]
meta_node = subgraph.add_to_graph({"avg_age": ("mean", "age")})
meta_node_id = meta_node.node_id

print(f"Meta-node ID: {meta_node_id}")

# Test accessor method
all_nodes = g.nodes[list(g.node_ids)]
entity_type_via_accessor = all_nodes.get_node_attribute(meta_node_id, "entity_type")
print(f"Via accessor: entity_type = '{entity_type_via_accessor}' (type: {type(entity_type_via_accessor)})")

# Test if the issue is in space mapping
print(f"Node is in space: {g.contains_node(meta_node_id)}")

# Create a minimal test to see if the method works at all
regular_node_result = g.nodes.get_meta_node(node1)
print(f"get_meta_node({node1}) [regular node]: {regular_node_result}")

meta_node_result = g.nodes.get_meta_node(meta_node_id) 
print(f"get_meta_node({meta_node_id}) [meta node]: {meta_node_result}")