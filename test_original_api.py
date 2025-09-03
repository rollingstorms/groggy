#!/usr/bin/env python3
"""
Test the original API that the user wanted to work.
"""

import groggy

# Test the original user's code
g = groggy.Graph()
node0 = g.add_node(name="Alice", age=25, salary=50000)
node1 = g.add_node(name="Bob", age=30, salary=60000)
node2 = g.add_node(name="Charlie", age=35, salary=70000)

print(f"Added nodes: {node0}, {node1}, {node2}")

# Create subgraph
subgraph = g.nodes[[0, 1, 2]]

# Test the high-level API (add_to_graph which calls collapse_to_meta_node)
try:
    meta_node = subgraph.add_to_graph({
        "total_salary": "sum",
        "avg_age": "mean",
        "person_count": "count"
    })

    print(f"Meta-node created: {meta_node}")
    print(f"Meta-node ID: {meta_node.node_id}")
    print(f"Has subgraph: {meta_node.has_subgraph}")
    
    attrs = meta_node.attributes()
    print(f"Attributes: {attrs}")
    
    # Check if this gives us the same issue
    # Let's manually check what nodes have contained_subgraph attributes
    print(f"\nChecking all graph nodes for contained_subgraph:")
    all_nodes = g.nodes[list(g.node_ids)]
    for node_id in g.node_ids:
        contained = all_nodes.get_node_attribute(node_id, "contained_subgraph")
        if contained is not None:
            print(f"  Node {node_id}: contained_subgraph = {contained}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()