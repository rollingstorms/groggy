import groggy

# Create graph and subgraphs
g = groggy.Graph()
g.add_node(name="Alice", age=25, salary=50000)
g.add_node(name="Bob", age=30, salary=60000)
g.add_node(name="Charlie", age=35, salary=70000)

# Create subgraph
subgraph = g.nodes[[0, 1, 2]]

# Collapse to meta-node with aggregation
meta_node = subgraph.add_to_graph({
  "total_salary": "sum",
  "avg_age": "mean",
  "person_count": "count"
})

# Access meta-node properties  
print(f"Meta-node ID: {meta_node.node_id}")
print(f"Attributes: {meta_node.attributes()}")

# Find all meta-nodes in graph
subgraph_nodes = g.nodes.subgraphs
print(f"Found {len(subgraph_nodes)} meta-nodes")