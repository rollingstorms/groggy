"""
Test raw out-degree queries.
"""
from groggy import Graph

# Create test graph
graph = Graph(directed=True)
nodes = [graph.add_node() for _ in range(5)]
graph.add_edge(nodes[0], nodes[1])
graph.add_edge(nodes[1], nodes[2])
graph.add_edge(nodes[2], nodes[0])
graph.add_edge(nodes[2], nodes[3])
graph.add_edge(nodes[3], nodes[4])
graph.add_edge(nodes[4], nodes[2])

print("Graph edges:")
sg = graph.view()
edge_count = 0
for edge in sg.edges:
    print(f"  {edge.source} → {edge.target}")
    edge_count += 1

print(f"\nTotal edges: {edge_count}")

print("\nNode out-degrees (from neighbors count):")
total_degree = 0
for node in sorted(sg.nodes, key=lambda n: n.id):
    neighbors = list(node.neighbors)
    degree = len(neighbors)
    total_degree += degree
    print(f"  Node {node.id}: {degree} neighbors = {neighbors}")

print(f"\nSum of out-degrees: {total_degree}")
print(f"Expected: should equal edge count = {edge_count}")
