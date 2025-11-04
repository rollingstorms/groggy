"""Test degrees directly with manual node creation"""
from groggy import Graph

graph = Graph()

# Add nodes explicitly
n0 = graph.add_node()  # Should be 0
n1 = graph.add_node()  # Should be 1
n2 = graph.add_node()  # Should be 2

print(f"Node IDs: n0={n0}, n1={n1}, n2={n2}")

# Add edges: 0→1, 1→2
e01 = graph.add_edge(n0, n1)
e12 = graph.add_edge(n1, n2)

print(f"Edges: {n0}→{n1} (id={e01}), {n1}→{n2} (id={e12})")

# Now test with PageRank to see what degrees it computes
from groggy.algorithms.centrality import pagerank

sg = graph.view()

# Enable profiling to see internal stats
result = sg.apply(pagerank(max_iter=1, output_attr="pr"), persist=True)

print("\nPageRank after 1 iteration:")
for node in sorted(result.nodes, key=lambda n: n.id):
    print(f"  Node {node.id}: pr={node.pr:.6f}")
