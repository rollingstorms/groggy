"""Debug edge handling"""
import groggy as gg

# Directed chain
g = gg.Graph(directed=True)
g.add_nodes(3)
g.add_edges([(0, 1), (1, 2)])

print("Directed graph 0->1->2")
print(f"Node count: {g.node_count()}")
print(f"Edge count: {g.edge_count()}")
print(f"Out-degrees: {[g.out_degree(i) for i in range(3)]}")
print(f"In-degrees: {[g.in_degree(i) for i in range(3)]}")

# Check neighbors using neighbors() method
for i in range(3):
    neighbors = list(g.neighbors(i))
    print(f"Node {i}: neighbors={neighbors}")

# Now test the neighbor_agg directly 
builder = gg.AlgorithmBuilder("debug")
values = builder.init_nodes(default=1.0)
agg_result = builder.core.neighbor_agg(values, agg="sum")
builder.attach_as("agg", agg_result)

result = g.apply(builder.build())
print("\nneighbor_agg(uniform=1.0):")
for i in range(3):
    val = result.get_node_attribute(i, 'agg')
    in_deg = g.in_degree(i)
    print(f"  Node {i}: {val} (expected: {in_deg} in-neighbors)")

# Test with undirected
print("\n\nUndirected graph 0-1-2")
g2 = gg.Graph(directed=False)
g2.add_nodes(3)
g2.add_edges([(0, 1), (1, 2)])

print(f"Edge count: {g2.edge_count()}")
print(f"Degrees: {[g2.degree(i) for i in range(3)]}")

for i in range(3):
    neighbors = list(g2.neighbors(i))
    print(f"Node {i}: neighbors={neighbors}")

result2 = g2.apply(builder.build())
print("\nneighbor_agg(uniform=1.0):")
for i in range(3):
    val = result2.get_node_attribute(i, 'agg')
    deg = g2.degree(i)
    print(f"  Node {i}: {val} (expected: {deg} neighbors)")
