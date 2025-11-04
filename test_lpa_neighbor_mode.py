"""Test LPA with neighbor_mode_update primitive."""
from groggy import Graph
from groggy.builder import AlgorithmBuilder

# Create a simple 3-node graph: 0 <-> 1 <-> 2
graph = Graph()
n0, n1, n2 = [graph.add_node() for _ in range(3)]
graph.add_edge(n0, n1)
graph.add_edge(n1, n0)
graph.add_edge(n1, n2)
graph.add_edge(n2, n1)

print("Graph: 0 <-> 1 <-> 2\n")

builder = AlgorithmBuilder("lpa_proper")
labels = builder.init_nodes(unique=True)
with builder.iterate(3):
    labels = builder.core.neighbor_mode_update(
        labels, 
        include_self=True,
        tie_break="lowest",
        ordered=True
    )
builder.attach_as("community", labels)
result = graph.view().apply(builder.build())

print("After 3 iterations:")
vals = [result.get_node_attribute(n.id, "community") for n in sorted(result.nodes, key=lambda x: x.id)]
print(f"  Labels: {vals}")

labels_set = set(vals)
print(f"  Unique labels: {labels_set}")
print(f"  Number of communities: {len(labels_set)}")
print("✅ Should be 1 community" if len(labels_set) == 1 else f"❌ Got {len(labels_set)} communities")
