"""Test the LPA fix."""
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

# Test with corrected argument order
builder = AlgorithmBuilder("test_lpa_fixed")
labels = builder.init_nodes(unique=True)
with builder.iterate(3):
    neighbor_labels = builder.core.collect_neighbor_values(labels, include_self=True)
    new_labels = builder.core.mode(neighbor_labels, tie_break="lowest")
    labels = builder.core.update_in_place(new_labels, labels, ordered=True)  # FIXED ORDER
builder.attach_as("final", labels)
result = graph.view().apply(builder.build())

print("After 3 iterations (FIXED):")
for node in sorted(result.nodes, key=lambda n: n.id):
    val = result.get_node_attribute(node.id, "final")
    print(f"  Node {node.id}: label = {val}")

# All three should converge to the same community!
labels_set = {result.get_node_attribute(n.id, "final") for n in result.nodes}
print(f"\nUnique labels: {labels_set}")
print(f"Number of communities: {len(labels_set)}")
print("✅ Expected: 1 community" if len(labels_set) == 1 else "❌ Still broken")
