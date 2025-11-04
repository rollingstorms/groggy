"""Trace LPA step by step to see what's happening."""
from groggy import Graph
from groggy.builder import AlgorithmBuilder

# Create a simple 3-node graph: 0 <-> 1 <-> 2
graph = Graph()
n0, n1, n2 = [graph.add_node() for _ in range(3)]
graph.add_edge(n0, n1)
graph.add_edge(n1, n0)
graph.add_edge(n1, n2)
graph.add_edge(n2, n1)

print("Graph: 0 <-> 1 <-> 2")
print("Expected behavior (async, ordered):")
print("  Init: [0, 1, 2]")
print("  Node 0 first: sees [0(self), 1, 1] → mode=1 → [1, 1, 2]")
print("  Node 1 next:  sees [1(self), 1, 2, 2] → mode is tie between 1 and 2, pick lowest=1 → [1, 1, 2]") 
print("  Node 2 last:  sees [2(self), 1, 1] → mode=1 → [1, 1, 1]")
print("  After iter 1: all should be 1\n")

for num_iters in [0, 1, 2]:
    builder = AlgorithmBuilder(f"lpa_{num_iters}")
    labels = builder.init_nodes(unique=True)
    
    if num_iters > 0:
        with builder.iterate(num_iters):
            neighbor_labels = builder.core.collect_neighbor_values(labels, include_self=True)
            new_labels = builder.core.mode(neighbor_labels, tie_break="lowest")
            labels = builder.core.update_in_place(new_labels, labels, ordered=True)
    
    builder.attach_as("labels", labels)
    result = graph.view().apply(builder.build())
    
    vals = [result.get_node_attribute(n.id, "labels") for n in sorted(result.nodes, key=lambda x: x.id)]
    print(f"After {num_iters} iterations: {vals}")
