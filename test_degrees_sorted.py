from groggy import Graph
from groggy.builder import AlgorithmBuilder

graph = Graph()
a, b, c = graph.add_node(), graph.add_node(), graph.add_node()
print(f"Created nodes: a={a}, b={b}, c={c}")

graph.add_edge(a, b)
graph.add_edge(b, c)
print(f"Created edges: {a}→{b}, {b}→{c}")

sg = graph.view()

builder = AlgorithmBuilder("test_degrees")
ranks = builder.init_nodes(default=1.0)
degrees = builder.node_degrees(ranks)
builder.attach_as("degrees", degrees)

algo = builder.build()
result = sg.apply(algo)

print("\nDegrees (sorted by node ID):")
for node in sorted(result.nodes, key=lambda n: n.id):
    print(f"  Node {node.id}: degree={node.degrees}")

# Also check with graph API
print("\nDegrees from graph API:")
graph_ref = graph.view().graph().borrow()
for node_id in [a, b, c]:
    deg = graph_ref.out_degree(node_id)
    print(f"  Node {node_id}: out_degree={deg}")
