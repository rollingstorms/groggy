from groggy import Graph

graph = Graph()
a, b, c = graph.add_node(), graph.add_node(), graph.add_node()
print(f"Nodes: a={a}, b={b}, c={c}")

e1 = graph.add_edge(a, b)
e2 = graph.add_edge(b, c)
print(f"Edges: e1={e1} ({a}→{b}), e2={e2} ({b}→{c})")

# Use native degree counting
sg = graph.view()

print("\nNative degree count:")
from groggy.algorithms.centrality import degree_centrality
result = sg.apply(degree_centrality(degree_type="out", output_attr="out_deg"), persist=True)
for node in sorted(result.nodes, key=lambda n: n.id):
    print(f"  Node {node.id}: out_degree={node.out_deg}")
