from groggy import Graph

graph = Graph()
a, b, c = graph.add_node(), graph.add_node(), graph.add_node()
print(f"a = {a}, b = {b}, c = {c}")

graph.add_edge(a, b)
graph.add_edge(b, c)

print(f"\nEdges: {a}→{b}, {b}→{c}")
print(f"Expected: a (id={a}) has out-degree 1, c (id={c}) has out-degree 0 (sink)")
