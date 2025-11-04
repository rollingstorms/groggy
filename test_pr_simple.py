"""Test PageRank on a simple graph."""
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms import centrality

# Create simple graph: 0 → 1 → 2
graph = Graph(directed=True)
n0, n1, n2 = [graph.add_node() for _ in range(3)]
graph.add_edge(n0, n1)
graph.add_edge(n1, n2)

print("Graph: 0 → 1 → 2 (directed)\n")

# Native PageRank
native_result = graph.view().apply(
    centrality.pagerank(max_iter=100, damping=0.85, output_attr="pr"),
    persist=True
)
print("Native PageRank (100 iterations):")
for node in sorted(native_result.nodes, key=lambda n: n.id):
    print(f"  Node {node.id}: {node.pr:.10f}")

# Builder PageRank
from benchmark_builder_vs_native import build_pagerank_algorithm
algo = build_pagerank_algorithm(n=3, damping=0.85, max_iter=100)
builder_result = graph.view().apply(algo)
print("\nBuilder PageRank (100 iterations):")
for node in sorted(builder_result.nodes, key=lambda n: n.id):
    print(f"  Node {node.id}: {node.pagerank:.10f}")

print("\nDifferences:")
for node in sorted(native_result.nodes, key=lambda n: n.id):
    native_val = node.pr
    builder_val = builder_result.get_node_attribute(node.id, "pagerank")
    diff = abs(native_val - builder_val)
    print(f"  Node {node.id}: diff = {diff:.10f}")
