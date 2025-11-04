"""Detailed debugging of PageRank."""
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms.centrality import pagerank

# Create simple test case: 3 nodes in a line
# A → B → C
graph = Graph()
a, b, c = graph.add_node(), graph.add_node(), graph.add_node()
graph.add_edge(a, b)
graph.add_edge(b, c)

sg = graph.view()

n = 3
damping = 0.85
max_iter = 5  # Just 5 iterations to debug

print("Graph: A → B → C")
print(f"Out-degrees: A=1, B=1, C=0 (C is a sink)")
print()

# Native PageRank
result_native = sg.apply(pagerank(max_iter=max_iter, damping=damping, output_attr="pr_native"), persist=True)

print("Native PageRank after {} iterations:".format(max_iter))
for node in result_native.nodes:
    print(f"  Node {node.id}: {node.pr_native:.10f}")
native_sum = sum(node.pr_native for node in result_native.nodes)
print(f"  Sum: {native_sum:.10f}")
print()

# Builder PageRank - but let's manually check what degrees we get
builder = AlgorithmBuilder("test_detailed")

ranks = builder.init_nodes(default=1.0 / n)
ranks = builder.var("ranks", ranks)

degrees = builder.node_degrees(ranks)

# Just to see degrees
builder.attach_as("degrees", degrees)

# Execute just to see degrees
algo_degrees = builder.build()
result_degrees = graph.view().apply(algo_degrees)

print("Degrees from builder:")
for node in result_degrees.nodes:
    print(f"  Node {node.id}: {node.degrees}")
print()

# Now full PageRank
builder2 = AlgorithmBuilder("test_detailed2")

ranks = builder2.init_nodes(default=1.0 / n)
ranks = builder2.var("ranks", ranks)

degrees = builder2.node_degrees(ranks)
inv_degrees = builder2.core.recip(degrees, epsilon=1e-12)
is_sink = builder2.core.compare(degrees, "eq", 0.0)

with builder2.iterate(max_iter):
    contrib = builder2.core.mul(ranks, inv_degrees)
    contrib = builder2.core.where(is_sink, 0.0, contrib)
    
    neighbor_sum = builder2.core.neighbor_agg(contrib, agg="sum")
    
    sink_ranks = builder2.core.where(is_sink, ranks, 0.0)
    sink_mass = builder2.core.reduce_scalar(sink_ranks, op="sum")
    sink_contrib = builder2.core.broadcast_scalar(sink_mass, ranks)
    
    damped_neighbors = builder2.core.mul(neighbor_sum, damping)
    damped_sinks = builder2.core.mul(sink_contrib, damping / n)
    
    teleport = (1.0 - damping) / n
    
    ranks = builder2.core.add(damped_neighbors, damped_sinks)
    ranks = builder2.core.add(ranks, teleport)
    ranks = builder2.var("ranks", ranks)

ranks = builder2.var("ranks", builder2.core.normalize_sum(ranks))
builder2.attach_as("pagerank", ranks)

algo2 = builder2.build()
result_builder = graph.view().apply(algo2)

print("Builder PageRank after {} iterations:".format(max_iter))
for node in result_builder.nodes:
    print(f"  Node {node.id}: {node.pagerank:.10f}")
builder_sum = sum(node.pagerank for node in result_builder.nodes)
print(f"  Sum: {builder_sum:.10f}")
print()

print("Comparison:")
native_map = {node.id: node.pr_native for node in result_native.nodes}
builder_map = {node.id: node.pagerank for node in result_builder.nodes}

for node_id in sorted(native_map.keys()):
    diff = abs(native_map[node_id] - builder_map[node_id])
    print(f"  Node {node_id}: native={native_map[node_id]:.10f}, builder={builder_map[node_id]:.10f}, diff={diff:.10f}")
