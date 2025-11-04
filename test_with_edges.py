"""Test PageRank with edges, compare to native."""
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms.centrality import pagerank

# Create test graph: simple 5-node graph
graph = Graph()
nodes = [graph.add_node() for _ in range(5)]

# Create edges (simple chain + one back-link)
graph.add_edge(nodes[0], nodes[1])
graph.add_edge(nodes[1], nodes[2])
graph.add_edge(nodes[2], nodes[3])
graph.add_edge(nodes[3], nodes[4])
graph.add_edge(nodes[4], nodes[0])  # cycle back

sg = graph.view()

n = 5
damping = 0.85
max_iter = 20

# Native PageRank
result_native = sg.apply(pagerank(max_iter=max_iter, damping=damping, output_attr="pr_native"), persist=True)

print("Native PageRank:")
native_values = {}
for node in result_native.nodes:
    val = node.pr_native
    native_values[node.id] = val
    print(f"  Node {node.id}: {val:.10f}")
total_native = sum(native_values.values())
print(f"Total: {total_native:.10f}")

# Builder PageRank with primitives
builder = AlgorithmBuilder("test_with_edges")

ranks = builder.init_nodes(default=1.0 / n)
ranks = builder.var("ranks", ranks)

degrees = builder.node_degrees(ranks)
inv_degrees = builder.core.recip(degrees, epsilon=1e-12)
is_sink = builder.core.compare(degrees, "eq", 0.0)

with builder.iterate(max_iter):
    contrib = builder.core.mul(ranks, inv_degrees)
    contrib = builder.core.where(is_sink, 0.0, contrib)
    
    neighbor_sum = builder.core.neighbor_agg(contrib, agg="sum")
    
    sink_ranks = builder.core.where(is_sink, ranks, 0.0)
    sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
    sink_contrib = builder.core.broadcast_scalar(sink_mass, ranks)
    
    damped_neighbors = builder.core.mul(neighbor_sum, damping)
    damped_sinks = builder.core.mul(sink_contrib, damping / n)
    
    teleport = (1.0 - damping) / n
    
    ranks = builder.core.add(damped_neighbors, damped_sinks)
    ranks = builder.core.add(ranks, teleport)
    ranks = builder.var("ranks", ranks)

ranks = builder.var("ranks", builder.core.normalize_sum(ranks))
builder.attach_as("pagerank", ranks)

# Execute
algo = builder.build()
result_builder = graph.view().apply(algo)

print("\nBuilder PageRank:")
builder_values = {}
for node in result_builder.nodes:
    val = node.pagerank
    builder_values[node.id] = val
    print(f"  Node {node.id}: {val:.10f}")
total_builder = sum(builder_values.values())
print(f"Total: {total_builder:.10f}")

print("\nComparison:")
print(f"  Differences:")
max_diff = 0
for node_id in native_values:
    diff = abs(native_values[node_id] - builder_values[node_id])
    max_diff = max(max_diff, diff)
    print(f"    Node {node_id}: {diff:.10f}")
print(f"  Max diff: {max_diff:.10f}")

if max_diff < 0.0000005:
    print(f"  ✅ Results match (max diff < 0.0000005)")
else:
    print(f"  ⚠️  Results differ (max diff = {max_diff:.10f})")
