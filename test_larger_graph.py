"""Test with larger graph."""
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms.centrality import pagerank
import random

random.seed(42)

# Create random graph
n = 100
graph = Graph()
nodes = [graph.add_node() for _ in range(n)]

# Add random edges
for _ in range(n * 5):  # ~10 edges per node (5 in each direction)
    src = random.choice(nodes)
    dst = random.choice(nodes)
    if src != dst:
        try:
            graph.add_edge(src, dst)
            graph.add_edge(dst, src)
        except:
            pass

sg = graph.view()

damping = 0.85
max_iter = 20

# Native PageRank
result_native = sg.apply(pagerank(max_iter=max_iter, damping=damping, output_attr="pr_native"), persist=True)

native_values = {node.id: node.pr_native for node in result_native.nodes}

# Builder PageRank
builder = AlgorithmBuilder("test_larger")

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

algo = builder.build()
result_builder = graph.view().apply(algo)

builder_values = {node.id: node.pagerank for node in result_builder.nodes}

# Compare
diffs = [abs(native_values[nid] - builder_values[nid]) for nid in native_values]
max_diff = max(diffs)
avg_diff = sum(diffs) / len(diffs)

print(f"100-node graph:")
print(f"  Avg diff: {avg_diff:.10f}")
print(f"  Max diff: {max_diff:.10f}")

# Sample values
sample_ids = sorted(list(native_values.keys()))[:5]
print(f"\nSample values:")
for nid in sample_ids:
    print(f"  Node {nid}: native={native_values[nid]:.10f}, builder={builder_values[nid]:.10f}, diff={abs(native_values[nid]-builder_values[nid]):.10f}")

if max_diff < 0.0000005:
    print(f"\n✅ Results match (max diff < 0.0000005)")
else:
    print(f"\n⚠️  Results differ (max diff = {max_diff:.10f})")
