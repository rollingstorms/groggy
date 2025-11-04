"""Test just 200k PageRank."""
import time
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms import centrality
import random

def build_pr(n):
    builder = AlgorithmBuilder("custom_pagerank")
    ranks = builder.init_nodes(default=1.0 / n)
    ranks = builder.var("ranks", ranks)
    degrees = builder.node_degrees(ranks)
    inv_degrees = builder.core.recip(degrees, epsilon=1e-12)
    is_sink = builder.core.compare(degrees, "eq", 0.0)
    
    with builder.iterate(20):
        contrib = builder.core.mul(ranks, inv_degrees)
        contrib = builder.core.where(is_sink, 0.0, contrib)
        neighbor_sum = builder.core.neighbor_agg(contrib, agg="sum")
        sink_ranks = builder.core.where(is_sink, ranks, 0.0)
        sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
        sink_contrib = builder.core.broadcast_scalar(sink_mass, ranks)
        damped_neighbors = builder.core.mul(neighbor_sum, 0.85)
        damped_sinks = builder.core.mul(sink_contrib, 0.85 / n)
        teleport = (1.0 - 0.85) / n
        ranks = builder.core.add(damped_neighbors, damped_sinks)
        ranks = builder.core.add(ranks, teleport)
        ranks = builder.var("ranks", ranks)
    
    ranks = builder.var("ranks", builder.core.normalize_sum(ranks))
    builder.attach_as("pagerank", ranks)
    return builder.build()

random.seed(42)
n = 200000
graph = Graph()
nodes = [graph.add_node() for _ in range(n)]

num_edges = n * 10 // 2
edges_created = 0
attempts = 0
max_attempts = num_edges * 3

while edges_created < num_edges and attempts < max_attempts:
    src = random.choice(nodes)
    dst = random.choice(nodes)
    if src != dst:
        try:
            graph.add_edge(src, dst)
            graph.add_edge(dst, src)
            edges_created += 2
        except:
            pass
    attempts += 1

print(f"Created graph: {n} nodes, {edges_created} edges\n")
sg = graph.view()

print("Running native...")
result_native = sg.apply(centrality.pagerank(max_iter=20, damping=0.85, output_attr="pr"), persist=True)
native_nodes = list(result_native.nodes)

print("Running builder...")
algo = build_pr(n)
result_builder = sg.apply(algo)

diffs = []
for node in native_nodes:
    nid = node.id
    native_val = result_native.get_node_attribute(nid, "pr")
    builder_val = result_builder.get_node_attribute(nid, "pagerank")
    diffs.append(abs(native_val - builder_val))

print(f"Max diff: {max(diffs):.10f}")
print(f"Avg diff: {sum(diffs)/len(diffs):.10f}")

if max(diffs) < 1e-5:
    print("✅ Results match!")
else:
    print(f"⚠️  Results differ (max diff: {max(diffs):.10f})")
