"""Run just the 50k PageRank benchmark."""
import time
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms import centrality
import random

def build_pagerank_algorithm(n, damping=0.85, max_iter=20):
    """Build PageRank using the builder DSL with proper primitives."""
    builder = AlgorithmBuilder("custom_pagerank")
    
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
    return builder.build()

def create_test_graph(num_nodes, avg_degree=10):
    """Create a random graph for testing."""
    random.seed(42)
    
    graph = Graph()
    nodes = [graph.add_node() for _ in range(num_nodes)]
    
    num_edges = num_nodes * avg_degree // 2
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
    
    print(f"Created graph: {num_nodes} nodes, ~{edges_created} edges")
    return graph

print("Building 50k node graph...")
graph = create_test_graph(50000, avg_degree=10)
sg = graph.view()
n = len(list(sg.nodes))

# Native
print("\nNative PageRank...")
result_native = sg.apply(centrality.pagerank(max_iter=20, damping=0.85, output_attr="pagerank_native"), persist=True)
native_nodes = list(result_native.nodes)
sample_ids = [node.id for node in native_nodes[:5]]

print(f"Sample native values:")
for nid in sample_ids:
    val = result_native.get_node_attribute(nid, "pagerank_native")
    print(f"  Node {nid}: {val:.8f}")

# Builder
print("\nBuilder PageRank...")
algo = build_pagerank_algorithm(n=n, damping=0.85, max_iter=20)
result_builder = sg.apply(algo)

print(f"Sample builder values:")
for nid in sample_ids:
    val = result_builder.get_node_attribute(nid, "pagerank")
    print(f"  Node {nid}: {val:.8f}")

# Compare
diffs = []
for node in native_nodes:
    nid = node.id
    native_val = result_native.get_node_attribute(nid, "pagerank_native")
    builder_val = result_builder.get_node_attribute(nid, "pagerank")
    diff = abs(native_val - builder_val)
    diffs.append((nid, native_val, builder_val, diff))

diffs.sort(key=lambda x: x[3], reverse=True)

print(f"\nTop 10 worst matches:")
for nid, native_val, builder_val, diff in diffs[:10]:
    print(f"  Node {nid}: Native={native_val:.8f}, Builder={builder_val:.8f}, Diff={diff:.8f}")

print(f"\nMax diff: {max(d[3] for d in diffs):.10f}")
print(f"Avg diff: {sum(d[3] for d in diffs)/len(diffs):.10f}")
