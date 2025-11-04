"""Test PageRank accuracy as graph size increases."""
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms.centrality import pagerank
import random

def build_pr(n, damping=0.85, max_iter=20):
    builder = AlgorithmBuilder("test_pr")
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
    
    builder.attach_as("pagerank", ranks)
    return builder.build()

def test_size(n, avg_deg=10):
    random.seed(42)
    graph = Graph(directed=True)
    nodes = [graph.add_node() for _ in range(n)]
    
    # Add edges
    for _ in range(n * avg_deg // 2):
        src = random.choice(nodes)
        dst = random.choice(nodes)
        if src != dst:
            try:
                graph.add_edge(src, dst)
            except:
                pass
    
    sg = graph.view()
    
    # Native
    result_native = sg.apply(pagerank(max_iter=20, damping=0.85))
    native_ranks = {node.id: result_native.get_node_attribute(node.id, "pagerank") for node in sg.nodes}
    
    # Builder
    algo = build_pr(n)
    result_builder = sg.apply(algo)
    builder_ranks = {node.id: node.pagerank for node in result_builder.nodes}
    
    # Compare
    diffs = [abs(native_ranks[nid] - builder_ranks[nid]) for nid in native_ranks]
    max_diff = max(diffs)
    avg_diff = sum(diffs) / len(diffs)
    
    return max_diff, avg_diff

# Test increasing sizes
sizes = [50, 100, 500, 1000, 5000, 10000]
print("Graph Size | Max Diff       | Avg Diff")
print("-" * 45)
for n in sizes:
    max_d, avg_d = test_size(n)
    print(f"{n:10d} | {max_d:14.10f} | {avg_d:14.10f}")
