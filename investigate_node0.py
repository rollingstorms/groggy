"""Investigate why node 0 diverges so much."""
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms.centrality import pagerank
import random

random.seed(42)
n = 1000
graph = Graph()
nodes = [graph.add_node() for _ in range(n)]

# Add edges
for _ in range(n * 5 // 2):
    src = random.choice(nodes)
    dst = random.choice(nodes)
    if src != dst:
        try:
            graph.add_edge(src, dst)
        except:
            pass

sg = graph.view()

# Check node 0 properties
node0_neighbors = []
for edge in sg.edges:
    if edge.source == 0:
        node0_neighbors.append(edge.target)
    elif edge.target == 0:
        node0_neighbors.append(edge.source)

node0_out = [e.target for e in sg.edges if e.source == 0]
node0_in = [e.source for e in sg.edges if e.target == 0]

print(f"Node 0 properties:")
print(f"  Out-degree: {len(node0_out)}")
print(f"  In-degree: {len(node0_in)}")
print(f"  Total neighbors: {len(set(node0_neighbors))}")

# Run both algorithms
result_native = sg.apply(pagerank(max_iter=20, damping=0.85))
native_rank_0 = result_native.get_node_attribute(0, "pagerank")

def build_pr(n):
    builder = AlgorithmBuilder("test_pr")
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
    
    builder.attach_as("pagerank", ranks)
    return builder.build()

algo = build_pr(n)
result_builder = sg.apply(algo)
builder_rank_0 = result_builder.get_node_attribute(0, "pagerank")

print(f"\nPageRank values:")
print(f"  Native:  {native_rank_0:.8f}")
print(f"  Builder: {builder_rank_0:.8f}")
print(f"  Diff:    {abs(native_rank_0 - builder_rank_0):.8f}")

# Check a few of node 0's neighbors
print(f"\nChecking node 0's neighbors' ranks:")
for neighbor_id in list(set(node0_neighbors))[:5]:
    native_val = result_native.get_node_attribute(neighbor_id, "pagerank")
    builder_val = result_builder.get_node_attribute(neighbor_id, "pagerank")
    print(f"  Node {neighbor_id}: Native={native_val:.8f}, Builder={builder_val:.8f}")

# Check if node 0 is part of a special subgraph
# Count total rank across all nodes
all_native = sum(result_native.get_node_attribute(nid, "pagerank") for nid in range(n))
all_builder = sum(result_builder.get_node_attribute(nid, "pagerank") for nid in range(n))

print(f"\nTotal rank:")
print(f"  Native:  {all_native:.10f}")
print(f"  Builder: {all_builder:.10f}")
