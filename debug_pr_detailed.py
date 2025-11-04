"""Debug PageRank divergence in detail."""
import random
from groggy import Graph
from groggy.builder import AlgorithmBuilder

def build_debug_pagerank(damping=0.85, max_iter=1):
    """One iteration only for debugging."""
    builder = AlgorithmBuilder("debug_pr")
    
    node_count = builder.graph_node_count()
    
    # Initialize ranks uniformly
    ranks = builder.init_nodes(default=1.0)
    inv_n_scalar = builder.core.recip(node_count, epsilon=1e-9)
    uniform = builder.core.broadcast_scalar(inv_n_scalar, ranks)
    ranks = builder.var("ranks", uniform)
    
    # Compute out-degrees  
    degrees = builder.node_degrees(ranks)
    
    # Safe reciprocal
    inv_degrees = builder.core.recip(degrees, epsilon=1e-9)
    
    # Identify sinks
    is_sink = builder.core.compare(degrees, "eq", 0.0)
    
    # Attach debug info
    builder.attach_as("init_ranks", ranks)
    builder.attach_as("degrees", degrees)
    builder.attach_as("inv_degrees", inv_degrees)
    builder.attach_as("is_sink", is_sink)
    
    with builder.iterate(max_iter):
        # Contribution
        contrib = builder.core.mul(ranks, inv_degrees)
        contrib = builder.core.where(is_sink, 0.0, contrib)
        
        builder.attach_as("contrib", contrib)
        
        # Neighbor sum
        neighbor_sum = builder.core.neighbor_agg(contrib, agg="sum")
        
        builder.attach_as("neighbor_sum", neighbor_sum)
        
        # Damped neighbors
        damped_neighbors = builder.core.mul(neighbor_sum, damping)
        
        # Teleport term
        inv_n_map = builder.core.broadcast_scalar(inv_n_scalar, degrees)
        teleport_map = builder.core.mul(inv_n_map, 1.0 - damping)
        
        builder.attach_as("teleport_map", teleport_map)
        
        # Sink mass
        sink_ranks = builder.core.where(is_sink, ranks, 0.0)
        sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
        sink_map = builder.core.mul(inv_n_map, sink_mass)
        sink_map = builder.core.mul(sink_map, damping)
        
        builder.attach_as("sink_map", sink_map)
        
        # Combine
        updated = builder.core.add(damped_neighbors, teleport_map)
        updated = builder.core.add(updated, sink_map)
        ranks = builder.var("ranks", updated)
    
    # No final normalization for debugging
    builder.attach_as("final_ranks", ranks)
    return builder.build()


# Create a simple test graph
random.seed(42)
g = Graph(directed=True)

# Add 5 nodes
nodes = [g.add_node() for _ in range(5)]
print("Nodes:", nodes)

# Add edges: 0->1, 1->2, 2->3, 3->4, 4->0 (cycle)
edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0)
]
for src_idx, tgt_idx in edges:
    g.add_edge(nodes[src_idx], nodes[tgt_idx])

print("\nEdges (src->tgt):")
for src_idx, tgt_idx in edges:
    print(f"  {nodes[src_idx]} -> {nodes[tgt_idx]}")

# Run builder version
algo = build_debug_pagerank(damping=0.85, max_iter=1)
result = g.view().apply(algo)

# Print all debug attachments
print("\nBuilder Debug Output:")
print("=" * 60)

for attr_name in ["init_ranks", "degrees", "inv_degrees", "is_sink", 
                  "contrib", "neighbor_sum", "teleport_map", "sink_map", "final_ranks"]:
    print(f"\n{attr_name}:")
    for node in nodes:
        val = result.get_node_attribute(node, attr_name)
        print(f"  Node {node}: {val}")

# Run native version for comparison
print("\n\nNative PageRank (1 iteration):")
print("=" * 60)
result_native = g.pagerank(damping=0.85, max_iterations=1)
for node in nodes:
    val = result_native.get_node_attribute(node, "pagerank")
    print(f"  Node {node}: {val}")

# Manual calculation
print("\n\nManual Calculation:")
print("=" * 60)
n = 5
damping = 0.85
init_rank = 1.0 / n
print(f"Initial rank: {init_rank}")

# Out-degrees
out_deg = {i: 1 for i in range(5)}  # Each has out-degree 1
print(f"Out-degrees: {out_deg}")

# Contributions
contrib_manual = {i: init_rank / out_deg[i] for i in range(5)}
print(f"Contributions (rank/degree): {contrib_manual}")

# Neighbor sums (incoming contributions)
# Node 0 receives from node 4
# Node 1 receives from node 0
# Node 2 receives from node 1
# Node 3 receives from node 2
# Node 4 receives from node 3
neighbor_sums = {
    0: contrib_manual[4],
    1: contrib_manual[0],
    2: contrib_manual[1],
    3: contrib_manual[2],
    4: contrib_manual[3]
}
print(f"Neighbor sums: {neighbor_sums}")

# Final ranks
damped = {i: damping * neighbor_sums[i] for i in range(5)}
teleport = (1.0 - damping) / n
sink_contrib = 0.0  # No sinks
final_manual = {i: damped[i] + teleport + sink_contrib for i in range(5)}
print(f"Final ranks: {final_manual}")

print(f"\nAll should be: {damping * init_rank + teleport} = {damping * init_rank + teleport}")
