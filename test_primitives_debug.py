"""Debug primitives-based PageRank."""
from groggy import Graph
from groggy.builder import AlgorithmBuilder

# Create test graph: 3 nodes, no edges
graph = Graph()
a, b, c = graph.add_node(), graph.add_node(), graph.add_node()
sg = graph.view()

n = 3
damping = 0.85

# Build PageRank using primitives
builder = AlgorithmBuilder("test_primitives")

# Initialize ranks uniformly
ranks = builder.init_nodes(default=1.0 / n)
ranks = builder.var("ranks", ranks)

# Compute out-degrees
degrees = builder.node_degrees(ranks)

# Safe reciprocal
inv_degrees = builder.core.recip(degrees, epsilon=1e-12)

# Identify sinks
is_sink = builder.core.compare(degrees, "eq", 0.0)

print(f"Initial ranks: {ranks.name}")
print(f"Degrees: {degrees.name}")
print(f"Inv degrees: {inv_degrees.name}")
print(f"Is sink: {is_sink.name}")

with builder.iterate(10):
    print(f"\nIteration starting with ranks: {ranks.name}")
    
    # Compute contribution: rank / out_degree
    contrib = builder.core.mul(ranks, inv_degrees)
    print(f"  contrib = {contrib.name}")
    
    contrib = builder.core.where(is_sink, 0.0, contrib)
    print(f"  contrib (after where) = {contrib.name}")
    
    # Sum neighbor contributions
    neighbor_sum = builder.core.neighbor_agg(contrib, agg="sum")
    print(f"  neighbor_sum = {neighbor_sum.name}")
    
    # Handle sink redistribution
    sink_ranks = builder.core.where(is_sink, ranks, 0.0)
    print(f"  sink_ranks = {sink_ranks.name}")
    
    sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
    print(f"  sink_mass = {sink_mass.name}")
    
    sink_contrib = builder.core.broadcast_scalar(sink_mass, ranks)
    print(f"  sink_contrib = {sink_contrib.name}")
    
    # Apply damping
    damped_neighbors = builder.core.mul(neighbor_sum, damping)
    print(f"  damped_neighbors = {damped_neighbors.name}")
    
    damped_sinks = builder.core.mul(sink_contrib, damping / n)
    print(f"  damped_sinks = {damped_sinks.name}")
    
    # Add teleport
    teleport = (1.0 - damping) / n
    
    # Combine
    ranks = builder.core.add(damped_neighbors, damped_sinks)
    print(f"  ranks (after adding sinks) = {ranks.name}")
    
    ranks = builder.core.add(ranks, teleport)
    print(f"  ranks (after teleport) = {ranks.name}")
    
    ranks = builder.var("ranks", ranks)
    print(f"  ranks (after alias) = {ranks.name}")

# Normalize once after iterations
ranks = builder.var("ranks", builder.core.normalize_sum(ranks))

builder.attach_as("pagerank", ranks)

# Execute
algo = builder.build()
result = sg.apply(algo)

# Check results
print("\n" + "="*60)
print("PageRank values:")
for node in result.nodes:
    print(f"  Node {node.id}: {node.pagerank:.10f}")

total = sum(node.pagerank for node in result.nodes)
print(f"\nTotal rank: {total:.10f}")
print(f"Expected per node: {1.0/3.0:.10f}")

# Check if values are uniform
values = [node.pagerank for node in result.nodes]
if all(abs(v - 1.0/3.0) < 0.0001 for v in values):
    print("✅ Values are uniform")
else:
    print(f"⚠️  Values are NOT uniform")
    print(f"  Differences from expected:")
    for i, v in enumerate(values):
        print(f"    Node {i}: diff = {v - 1.0/3.0:.10f}")
