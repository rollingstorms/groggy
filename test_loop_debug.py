"""Debug loop variable threading."""
from groggy import Graph
from groggy.builder import AlgorithmBuilder
import json

# Create test graph: 3 nodes, no edges
graph = Graph()
a, b, c = graph.add_node(), graph.add_node(), graph.add_node()
sg = graph.view()

# Build simple PageRank (just sum neighbors, multiply by 0.85, add 0.15, normalize)
builder = AlgorithmBuilder("test_loop_debug")
ranks = builder.init_nodes(default=1.0)

print("Initial variable:")
print(f"  ranks = {ranks.name}")

with builder.iterate(3):
    # Sum neighbor ranks
    neighbor_sums = builder.map_nodes(
        "sum(ranks[neighbors(node)])",
        inputs={"ranks": ranks}
    )
    print(f"  neighbor_sums = {neighbor_sums.name}, inputs=ranks:{ranks.name}")
    
    # Apply damping
    damped = builder.core.mul(neighbor_sums, 0.85)
    print(f"  damped = {damped.name}")
    
    # Add teleport
    ranks = builder.var("ranks", builder.core.add(damped, 0.15))
    print(f"  ranks = {ranks.name} (after alias)")
    
    # Normalize
    ranks = builder.var("ranks", builder.core.normalize_sum(ranks))
    print(f"  ranks = {ranks.name} (after normalize)")

builder.attach_as("pagerank", ranks)

# Build
algo = builder.build()

# Check if we can get spec differently
print(f"\nAlgorithm: {algo}")
print(f"Algorithm type: {type(algo)}")
print(f"Algorithm attributes: {dir(algo)}")

# Execute
result = sg.apply(algo)

# Check results
print("\nPageRank values:")
for node in result.nodes:
    print(f"  Node {node.id}: {node.pagerank:.6f}")

# Expected: all nodes should have equal rank (1/3 each) since no edges
# If loop isn't threading variables correctly, we'll see non-uniform values
total = sum(node.pagerank for node in result.nodes)
print(f"\nTotal rank: {total:.6f}")
print(f"Expected per node: {1.0/3.0:.6f}")
print(f"Actual avg: {total/3:.6f}")

# Check if values are uniform
values = [node.pagerank for node in result.nodes]
if all(abs(v - 1.0/3.0) < 0.01 for v in values):
    print("✅ Values are uniform (no edges graph working correctly)")
else:
    print(f"⚠️  Values are NOT uniform: {values}")
