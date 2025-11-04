"""Debug PageRank with loop"""
import groggy as gg

g = gg.Graph(directed=True)
g.add_nodes(3)
g.add_edges([(0, 1), (1, 2)])

print("Graph: 0->1->2\n")

# With loop (1 iteration)
builder = gg.AlgorithmBuilder("with_loop")
ranks = builder.init_nodes(default=1.0/3.0)
degrees = builder.node_degrees(ranks)
safe_deg = builder.core.clip(degrees, min_value=1.0)

with builder.iterate(1):  # Just 1 iteration
    contrib = builder.core.div(ranks, safe_deg)
    neighbor_sum = builder.core.neighbor_agg(contrib, agg="sum")
    damped = builder.core.mul(neighbor_sum, 0.85)
    teleport = builder.core.add(damped, 0.15/3.0)
    ranks = builder.core.normalize_sum(teleport)

builder.attach_as("pagerank", ranks)
builder.attach_as("contrib", contrib)
builder.attach_as("neighbor_sum", neighbor_sum)

result = g.apply(builder.build())
print("With 1 iteration loop:")
print(f"  Contrib: {[result.get_node_attribute(i, 'contrib') for i in range(3)]}")
print(f"  Neighbor sums: {[result.get_node_attribute(i, 'neighbor_sum') for i in range(3)]}")
print(f"  PageRank: {[result.get_node_attribute(i, 'pagerank') for i in range(3)]}")

# Without loop
builder2 = gg.AlgorithmBuilder("without_loop")
ranks2 = builder2.init_nodes(default=1.0/3.0)
degrees2 = builder2.node_degrees(ranks2)
safe_deg2 = builder2.core.clip(degrees2, min_value=1.0)
contrib2 = builder2.core.div(ranks2, safe_deg2)
neighbor_sum2 = builder2.core.neighbor_agg(contrib2, agg="sum")
damped2 = builder2.core.mul(neighbor_sum2, 0.85)
teleport2 = builder2.core.add(damped2, 0.15/3.0)
ranks_final = builder2.core.normalize_sum(teleport2)

builder2.attach_as("pagerank", ranks_final)
builder2.attach_as("contrib", contrib2)
builder2.attach_as("neighbor_sum", neighbor_sum2)

result2 = g.apply(builder2.build())
print("\nWithout loop:")
print(f"  Contrib: {[result2.get_node_attribute(i, 'contrib') for i in range(3)]}")
print(f"  Neighbor sums: {[result2.get_node_attribute(i, 'neighbor_sum') for i in range(3)]}")
print(f"  PageRank: {[result2.get_node_attribute(i, 'pagerank') for i in range(3)]}")
