"""Minimal reproduction of state leak between algorithm runs"""
import groggy as gg

g = gg.Graph(directed=True)
g.add_nodes(3)
g.add_edges([(0, 1), (1, 2)])

# First run - simple PageRank
builder1 = gg.AlgorithmBuilder("test1")
ranks1 = builder1.init_nodes(default=1.0/3.0)
degrees1 = builder1.node_degrees(ranks1)
safe_deg1 = builder1.core.clip(degrees1, min_value=1.0)
with builder1.iterate(100):
    contrib1 = builder1.core.div(ranks1, safe_deg1)
    neighbor_sum1 = builder1.core.neighbor_agg(contrib1)
    damped1 = builder1.core.mul(neighbor_sum1, 0.85)
    teleport1 = builder1.core.add(damped1, 0.15/3.0)
    ranks1 = builder1.core.normalize_sum(teleport1)
builder1.attach_as("pagerank", ranks1)

result1 = g.apply(builder1.build())
print("First run:")
print(f"  PageRank: {[result1.get_node_attribute(i, 'pagerank') for i in range(3)]}")

# Second run - with debug outputs (should give same result)
builder2 = gg.AlgorithmBuilder("test2")
ranks2 = builder2.init_nodes(default=1.0/3.0)
degrees2 = builder2.node_degrees(ranks2)
safe_deg2 = builder2.core.clip(degrees2, min_value=1.0)

contrib2 = builder2.core.div(ranks2, safe_deg2)
neighbor_sum2 = builder2.core.neighbor_agg(contrib2)
damped2 = builder2.core.mul(neighbor_sum2, 0.85)
teleport2 = builder2.core.add(damped2, 0.15/3.0)
ranks_final2 = builder2.core.normalize_sum(teleport2)

builder2.attach_as("pagerank", ranks_final2)
builder2.attach_as("debug_neighbor_sum", neighbor_sum2)

result2 = g.apply(builder2.build())
print("\nSecond run (single iteration with debug):")
print(f"  Neighbor sums: {[result2.get_node_attribute(i, 'debug_neighbor_sum') for i in range(3)]}")
print(f"  PageRank: {[result2.get_node_attribute(i, 'pagerank') for i in range(3)]}")

print("\nExpected neighbor_sum: [0.0, 0.333, 0.333]")
print("If we see [0.0, 0.666, 0.666] then state leaked from first run!")
