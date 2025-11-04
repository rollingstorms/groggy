"""Test applying algorithm to the same graph multiple times"""
import groggy as gg

g = gg.Graph(directed=True)
g.add_nodes(3)
g.add_edges([(0, 1), (1, 2)])

# Build algo once
builder = gg.AlgorithmBuilder("test")
ranks = builder.init_nodes(default=1.0/3.0)
degrees = builder.node_degrees(ranks)
safe_deg = builder.core.clip(degrees, min_value=1.0)
contrib = builder.core.div(ranks, safe_deg)
neighbor_sum = builder.core.neighbor_agg(contrib)
builder.attach_as("neighbor_sum", neighbor_sum)
algo = builder.build()

# Apply it twice
print("First application:")
result1 = g.apply(algo)
vals1 = [result1.get_node_attribute(i, 'neighbor_sum') for i in range(3)]
print(f"  Neighbor sums: {vals1}")

print("\nSecond application (same graph, same algo):")
result2 = g.apply(algo)
vals2 = [result2.get_node_attribute(i, 'neighbor_sum') for i in range(3)]
print(f"  Neighbor sums: {vals2}")

print("\nThird application:")
result3 = g.apply(algo)
vals3 = [result3.get_node_attribute(i, 'neighbor_sum') for i in range(3)]
print(f"  Neighbor sums: {vals3}")

if vals1 != vals2 or vals2 != vals3:
    print("\n❌ VALUES DIFFER - STATE LEAK DETECTED!")
else:
    print("\n✓ Values are consistent")
