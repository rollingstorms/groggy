"""Debug PageRank step by step"""
import groggy as gg

g = gg.Graph(directed=True)
g.add_nodes(3)
g.add_edges([(0, 1), (1, 2)])

print("Graph: 0->1->2")
print(f"Out-degrees: {[g.out_degree(i) for i in range(3)]}")
print(f"In-degrees: {[g.in_degree(i) for i in range(3)]}\n")

# Step 1: Initialize ranks
builder = gg.AlgorithmBuilder("step1")
ranks = builder.init_nodes(default=1.0/3.0)
builder.attach_as("ranks", ranks)
result = g.apply(builder.build())
print("Step 1 - Initial ranks:")
for i in range(3):
    print(f"  Node {i}: {result.get_node_attribute(i, 'ranks')}")

# Step 2: Get degrees
builder2 = gg.AlgorithmBuilder("step2")
ranks2 = builder2.init_nodes(default=1.0/3.0)
degrees = builder2.node_degrees(ranks2)
builder2.attach_as("degrees", degrees)
result2 = g.apply(builder2.build())
print("\nStep 2 - Degrees:")
for i in range(3):
    print(f"  Node {i}: {result2.get_node_attribute(i, 'degrees')}")

# Step 3: Clip degrees
builder3 = gg.AlgorithmBuilder("step3")
ranks3 = builder3.init_nodes(default=1.0/3.0)
degrees3 = builder3.node_degrees(ranks3)
safe_deg = builder3.core.clip(degrees3, min_value=1.0)
builder3.attach_as("safe_deg", safe_deg)
result3 = g.apply(builder3.build())
print("\nStep 3 - Safe degrees (clipped to min 1.0):")
for i in range(3):
    print(f"  Node {i}: {result3.get_node_attribute(i, 'safe_deg')}")

# Step 4: Compute contrib = ranks / safe_deg
builder4 = gg.AlgorithmBuilder("step4")
ranks4 = builder4.init_nodes(default=1.0/3.0)
degrees4 = builder4.node_degrees(ranks4)
safe_deg4 = builder4.core.clip(degrees4, min_value=1.0)
contrib = builder4.core.div(ranks4, safe_deg4)
builder4.attach_as("contrib", contrib)
result4 = g.apply(builder4.build())
print("\nStep 4 - Contrib (ranks / safe_deg):")
for i in range(3):
    print(f"  Node {i}: {result4.get_node_attribute(i, 'contrib')}")

# Step 5: Aggregate neighbors
builder5 = gg.AlgorithmBuilder("step5")
ranks5 = builder5.init_nodes(default=1.0/3.0)
degrees5 = builder5.node_degrees(ranks5)
safe_deg5 = builder5.core.clip(degrees5, min_value=1.0)
contrib5 = builder5.core.div(ranks5, safe_deg5)
neighbor_sum = builder5.core.neighbor_agg(contrib5, agg="sum")
builder5.attach_as("neighbor_sum", neighbor_sum)
result5 = g.apply(builder5.build())
print("\nStep 5 - Neighbor aggregation (sum of contrib from in-neighbors):")
for i in range(3):
    val = result5.get_node_attribute(i, 'neighbor_sum')
    print(f"  Node {i}: {val}")
print("\nExpected:")
print(f"  Node 0: 0.0 (no in-neighbors)")
print(f"  Node 1: 0.333... (from node 0)")
print(f"  Node 2: 0.333... (from node 1)")
