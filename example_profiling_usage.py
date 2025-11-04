"""
Example demonstrating profiling feature usage.

This script shows how to use the new print_profile() function
to analyze algorithm performance.
"""

import groggy as gg
from groggy.algorithms import centrality

# Create a test graph
print("Creating test graph (100 nodes)...")
g = gg.Graph(directed=True)
edges = [(i, (i + 1) % 100) for i in range(100)]
g.add_edges(edges)
sg = g.to_subgraph()

print("\n" + "="*80)
print("Example 1: Native PageRank with Basic Profiling")
print("="*80)

# Run algorithm with profiling enabled
result, profile = sg.apply(
    centrality.pagerank(max_iter=10, damping=0.85, output_attr="pr"),
    persist=True,
    return_profile=True
)

# Display profiling information
gg.print_profile(profile)

print("\n" + "="*80)
print("Example 2: Builder Algorithm with Profiling")
print("="*80)

# Create a simple builder algorithm
from groggy import builder

b = builder("simple_mult")
nodes = b.init_nodes(default=2.0)
doubled = b.core.mul(nodes, nodes)  # Square each value
b.attach_as("squared", doubled)

algo = b.build()
result2, profile2 = sg.apply(algo, return_profile=True)

# Show basic profiling
gg.print_profile(profile2)

print("\n" + "="*80)
print("Example 3: Multiple Algorithms in Pipeline")
print("="*80)

# Create a pipeline
from groggy import pipeline

pipe = pipeline([
    centrality.pagerank(max_iter=5, output_attr="pr"),
    algo,  # Our builder algorithm
])

result3, profile3 = pipe.run(sg, return_profile=True)

# Show profiling for the full pipeline
gg.print_profile(profile3, show_steps=True)

print("\n✅ All profiling examples completed successfully!")
print("\nKey Takeaways:")
print("  - Use return_profile=True on .apply() to get profiling data")
print("  - Use gg.print_profile() to display formatted profiling results")
print("  - Set show_steps=True to see per-step timing breakdown")
print("  - Set show_details=True for detailed call counters and stats")
