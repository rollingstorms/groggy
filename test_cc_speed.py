#!/usr/bin/env python3

import groggy
import time

print("Testing connected components speed...")

# Create a test graph like the benchmark does
g = groggy.Graph()

# Add nodes in batches - 1000 nodes first
print("Adding 1000 nodes...")
for i in range(1000):
    g.add_node(i)

# Add edges to create components
print("Adding 500 edges...")
for i in range(500):
    g.add_edge(i, i + 500)

print(f"Graph: {g.get_node_count()} nodes, {g.get_edge_count()} edges")

# Test the core connected_components call timing
print("Testing core connected_components()...")
start = time.time()
result = g.analytics().connected_components()
end = time.time()

print(f"connected_components() took: {end - start:.4f}s")
print(f"Number of components: {len(result)}")

# Now test just accessing len()
print("Testing len(result)...")
start = time.time()
count = len(result)
end = time.time()
print(f"len(result) took: {end - start:.6f}s")
print(f"Component count: {count}")

# Test accessing individual components
print("Testing access to first component...")
start = time.time()
first_comp = result[0]
end = time.time()
print(f"result[0] took: {end - start:.6f}s")

# Test node count on component
print("Testing first_comp.node_count()...")
start = time.time()
node_count = first_comp.node_count()
end = time.time()
print(f"first_comp.node_count() took: {end - start:.6f}s")
print(f"First component has {node_count} nodes")
