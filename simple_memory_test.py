#!/usr/bin/env python3
import sys
import psutil
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')
import groggy as gr

def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

print("Testing small graph memory...")
g = gr.Graph()

# Start with small test
print(f"Empty graph: {get_memory_mb():.2f} MB")
print(f"Empty graph info: {g.info()}")

# Add 10 nodes
node_ids = [f'n{i}' for i in range(10)]
g.nodes.add(node_ids)
print(f"After 10 nodes: {get_memory_mb():.2f} MB")
print(f"Nodes info: {g.info()}")

# Add 5 edges (make sure nodes exist)
edge_tuples = []
for i in range(5):
    src = f'n{i}'
    tgt = f'n{i+1}'
    edge_tuples.append((src, tgt))

print(f"Adding edges: {edge_tuples}")
g.edges.add(edge_tuples)
print(f"After 5 edges: {get_memory_mb():.2f} MB")
print(f"Final info: {g.info()}")