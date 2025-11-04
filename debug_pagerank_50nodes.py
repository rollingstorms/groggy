#!/usr/bin/env python
"""Debug PageRank on 50-node graph."""
import sys
sys.path.insert(0, 'python-groggy/python')

import groggy as gg
from groggy.algorithms import centrality
from benchmark_builder_vs_native import build_pagerank_algorithm
import random

random.seed(42)

# Create small test graph
g = gg.Graph()
nodes = [g.add_node() for _ in range(50)]

# Add random edges
for _ in range(100):
    src = random.choice(nodes)
    dst = random.choice(nodes)
    if src != dst:
        try:
            g.add_edge(src, dst)
        except:
            pass

print(f"Graph: {len(nodes)} nodes, {len(list(g.edges))} edges")

# Native
result_native = g.view().apply(centrality.pagerank(max_iter=20, damping=0.85), persist=True)
pr_native = {node.id: result_native.get_node_attribute(node.id, "pagerank") for node in result_native.nodes}

# Builder
algo_builder = build_pagerank_algorithm(n=50, max_iter=20, damping=0.85)
result_builder = g.view().apply(algo_builder)
pr_builder = {node.id: node.pagerank for node in result_builder.nodes}

# Compare
diffs = {nid: abs(pr_native[nid] - pr_builder[nid]) for nid in pr_native.keys()}
max_diff = max(diffs.values())
avg_diff = sum(diffs.values()) / len(diffs)

print(f"\nMax difference: {max_diff:.2e}")
print(f"Avg difference: {avg_diff:.2e}")

# Show worst offenders
worst = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"\nWorst 5 nodes:")
for nid, diff in worst:
    print(f"  Node {nid}: native={pr_native[nid]:.6f}, builder={pr_builder[nid]:.6f}, diff={diff:.6f}")

# Check normalization
sum_native = sum(pr_native.values())
sum_builder = sum(pr_builder.values())
print(f"\nSum native: {sum_native:.6f}")
print(f"Sum builder: {sum_builder:.6f}")

if max_diff > 5e-7:
    print(f"\n⚠️  FAILED: Max diff {max_diff:.2e} > 5e-07")
else:
    print(f"\n✅ PASSED: Max diff {max_diff:.2e} <= 5e-07")
