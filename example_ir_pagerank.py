"""
Example: Visualizing PageRank algorithm with IR

This demonstrates the new IR visualization capabilities.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

from groggy.builder.ir import IRGraph, CoreIRNode, GraphIRNode, AttrIRNode

# Build a simplified PageRank IR manually
graph = IRGraph("pagerank_simple")

# Step 1: Initialize ranks
node1 = AttrIRNode("n1", "load", [], "ranks", name="ranks", default=1.0)
graph.add_node(node1)

# Step 2: Get degrees
node2 = GraphIRNode("n2", "degree", [], "degrees")
graph.add_node(node2)

# Step 3: Compute inverse degrees (for normalization)
node3 = CoreIRNode("n3", "recip", ["degrees"], "inv_deg", epsilon=1e-9)
graph.add_node(node3)

# Step 4: Multiply ranks by inverse degrees
node4 = CoreIRNode("n4", "mul", ["ranks", "inv_deg"], "contrib")
graph.add_node(node4)

# Step 5: Aggregate neighbor contributions
node5 = GraphIRNode("n5", "neighbor_agg", ["contrib"], "neighbor_sum", agg="sum")
graph.add_node(node5)

# Step 6: Apply damping factor (0.85 * neighbor_sum)
node6 = CoreIRNode("n6", "mul", ["neighbor_sum"], "damped", b=0.85)
graph.add_node(node6)

# Step 7: Add teleport term (0.15 / N)
# Simplified - assume teleport is pre-computed
node7 = CoreIRNode("n7", "add", ["damped"], "new_ranks", b=0.15)
graph.add_node(node7)

# Step 8: Attach result as attribute
node8 = AttrIRNode("n8", "attach", ["new_ranks"], None, name="pagerank")
graph.add_node(node8)

print("=" * 70)
print("PageRank Algorithm IR Visualization")
print("=" * 70)
print()

# Show statistics
print("Statistics:")
stats = graph.stats()
for key, value in stats.items():
    if key != "operation_types":
        print(f"  {key}: {value}")
print()

print("Operation types:")
for op, count in stats["operation_types"].items():
    print(f"  {op}: {count}")
print()

# Show pretty printed version
print("=" * 70)
print("Algorithm Structure (Pretty Print):")
print("=" * 70)
print(graph.pretty_print())
print()

# Show DOT format
print("=" * 70)
print("Graphviz DOT Format (for visualization):")
print("=" * 70)
print(graph.to_dot())
print()

print("=" * 70)
print("To visualize this graph:")
print("  1. Copy the DOT output above")
print("  2. Save to pagerank.dot")
print("  3. Run: dot -Tpng pagerank.dot -o pagerank.png")
print("  4. Or paste into: https://dreampuf.github.io/GraphvizOnline/")
print("=" * 70)
