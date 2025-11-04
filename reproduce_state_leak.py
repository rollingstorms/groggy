"""
Minimal reproduction of builder PageRank state leakage bug.

Run this script to see:
- First execution after running algo1 on graph1: FAILS (wrong values)
- Second execution on same graph: PASSES (correct values)

This demonstrates cross-graph state contamination in builder algorithms.
"""

from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms.centrality import pagerank


def build_pagerank(name, max_iter=1):
    """Build a PageRank algorithm using builder primitives."""
    builder = AlgorithmBuilder(name)
    ranks = builder.init_nodes(default=1.0)
    node_count = builder.graph_node_count()
    inv_n_scalar = builder.core.recip(node_count, epsilon=1e-9)
    uniform = builder.core.broadcast_scalar(inv_n_scalar, ranks)
    ranks = builder.var("ranks", uniform)
    
    with builder.iterate(max_iter):
        degrees = builder.node_degrees(ranks)
        inv_degrees = builder.core.recip(degrees, epsilon=1e-9)
        is_sink = builder.core.compare(degrees, "eq", 0.0)
        
        weighted = builder.core.mul(ranks, inv_degrees)
        weighted = builder.core.where(is_sink, 0.0, weighted)
        
        neighbor_sums = builder.core.neighbor_agg(weighted, agg="sum")
        
        damped = builder.core.mul(neighbor_sums, 0.85)
        
        inv_n_map = builder.core.broadcast_scalar(inv_n_scalar, degrees)
        teleport_map = builder.core.mul(inv_n_map, 0.15)
        
        sink_ranks = builder.core.where(is_sink, ranks, 0.0)
        sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
        sink_map = builder.core.mul(inv_n_map, sink_mass)
        sink_map = builder.core.mul(sink_map, 0.85)
        
        total = builder.core.add(damped, teleport_map)
        total = builder.core.add(total, sink_map)
        ranks = builder.var("ranks", total)
    
    ranks = builder.core.normalize_sum(ranks)
    builder.attach_as("pagerank", ranks)
    return builder.build()


# Create two independent graphs
graph1 = Graph(directed=True)
a, b, c = graph1.add_node(), graph1.add_node(), graph1.add_node()
graph1.add_edge(a, b)
graph1.add_edge(b, c)
graph1.add_edge(c, a)

graph2 = Graph(directed=True)
nodes = [graph2.add_node() for _ in range(5)]
graph2.add_edge(nodes[0], nodes[1])
graph2.add_edge(nodes[1], nodes[2])
graph2.add_edge(nodes[2], nodes[0])
graph2.add_edge(nodes[2], nodes[3])
graph2.add_edge(nodes[3], nodes[4])
graph2.add_edge(nodes[4], nodes[2])

# Step 1: Run algo1 on graph1 (this contaminates state)
print("Step 1: Running algorithm on graph1...")
algo1 = build_pagerank("algo1")
result1 = graph1.view().apply(algo1)
print(f"✓ Completed")

# Step 2: Run algo2 on graph2 (FIRST time - will FAIL)
print("\nStep 2: Running algorithm on graph2 (1st time)...")
algo2 = build_pagerank("algo2")
result2_builder = graph2.view().apply(algo2)
builder_values = {node.id: node.pagerank for node in result2_builder.nodes}

# Get expected values from native implementation
result2_native = graph2.view().apply(pagerank(max_iter=1, damping=0.85))
native_values = {node.id: result2_native.get_node_attribute(node.id, "pagerank") 
                 for node in result2_native.nodes}

# Compare
max_diff = max(abs(builder_values[nid] - native_values[nid]) for nid in builder_values.keys())
if max_diff < 1e-6:
    print(f"✅ PASSED: max difference = {max_diff:.2e}")
else:
    print(f"❌ FAILED: max difference = {max_diff:.2e}")
    print("\nDetailed comparison:")
    for nid in sorted(builder_values.keys()):
        diff = abs(builder_values[nid] - native_values[nid])
        print(f"  Node {nid}: builder={builder_values[nid]:.6f}, native={native_values[nid]:.6f}, diff={diff:.2e}")

# Step 3: Run algo2 on graph2 again (SECOND time - will PASS)
print("\nStep 3: Running algorithm on graph2 (2nd time)...")
result2_second = graph2.view().apply(algo2)
builder_values2 = {node.id: node.pagerank for node in result2_second.nodes}

max_diff2 = max(abs(builder_values2[nid] - native_values[nid]) for nid in builder_values2.keys())
if max_diff2 < 1e-6:
    print(f"✅ PASSED: max difference = {max_diff2:.2e}")
else:
    print(f"❌ FAILED: max difference = {max_diff2:.2e}")

print("\n" + "="*60)
if max_diff > 1e-6 and max_diff2 < 1e-6:
    print("BUG REPRODUCED: First run failed, second run passed!")
    print("This confirms cross-graph state leakage in builder algorithms.")
else:
    print("Bug NOT reproduced (state may have been cleared elsewhere)")
