"""Debug the _pagerank_step function vs working implementation."""
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms.centrality import pagerank

def _pagerank_step(builder, ranks, node_count, damping=0.85):
    """From test file."""
    degrees = builder.node_degrees(ranks)
    inv_degrees = builder.core.recip(degrees, epsilon=1e-9)
    is_sink = builder.core.compare(degrees, "eq", 0.0)

    weighted = builder.core.mul(ranks, inv_degrees)
    weighted = builder.core.where(is_sink, 0.0, weighted)

    neighbor_sums = builder.core.neighbor_agg(weighted, agg="sum")

    damped = builder.core.mul(neighbor_sums, damping)

    inv_n_scalar = builder.core.recip(node_count, epsilon=1e-9)
    inv_n_map = builder.core.broadcast_scalar(inv_n_scalar, degrees)
    teleport_map = builder.core.mul(inv_n_map, 1.0 - damping)

    sink_ranks = builder.core.where(is_sink, ranks, 0.0)
    sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
    sink_map = builder.core.mul(inv_n_map, sink_mass)
    sink_map = builder.core.mul(sink_map, damping)

    total = builder.core.add(damped, teleport_map)
    total = builder.core.add(total, sink_map)
    ranks = builder.var("ranks", total)
    return ranks

# Create test graph
graph = Graph(directed=True)
nodes = [graph.add_node() for _ in range(5)]
graph.add_edge(nodes[0], nodes[1])
graph.add_edge(nodes[1], nodes[2])
graph.add_edge(nodes[2], nodes[0])
graph.add_edge(nodes[2], nodes[3])
graph.add_edge(nodes[3], nodes[4])
graph.add_edge(nodes[4], nodes[2])

sg = graph.view()

# Build PageRank with test's _pagerank_step
builder = AlgorithmBuilder("test_pr_step")
ranks = builder.init_nodes(default=1.0)
node_count = builder.graph_node_count()
inv_n_scalar = builder.core.recip(node_count, epsilon=1e-9)
uniform = builder.core.broadcast_scalar(inv_n_scalar, ranks)
ranks = builder.var("ranks", uniform)

with builder.iterate(20):
    ranks = _pagerank_step(builder, ranks, node_count, damping=0.85)

builder.attach_as("pagerank", ranks)

algo = builder.build()
result_builder = sg.apply(algo)

# Run native
result_native = graph.view().apply(pagerank(max_iter=20, damping=0.85))

print("Native vs Test's Builder:")
for node in sorted(result_native.nodes, key=lambda x: x.id):
    native_pr = result_native.get_node_attribute(node.id, "pagerank")
    builder_pr = result_builder.get_node_attribute(node.id, "pagerank")
    diff = abs(native_pr - builder_pr)
    print(f"  Node {node.id}: native={native_pr:.10f}, builder={builder_pr:.10f}, diff={diff:.10f}")
