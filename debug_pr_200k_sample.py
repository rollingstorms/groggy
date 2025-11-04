"""Debug a subset of the 200k graph to understand divergence."""
import random
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms import centrality


def build_pagerank_algorithm(damping=0.85, max_iter=100):
    """Build PageRank using the builder DSL with proper primitives."""
    builder = AlgorithmBuilder("custom_pagerank")
    
    # Get node count from the graph at runtime
    node_count = builder.graph_node_count()
    
    # Initialize ranks uniformly (will be 1/N at runtime)
    ranks = builder.init_nodes(default=1.0)
    inv_n_scalar = builder.core.recip(node_count, epsilon=1e-9)
    uniform = builder.core.broadcast_scalar(inv_n_scalar, ranks)
    ranks = builder.var("ranks", uniform)
    
    # Compute out-degrees  
    degrees = builder.node_degrees(ranks)
    
    # Safe reciprocal for division (avoid division by zero)
    inv_degrees = builder.core.recip(degrees, epsilon=1e-9)
    
    # Identify sinks (nodes with no outgoing edges)
    is_sink = builder.core.compare(degrees, "eq", 0.0)
    
    with builder.iterate(max_iter):
        # Compute contribution from each node: rank / out_degree
        contrib = builder.core.mul(ranks, inv_degrees)
        contrib = builder.core.where(is_sink, 0.0, contrib)
        
        # Sum neighbor contributions (via incoming edges)
        neighbor_sum = builder.core.neighbor_agg(contrib, agg="sum")
        
        # Apply damping to neighbor contributions
        damped_neighbors = builder.core.mul(neighbor_sum, damping)
        
        # Compute teleport term: (1-damping)/N broadcast to all nodes
        inv_n_map = builder.core.broadcast_scalar(inv_n_scalar, degrees)
        teleport_map = builder.core.mul(inv_n_map, 1.0 - damping)
        
        # Handle sink redistribution: collect rank from sinks and redistribute
        sink_ranks = builder.core.where(is_sink, ranks, 0.0)
        sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
        sink_map = builder.core.mul(inv_n_map, sink_mass)
        sink_map = builder.core.mul(sink_map, damping)
        
        # Combine all components
        updated = builder.core.add(damped_neighbors, teleport_map)
        updated = builder.core.add(updated, sink_map)
        ranks = builder.var("ranks", updated)
    
    # Normalize once after iterations to ensure sum = 1.0
    ranks = builder.core.normalize_sum(ranks)
    
    builder.attach_as("pagerank", ranks)
    return builder.build()


def create_test_graph(num_nodes, avg_degree=10):
    """Create a random graph for testing."""
    random.seed(42)
    
    graph = Graph()
    nodes = [graph.add_node() for _ in range(num_nodes)]
    
    # Create random edges
    num_edges = num_nodes * avg_degree // 2
    edges_created = 0
    attempts = 0
    max_attempts = num_edges * 3
    
    while edges_created < num_edges and attempts < max_attempts:
        src = random.choice(nodes)
        dst = random.choice(nodes)
        if src != dst:
            try:
                graph.add_edge(src, dst)
                graph.add_edge(dst, src)  # Make undirected
                edges_created += 2
            except:
                pass
        attempts += 1
    
    print(f"Created graph: {num_nodes} nodes, ~{edges_created} edges")
    return graph


# Test on 1000 node graph
print("Building 1000 node graph...")
graph = create_test_graph(1000, avg_degree=10)

damping = 0.85

# Native
print("\nNative PageRank:")
sg = graph.view()
result_native, stats_native = sg.apply(
    centrality.pagerank(max_iter=100, damping=damping, output_attr="pr_native"),
    persist=True,
    return_profile=True
)

native_nodes = list(result_native.nodes)
native_map = {node.id: node.pr_native for node in native_nodes}

# Get stats
native_values = list(native_map.values())
native_sum = sum(native_values)
native_min = min(native_values)
native_max = max(native_values)
native_mean = native_sum / len(native_values)

print(f"  Sum: {native_sum:.10f}")
print(f"  Min: {native_min:.10f}")
print(f"  Max: {native_max:.10f}")
print(f"  Mean: {native_mean:.10f}")

# Top 5
sorted_native = sorted(native_map.items(), key=lambda x: x[1], reverse=True)[:5]
print("  Top 5 nodes:")
for node_id, val in sorted_native:
    print(f"    Node {node_id}: {val:.10f}")

# Builder
print("\nBuilder PageRank:")
algo = build_pagerank_algorithm(damping=damping, max_iter=100)
result_builder = sg.apply(algo)

builder_map = {node.id: node.pagerank for node in result_builder.nodes}

# Get stats
builder_values = list(builder_map.values())
builder_sum = sum(builder_values)
builder_min = min(builder_values)
builder_max = max(builder_values)
builder_mean = builder_sum / len(builder_values)

print(f"  Sum: {builder_sum:.10f}")
print(f"  Min: {builder_min:.10f}")
print(f"  Max: {builder_max:.10f}")
print(f"  Mean: {builder_mean:.10f}")

# Top 5
sorted_builder = sorted(builder_map.items(), key=lambda x: x[1], reverse=True)[:5]
print("  Top 5 nodes:")
for node_id, val in sorted_builder:
    print(f"    Node {node_id}: {val:.10f}")

# Compare
print("\nComparison:")
diffs = []
for node_id in native_map:
    native_val = native_map[node_id]
    builder_val = builder_map[node_id]
    diff = abs(native_val - builder_val)
    diffs.append((node_id, native_val, builder_val, diff))

diffs.sort(key=lambda x: x[3], reverse=True)

avg_diff = sum(d[3] for d in diffs) / len(diffs)
max_diff = diffs[0][3]

print(f"  Avg diff: {avg_diff:.10e}")
print(f"  Max diff: {max_diff:.10e}")
print(f"\n  Top 5 differences:")
for node_id, native_val, builder_val, diff in diffs[:5]:
    print(f"    Node {node_id}: native={native_val:.10f}, builder={builder_val:.10f}, diff={diff:.10e}")
