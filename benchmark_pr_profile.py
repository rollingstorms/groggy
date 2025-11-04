"""Profile just the builder PageRank to identify bottlenecks."""
import time
from groggy import Graph, print_profile
from groggy.builder import AlgorithmBuilder
import random

def build_pagerank_algorithm(damping=0.85, max_iter=100):
    """Build PageRank using the builder DSL with proper primitives."""
    builder = AlgorithmBuilder("custom_pagerank")
    
    # Get node count from the graph at runtime
    node_count = builder.graph_node_count()
    
    # Initialize ranks uniformly
    ranks = builder.init_nodes(default=1.0)
    inv_n_scalar = builder.core.recip(node_count, epsilon=1e-9)
    uniform = builder.core.broadcast_scalar(inv_n_scalar, ranks)
    ranks = builder.var("ranks", uniform)
    
    # Compute out-degrees  
    degrees = builder.node_degrees(ranks)
    
    # Safe reciprocal for division
    inv_degrees = builder.core.recip(degrees, epsilon=1e-9)
    
    # Identify sinks
    is_sink = builder.core.compare(degrees, "eq", 0.0)
    
    with builder.iterate(max_iter):
        contrib = builder.core.mul(ranks, inv_degrees)
        contrib = builder.core.where(is_sink, 0.0, contrib)
        neighbor_sum = builder.core.neighbor_agg(contrib, agg="sum")
        damped_neighbors = builder.core.mul(neighbor_sum, damping)
        inv_n_map = builder.core.broadcast_scalar(inv_n_scalar, degrees)
        teleport_map = builder.core.mul(inv_n_map, 1.0 - damping)
        sink_ranks = builder.core.where(is_sink, ranks, 0.0)
        sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
        sink_map = builder.core.mul(inv_n_map, sink_mass)
        sink_map = builder.core.mul(sink_map, damping)
        updated = builder.core.add(damped_neighbors, teleport_map)
        updated = builder.core.add(updated, sink_map)
        ranks = builder.var("ranks", updated)
    
    ranks = builder.core.normalize_sum(ranks)
    builder.attach_as("pagerank", ranks)
    return builder.build()

def create_test_graph(num_nodes, avg_degree=10):
    random.seed(42)
    graph = Graph()
    nodes = [graph.add_node() for i in range(num_nodes)]
    num_edges = num_nodes * avg_degree // 2
    edges_data = []
    seen = set()
    
    for _ in range(num_edges * 2):
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)
        if i != j:
            edge = (min(i, j), max(i, j))
            if edge not in seen:
                edges_data.append((nodes[i], nodes[j]))
                seen.add(edge)
                if len(edges_data) >= num_edges:
                    break
    
    graph.add_edges(edges_data)
    print(f"Created graph: {num_nodes} nodes, {len(edges_data)} edges")
    return graph

print("Building 50k graph...")
graph = create_test_graph(50000, avg_degree=10)

print("\nRunning builder PageRank with profiling...")
algo = build_pagerank_algorithm(damping=0.85, max_iter=100)
sg = graph.view()

start = time.perf_counter()
result, profile = sg.apply(algo, return_profile=True)
elapsed = time.perf_counter() - start

print(f"\nTotal time: {elapsed:.3f}s")
print("\nDetailed profiling:")
print_profile(profile, show_steps=True, show_details=True)

# Show top node values
print("\nSample results:")
for i, node in enumerate(list(result.nodes)[:5]):
    print(f"  Node {node.id}: {node.pagerank:.8f}")
