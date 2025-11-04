"""Test PageRank on directed vs undirected graphs to identify edge handling issues."""
import groggy as gg
from groggy.algorithms import centrality
import numpy as np

def build_simple_pr(n, damping=0.85, max_iter=100):
    """Simple PageRank builder for testing."""
    builder = gg.AlgorithmBuilder("test_pr")
    ranks = builder.init_nodes(default=1.0/n)
    degrees = builder.node_degrees(ranks)
    safe_deg = builder.core.clip(degrees, min_value=1.0)
    
    with builder.iterate(max_iter):
        contrib = builder.core.div(ranks, safe_deg)
        neighbor_sum = builder.core.neighbor_agg(contrib)
        damped = builder.core.mul(neighbor_sum, damping)
        teleport = builder.core.add(damped, (1.0 - damping)/n)
        ranks = builder.core.normalize_sum(teleport)
    
    builder.attach_as("pagerank", ranks)
    return builder.build()

def test_simple_chain():
    """Test on simple 3-node chain: 0->1->2"""
    print("\n=== DIRECTED CHAIN: 0->1->2 ===")
    g_dir = gg.Graph(directed=True)
    g_dir.add_nodes(3)
    g_dir.add_edges([(0, 1), (1, 2)])
    
    # Native
    pr_algo = centrality.pagerank(damping=0.85, max_iter=100, tolerance=1e-6)
    native = g_dir.apply(pr_algo)
    native_vals = [native.get_node_attribute(i, 'pagerank') for i in range(3)]
    print(f"Native:  {native_vals}")
    
    # Builder
    algo = build_simple_pr(3, damping=0.85, max_iter=100)
    result = g_dir.apply(algo)
    builder_vals = [result.get_node_attribute(i, 'pagerank') for i in range(3)]
    print(f"Builder: {builder_vals}")
    print(f"Out-degrees (directed): {[g_dir.out_degree(i) for i in range(3)]}")
    
    diffs = [abs(n - b) for n, b in zip(native_vals, builder_vals)]
    print(f"Differences: {diffs}, Max: {max(diffs):.2e}")
    
    print("\n=== UNDIRECTED CHAIN: 0-1-2 ===")
    g_undir = gg.Graph(directed=False)
    g_undir.add_nodes(3)
    g_undir.add_edges([(0, 1), (1, 2)])
    
    native_u = g_undir.apply(pr_algo)
    native_u_vals = [native_u.get_node_attribute(i, 'pagerank') for i in range(3)]
    print(f"Native:  {native_u_vals}")
    
    algo_u = build_simple_pr(3, damping=0.85, max_iter=100)
    result_u = g_undir.apply(algo_u)
    builder_u_vals = [result_u.get_node_attribute(i, 'pagerank') for i in range(3)]
    print(f"Builder: {builder_u_vals}")
    print(f"Out-degrees (undirected): {[g_undir.out_degree(i) for i in range(3)]}")
    
    diffs_u = [abs(n - b) for n, b in zip(native_u_vals, builder_u_vals)]
    print(f"Differences: {diffs_u}, Max: {max(diffs_u):.2e}")

def test_star_graph():
    """Test on star: center(0) connected to 1,2,3"""
    print("\n\n=== DIRECTED STAR: 0->1, 0->2, 0->3 ===")
    g_dir = gg.Graph(directed=True)
    g_dir.add_nodes(4)
    g_dir.add_edges([(0, 1), (0, 2), (0, 3)])
    
    pr_algo = centrality.pagerank(damping=0.85, max_iter=100, tolerance=1e-6)
    native = g_dir.apply(pr_algo)
    native_vals = [native.get_node_attribute(i, 'pagerank') for i in range(4)]
    print(f"Native:  {native_vals}")
    
    algo = build_simple_pr(4, damping=0.85, max_iter=100)
    result = g_dir.apply(algo)
    builder_vals = [result.get_node_attribute(i, 'pagerank') for i in range(4)]
    print(f"Builder: {builder_vals}")
    print(f"Out-degrees: {[g_dir.out_degree(i) for i in range(4)]}")
    
    diffs = [abs(n - b) for n, b in zip(native_vals, builder_vals)]
    print(f"Differences: {diffs}, Max: {max(diffs):.2e}")
    
    print("\n=== UNDIRECTED STAR ===")
    g_undir = gg.Graph(directed=False)
    g_undir.add_nodes(4)
    g_undir.add_edges([(0, 1), (0, 2), (0, 3)])
    
    native_u = g_undir.apply(pr_algo)
    native_u_vals = [native_u.get_node_attribute(i, 'pagerank') for i in range(4)]
    print(f"Native:  {native_u_vals}")
    
    result_u = g_undir.apply(algo)
    builder_u_vals = [result_u.get_node_attribute(i, 'pagerank') for i in range(4)]
    print(f"Builder: {builder_u_vals}")
    print(f"Out-degrees: {[g_undir.out_degree(i) for i in range(4)]}")
    
    diffs_u = [abs(n - b) for n, b in zip(native_u_vals, builder_u_vals)]
    print(f"Differences: {diffs_u}, Max: {max(diffs_u):.2e}")

def test_with_detailed_steps():
    """Show step-by-step values for first iteration"""
    print("\n\n=== DETAILED STEP TRACE (directed 0->1->2) ===")
    g = gg.Graph(directed=True)
    g.add_nodes(3)
    g.add_edges([(0, 1), (1, 2)])
    
    print(f"Initial ranks: [1/3, 1/3, 1/3]")
    print(f"Out-degrees: {[g.out_degree(i) for i in range(3)]}")
    
    # Manual first iteration
    ranks = np.array([1/3, 1/3, 1/3])
    degrees = np.array([1.0, 1.0, 1.0])  # clipped
    contrib = ranks / degrees
    print(f"Contrib (rank/degree): {contrib}")
    
    # Neighbor sums (incoming contributions)
    # Node 0: receives from no one -> 0
    # Node 1: receives from 0 -> contrib[0]
    # Node 2: receives from 1 -> contrib[1]
    neighbor_sum = np.array([0.0, contrib[0], contrib[1]])
    print(f"Neighbor sums (incoming): {neighbor_sum}")
    
    damped = neighbor_sum * 0.85
    teleport = damped + 0.15/3
    print(f"After damping + teleport: {teleport}")
    
    normalized = teleport / teleport.sum()
    print(f"After normalization: {normalized}")
    
    # Builder trace
    builder = gg.AlgorithmBuilder("pr_trace")
    n = 3
    ranks_b = builder.init_nodes(default=1.0/n)
    degrees_b = builder.node_degrees(ranks_b)
    safe_deg = builder.core.clip(degrees_b, min_value=1.0)
    
    contrib_b = builder.core.div(ranks_b, safe_deg)
    neighbor_sum_b = builder.core.neighbor_agg(contrib_b)
    damped_b = builder.core.mul(neighbor_sum_b, 0.85)
    teleport_b = builder.core.add(damped_b, 0.15/n)
    ranks_final = builder.core.normalize_sum(teleport_b)
    
    builder.attach_as("pagerank", ranks_final)
    builder.attach_as("debug_contrib", contrib_b)
    builder.attach_as("debug_neighbor_sum", neighbor_sum_b)
    builder.attach_as("debug_damped", damped_b)
    builder.attach_as("debug_teleport", teleport_b)
    
    result = g.apply(builder.build())
    print(f"\nBuilder results:")
    print(f"  Contrib: {[result.get_node_attribute(i, 'debug_contrib') for i in range(3)]}")
    print(f"  Neighbor sums: {[result.get_node_attribute(i, 'debug_neighbor_sum') for i in range(3)]}")
    print(f"  Damped: {[result.get_node_attribute(i, 'debug_damped') for i in range(3)]}")
    print(f"  Teleport: {[result.get_node_attribute(i, 'debug_teleport') for i in range(3)]}")
    print(f"  Final: {[result.get_node_attribute(i, 'pagerank') for i in range(3)]}")

if __name__ == "__main__":
    test_simple_chain()
    test_star_graph()
    test_with_detailed_steps()
