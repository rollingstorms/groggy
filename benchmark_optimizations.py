#!/usr/bin/env python3
"""
Benchmark script to measure algorithm optimizations.
Compares before/after performance for Connected Components and PageRank.
"""

import time
import groggy as gg
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def create_test_graph(n_nodes: int, avg_degree: int = 3) -> gg.Graph:
    """Create a random graph for testing."""
    g = gg.Graph()
    
    # Add nodes
    nodes = [g.add_node() for _ in range(n_nodes)]
    
    # Add edges to create connected components with average degree
    n_edges = (n_nodes * avg_degree) // 2
    edges_added = 0
    
    # First, ensure connectivity by creating a spanning tree
    for i in range(1, n_nodes):
        parent = np.random.randint(0, i)
        g.add_edge(nodes[i], nodes[parent])
        g.add_edge(nodes[parent], nodes[i])  # Make it undirected
        edges_added += 1
    
    # Add random edges to reach target degree
    while edges_added < n_edges:
        src = np.random.randint(0, n_nodes)
        dst = np.random.randint(0, n_nodes)
        if src != dst:
            try:
                g.add_edge(nodes[src], nodes[dst])
                g.add_edge(nodes[dst], nodes[src])
                edges_added += 1
            except:
                pass  # Edge already exists
    
    return g

def benchmark_connected_components(sizes: List[int], trials: int = 5) -> pd.DataFrame:
    """Benchmark Connected Components algorithm."""
    print("\n=== Benchmarking Connected Components ===")
    results = []
    
    for n_nodes in sizes:
        print(f"\nTesting with {n_nodes:,} nodes...")
        
        # Create test graph
        g = create_test_graph(n_nodes)
        n_edges = g.edge_count()
        print(f"  Graph: {n_nodes:,} nodes, {n_edges:,} edges")
        
        times = []
        for trial in range(trials):
            start = time.perf_counter()
            components = g.connected_components(mode='undirected')
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to milliseconds
            
            if trial == 0:
                n_components = len(set(components.values()))
                print(f"  Found {n_components} components")
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Time: {mean_time:.2f} ± {std_time:.2f} ms")
        
        results.append({
            'algorithm': 'Connected Components',
            'nodes': n_nodes,
            'edges': n_edges,
            'mean_ms': mean_time,
            'std_ms': std_time,
            'min_ms': min(times),
            'max_ms': max(times)
        })
    
    return pd.DataFrame(results)

def benchmark_pagerank(sizes: List[int], trials: int = 5) -> pd.DataFrame:
    """Benchmark PageRank algorithm."""
    print("\n=== Benchmarking PageRank ===")
    results = []
    
    for n_nodes in sizes:
        print(f"\nTesting with {n_nodes:,} nodes...")
        
        # Create test graph
        g = create_test_graph(n_nodes)
        n_edges = g.edge_count()
        print(f"  Graph: {n_nodes:,} nodes, {n_edges:,} edges")
        
        times = []
        for trial in range(trials):
            start = time.perf_counter()
            ranks = g.pagerank(damping=0.85, max_iter=50, tolerance=1e-6)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to milliseconds
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Time: {mean_time:.2f} ± {std_time:.2f} ms")
        
        results.append({
            'algorithm': 'PageRank',
            'nodes': n_nodes,
            'edges': n_edges,
            'mean_ms': mean_time,
            'std_ms': std_time,
            'min_ms': min(times),
            'max_ms': max(times)
        })
    
    return pd.DataFrame(results)

def benchmark_label_propagation(sizes: List[int], trials: int = 5) -> pd.DataFrame:
    """Benchmark Label Propagation algorithm."""
    print("\n=== Benchmarking Label Propagation ===")
    results = []
    
    for n_nodes in sizes:
        print(f"\nTesting with {n_nodes:,} nodes...")
        
        # Create test graph
        g = create_test_graph(n_nodes)
        n_edges = g.edge_count()
        print(f"  Graph: {n_nodes:,} nodes, {n_edges:,} edges")
        
        times = []
        for trial in range(trials):
            start = time.perf_counter()
            communities = g.label_propagation(max_iter=25, tolerance=0.001)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to milliseconds
            
            if trial == 0:
                n_communities = len(set(communities.values()))
                print(f"  Found {n_communities} communities")
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Time: {mean_time:.2f} ± {std_time:.2f} ms")
        
        results.append({
            'algorithm': 'Label Propagation',
            'nodes': n_nodes,
            'edges': n_edges,
            'mean_ms': mean_time,
            'std_ms': std_time,
            'min_ms': min(times),
            'max_ms': max(times)
        })
    
    return pd.DataFrame(results)

def main():
    """Run benchmarks and save results."""
    print("=" * 60)
    print("Groggy Algorithm Optimization Benchmarks")
    print("=" * 60)
    
    # Test sizes (start small to verify, then scale up)
    sizes = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
    
    # Run benchmarks
    results = []
    
    try:
        cc_results = benchmark_connected_components(sizes)
        results.append(cc_results)
    except Exception as e:
        print(f"Error in Connected Components benchmark: {e}")
    
    try:
        pr_results = benchmark_pagerank(sizes)
        results.append(pr_results)
    except Exception as e:
        print(f"Error in PageRank benchmark: {e}")
    
    try:
        lpa_results = benchmark_label_propagation(sizes)
        results.append(lpa_results)
    except Exception as e:
        print(f"Error in Label Propagation benchmark: {e}")
    
    # Combine results
    if results:
        all_results = pd.concat(results, ignore_index=True)
        
        # Save to CSV
        output_file = 'optimization_benchmark_results.csv'
        all_results.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for algo in all_results['algorithm'].unique():
            algo_data = all_results[all_results['algorithm'] == algo]
            print(f"\n{algo}:")
            for _, row in algo_data.iterrows():
                print(f"  {row['nodes']:>7,} nodes: {row['mean_ms']:>8.2f} ms")
            
            # Calculate scaling
            if len(algo_data) >= 2:
                first = algo_data.iloc[0]
                last = algo_data.iloc[-1]
                node_ratio = last['nodes'] / first['nodes']
                time_ratio = last['mean_ms'] / first['mean_ms']
                print(f"  Scaling: {node_ratio:.1f}x nodes → {time_ratio:.1f}x time")

if __name__ == '__main__':
    main()
