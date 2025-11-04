#!/usr/bin/env python3
"""
Benchmark Suite for Parallel Algorithm Performance

Tests different graph structures and sizes to identify:
1. When parallel provides speedup vs overhead
2. Memory usage characteristics
3. Convergence patterns
4. Optimal threshold for enabling parallel mode
"""

import groggy as gg
from groggy.algorithms import centrality
import time
import random
import csv
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    graph_type: str
    n_nodes: int
    n_edges: int
    avg_degree: float
    max_iter: int
    actual_iter: int
    serial_time: float
    parallel_time: float
    speedup: float
    serial_core_time: float
    parallel_core_time: float
    
    def to_dict(self) -> Dict:
        return {
            'graph_type': self.graph_type,
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'avg_degree': f'{self.avg_degree:.1f}',
            'max_iter': self.max_iter,
            'actual_iter': self.actual_iter,
            'serial_time_ms': f'{self.serial_time*1000:.2f}',
            'parallel_time_ms': f'{self.parallel_time*1000:.2f}',
            'speedup': f'{self.speedup:.2f}',
            'serial_core_ms': f'{self.serial_core_time*1000:.2f}',
            'parallel_core_ms': f'{self.parallel_core_time*1000:.2f}',
        }

def create_ring_graph(n_nodes: int, cross_links: int = 0) -> tuple:
    """Create a ring graph with optional cross-links."""
    g = gg.Graph()
    nodes = [g.add_node() for _ in range(n_nodes)]
    
    # Ring structure
    n_edges = 0
    for i in range(len(nodes)):
        g.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])
        n_edges += 1
    
    # Add cross-links
    if cross_links > 0:
        random.seed(42)
        for _ in range(cross_links):
            src = random.randint(0, len(nodes) - 1)
            dst = random.randint(0, len(nodes) - 1)
            if src != dst:
                g.add_edge(nodes[src], nodes[dst])
                n_edges += 1
    
    return g, nodes, n_edges

def create_random_graph(n_nodes: int, avg_degree: int) -> tuple:
    """Create a random graph with target average degree."""
    g = gg.Graph()
    nodes = [g.add_node() for _ in range(n_nodes)]
    
    target_edges = (n_nodes * avg_degree) // 2
    n_edges = 0
    random.seed(42)
    
    for _ in range(target_edges):
        src = random.randint(0, len(nodes) - 1)
        dst = random.randint(0, len(nodes) - 1)
        if src != dst:
            g.add_edge(nodes[src], nodes[dst])
            n_edges += 1
    
    return g, nodes, n_edges

def benchmark_pagerank(
    graph_type: str,
    n_nodes: int,
    create_func,
    max_iter: int = 50,
    **create_kwargs
) -> BenchmarkResult:
    """Benchmark PageRank on a specific graph configuration."""
    
    # Create graph
    g, nodes, n_edges = create_func(n_nodes, **create_kwargs)
    sg = g.induced_subgraph(nodes)
    avg_degree = (2 * n_edges) / n_nodes
    
    # Serial benchmark
    start = time.time()
    result_serial, stats_serial = sg.apply(
        centrality.pagerank(max_iter=max_iter, tolerance=1e-6, output_attr='pr_s'),
        parallel=False,
        return_profile=True
    )
    serial_time = time.time() - start
    
    # Parallel benchmark
    start = time.time()
    result_parallel, stats_parallel = sg.apply(
        centrality.pagerank(max_iter=max_iter, tolerance=1e-6, output_attr='pr_p'),
        parallel=True,
        return_profile=True
    )
    parallel_time = time.time() - start
    
    # Calculate speedup
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    
    return BenchmarkResult(
        graph_type=graph_type,
        n_nodes=n_nodes,
        n_edges=n_edges,
        avg_degree=avg_degree,
        max_iter=max_iter,
        actual_iter=max_iter,  # Would need to track actual convergence
        serial_time=serial_time,
        parallel_time=parallel_time,
        speedup=speedup,
        serial_core_time=stats_serial['run_time'],
        parallel_core_time=stats_parallel['run_time']
    )

def run_benchmark_suite() -> List[BenchmarkResult]:
    """Run comprehensive benchmark suite."""
    results = []
    
    print("="*70)
    print("Parallel Algorithm Benchmark Suite")
    print("="*70)
    
    # Test 1: Sparse rings (worst case for parallelism - high overhead, low work)
    print("\n1. Sparse Rings (avg degree ~2)")
    print("-"*70)
    for n_nodes in [100, 500, 1000, 5000, 10000]:
        print(f"  Testing {n_nodes:,} nodes...", end="", flush=True)
        result = benchmark_pagerank('ring', n_nodes, create_ring_graph)
        results.append(result)
        print(f" {result.speedup:.2f}x")
    
    # Test 2: Rings with cross-links (moderate connectivity)
    print("\n2. Rings with Cross-links (avg degree ~10)")
    print("-"*70)
    for n_nodes in [100, 500, 1000, 5000, 10000]:
        cross_links = n_nodes * 4  # Adds ~8 to average degree
        print(f"  Testing {n_nodes:,} nodes...", end="", flush=True)
        result = benchmark_pagerank('ring+cross', n_nodes, create_ring_graph, cross_links=cross_links)
        results.append(result)
        print(f" {result.speedup:.2f}x")
    
    # Test 3: Random graphs (varying density)
    print("\n3. Random Graphs (varying density)")
    print("-"*70)
    for avg_degree in [5, 10, 20, 50]:
        n_nodes = 5000
        print(f"  Testing {n_nodes:,} nodes, degree {avg_degree}...", end="", flush=True)
        result = benchmark_pagerank(
            f'random_d{avg_degree}',
            n_nodes,
            create_random_graph,
            avg_degree=avg_degree
        )
        results.append(result)
        print(f" {result.speedup:.2f}x")
    
    # Test 4: Varying iterations (does more work help?)
    print("\n4. Varying Iterations (5K nodes, avg degree 10)")
    print("-"*70)
    for max_iter in [10, 25, 50, 100, 200]:
        print(f"  Testing {max_iter} iterations...", end="", flush=True)
        result = benchmark_pagerank(
            f'iter{max_iter}',
            5000,
            create_ring_graph,
            cross_links=20000,
            max_iter=max_iter
        )
        results.append(result)
        print(f" {result.speedup:.2f}x")
    
    return results

def analyze_results(results: List[BenchmarkResult]):
    """Analyze and report findings."""
    print("\n" + "="*70)
    print("Analysis")
    print("="*70)
    
    # Find best speedup
    best = max(results, key=lambda r: r.speedup)
    print(f"\nBest speedup: {best.speedup:.2f}x")
    print(f"  Graph: {best.graph_type}, {best.n_nodes:,} nodes, {best.avg_degree:.1f} avg degree")
    
    # Find worst slowdown
    worst = min(results, key=lambda r: r.speedup)
    print(f"\nWorst slowdown: {worst.speedup:.2f}x")
    print(f"  Graph: {worst.graph_type}, {worst.n_nodes:,} nodes, {worst.avg_degree:.1f} avg degree")
    
    # Count wins vs losses
    wins = sum(1 for r in results if r.speedup > 1.0)
    losses = sum(1 for r in results if r.speedup < 1.0)
    
    print(f"\nOverall:")
    print(f"  Parallel faster: {wins}/{len(results)} cases")
    print(f"  Parallel slower: {losses}/{len(results)} cases")
    
    # Identify threshold
    if losses > wins:
        print(f"\n⚠️  Parallel mode is slower in majority of cases")
        print(f"    Current implementation not ready for general use")
    else:
        print(f"\n✅ Parallel mode shows promise")
        
        # Find threshold
        by_size = sorted([r for r in results if 'ring' in r.graph_type], 
                        key=lambda r: r.n_nodes)
        for r in by_size:
            if r.speedup > 1.0:
                print(f"    Speedup starts at ~{r.n_nodes:,} nodes")
                break

def save_results(results: List[BenchmarkResult], filename: str):
    """Save results to CSV."""
    with open(filename, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
            writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())
    print(f"\n✓ Results saved to {filename}")

def main():
    try:
        results = run_benchmark_suite()
        analyze_results(results)
        save_results(results, 'parallel_benchmark_results.csv')
        
        print("\n" + "="*70)
        print("Benchmark Complete")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
