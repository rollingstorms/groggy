#!/usr/bin/env python3
"""
Detailed profiling demonstration for Connected Components algorithm.

This script shows the granular profiling capabilities, including:
- Phase-by-phase timing with call counts
- Node/edge processing statistics  
- Memory allocation tracking
- Cache hit/miss statistics
- Per-component metrics

Set GROGGY_PROFILE_CC=1 environment variable to enable detailed profiling output.
"""

import os
import groggy as gr
from groggy import algorithms
from groggy.pipeline import Pipeline, AlgorithmHandle


def _run_cc_pipeline(graph, *, mode: str, output_attr: str):
    algo = algorithms.community.connected_components(mode=mode, output_attr=output_attr)
    pipe = Pipeline([algo])
    return pipe(graph.view(), persist=False, return_profile=True)


def _print_phase_table(title: str, rows, unit: str = "ms", scale: float = 1000.0):
    if not rows:
        return
    print(f"\n{title}")
    print("-" * 75)
    for name, value in rows:
        print(f"  {name:<40}{value * scale:>12.3f} {unit}")


def _print_call_counters(counters):
    if not counters:
        return
    print("\nAlgorithm Call Counters")
    print("-" * 75)
    for name, data in counters:
        count = data.get("count", 0)
        total = data.get("total", 0.0)
        avg = data.get("avg", 0.0)
        print(
            f"  {name:<40}count={count:<8} total={total*1000:7.3f} ms  avg={avg*1e6:7.3f} µs"
        )


def _print_stats(stats):
    if not stats:
        return
    print("\nAlgorithm Statistics")
    print("-" * 75)
    for name, value in stats:
        print(f"  {name:<40}{value:>12.3f}")

# Enable detailed profiling
os.environ['GROGGY_PROFILE_CC'] = '1'

def profile_large_graph():
    """Profile connected components on a substantial graph."""
    print("=" * 80)
    print("Connected Components - Detailed Profiling Demonstration")
    print("=" * 80)
    
    
    # Create a graph with multiple components of varying sizes
    print("Building test graph...")
    g = gr.Graph()
    
    # Component 1: Large component (1000 nodes in a path)
    nodes1 = g.add_nodes(100000)
    edge_data = [(nodes1[i], nodes1[i + 1]) for i in range(99999)]
    g.add_edges(edge_data)
    
    # Component 2: Medium component (500 nodes in a dense mesh)
    nodes2 = g.add_nodes(50000)
    edge_data = [(nodes2[i], nodes2[j]) for i in range(0, 50000, 10) for j in range(i, min(i + 10, 50000)) if j > i]
    g.add_edges(edge_data)
    
    # Component 3: Small components (10 components of 50 nodes each)
    for comp in range(10):
        nodes_small = g.add_nodes(5000)
        edge_data = [(nodes_small[i], nodes_small[i + 1]) for i in range(4999)]
        g.add_edges(edge_data)
    
    total_nodes = 100000 + 50000 + 50000
    total_edges = len(list(g.edges.ids()))
    expected_components = 1 + 1 + 10  # 12 total
    
    print(f"Graph created:")
    print(f"  Nodes: {total_nodes:,}")
    print(f"  Edges: {total_edges:,}")
    print(f"  Expected components: {expected_components}")
    
    
    # Run algorithm with profiling enabled
    print("Running Connected Components algorithm...")
    print("(Profiling report will be printed by the algorithm)")
    
    _, profile = _run_cc_pipeline(
        g,
        mode="undirected",
        output_attr="component",
    )

    timers = profile.get("timers", {})
    _print_phase_table(
        "Algorithm Phase Timings",
        sorted(timers.items(), key=lambda item: item[1], reverse=True),
    )

    ffi_timers = profile.get("ffi_timers", {})
    _print_phase_table(
        "FFI Timings",
        sorted(ffi_timers.items(), key=lambda item: item[1], reverse=True),
    )

    call_counters = profile.get("call_counters", {})
    _print_call_counters(sorted(call_counters.items()))

    stats = profile.get("stats", {})
    _print_stats(sorted(stats.items()))
    
    # print("Algorithm completed successfully!")
    
    # # Verify results
    # components_found = len(result.nodes['component'].unique())
    # print(f"Components found: {components_found}")
    # print(f"Match expected: {'✓' if components_found == expected_components else '✗'}")
    

def profile_strong_components():
    """Profile strongly connected components (Tarjan's algorithm)."""
    print("=" * 80)
    print("Strongly Connected Components - Tarjan's Algorithm Profiling")
    print("=" * 80)
    
    
    g = gr.Graph(directed=True)
    
    print("Building directed graph with cycles...")
    
    # Create a large strongly connected component (cycle)
    nodes_cycle = g.add_nodes(50000)
    edge_data = [(nodes_cycle[i], nodes_cycle[(i + 1) % 50000]) for i in range(50000)]
    g.add_edges(edge_data)
    
    # Create a DAG (each node is its own SCC)
    nodes_dag = g.add_nodes(300)
    edge_data = [(nodes_dag[i], nodes_dag[i + 1]) for i in range(299)]
    g.add_edges(edge_data)
    
    print(f"Graph created:")
    print(f"  Nodes: {len(list(g.nodes.ids())):,}")
    print(f"  Edges: {len(list(g.edges.ids())):,}")
    print(f"  Expected SCCs: ~301 (1 cycle + 300 individual nodes)")
    
    
    print("Running Strongly Connected Components (Tarjan)...")
    
    
    _, profile = _run_cc_pipeline(
        g,
        mode="strong",
        output_attr="scc",
    )

    timers = profile.get("timers", {})
    _print_phase_table(
        "Algorithm Phase Timings (Strong)",
        sorted(timers.items(), key=lambda item: item[1], reverse=True),
    )

    ffi_timers = profile.get("ffi_timers", {})
    _print_phase_table(
        "FFI Timings", sorted(ffi_timers.items(), key=lambda item: item[1], reverse=True)
    )

    call_counters = profile.get("call_counters", {})
    _print_call_counters(sorted(call_counters.items()))

    stats = profile.get("stats", {})
    _print_stats(sorted(stats.items()))
    
    
    print("Algorithm completed successfully!")
    

def profile_cached_execution():
    """Demonstrate cache hit profiling."""
    print("=" * 80)
    print("Cache Performance Profiling")
    print("=" * 80)
    
    
    g = gr.Graph()
    nodes = g.add_nodes(100000)
    edge_data = [(nodes[i], nodes[i + 1]) for i in range(99999)]
    g.add_edges(edge_data)
    
    print("First execution (cache miss expected):")
    print("-" * 80)
    _, profile_miss = _run_cc_pipeline(
        g,
        mode="undirected",
        output_attr="comp1",
    )
    _print_phase_table(
        "First Run (Cache Miss) - FFI Timings",
        sorted(profile_miss.get("ffi_timers", {}).items(), key=lambda item: item[1], reverse=True),
    )
    _print_phase_table(
        "First Run (Cache Miss) - Algorithm Timings",
        sorted(profile_miss.get("timers", {}).items(), key=lambda item: item[1], reverse=True),
    )
    
    
    print("Second execution (cache hit expected):")
    print("-" * 80)
    _, profile_hit = _run_cc_pipeline(
        g,
        mode="undirected",
        output_attr="comp2",
    )
    _print_phase_table(
        "Second Run (Cache Hit) - FFI Timings",
        sorted(profile_hit.get("ffi_timers", {}).items(), key=lambda item: item[1], reverse=True),
    )
    _print_phase_table(
        "Second Run (Cache Hit) - Algorithm Timings",
        sorted(profile_hit.get("timers", {}).items(), key=lambda item: item[1], reverse=True),
    )
    
    
    print("Cache demonstration completed!")
    

if __name__ == '__main__':
    try:
        # Run profiling demonstrations
        profile_large_graph()
        print("\n")
        profile_strong_components()
        print("\n")
        profile_cached_execution()
        
        print("=" * 80)
        print("Profiling demonstration completed successfully!")
        
        print("Key insights from profiling:")
        print("- Phase timing shows where time is spent")
        print("- Call counts reveal algorithm behavior")
        print("- Node/edge statistics validate correctness")
        print("- Cache metrics show optimization effectiveness")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
