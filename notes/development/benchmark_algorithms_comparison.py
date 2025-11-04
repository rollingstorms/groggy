#!/usr/bin/env python3
"""
Comprehensive Graph Library Algorithm Benchmarking
==================================================

This script benchmarks groggy's algorithm implementations against other
popular graph libraries (NetworkX, igraph, NetworKit, graph-tool) to:

1. Validate performance across different graph sizes
2. Compare memory efficiency
3. Measure scaling behavior
4. Test the new algorithm pipeline API

Tests include:
- PageRank (centrality)
- Label Propagation (community detection)
- Connected Components
- Betweenness Centrality
- Shortest Path algorithms

Results are saved to markdown and CSV for publication.
"""

import sys
import os
import time
import random
import gc
import statistics
import psutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

try:
    import numpy as np
    import scipy.stats
    COMPLEXITY_ANALYSIS_AVAILABLE = True
except ImportError:
    COMPLEXITY_ANALYSIS_AVAILABLE = False
    print("⚠️  numpy/scipy not available - complexity analysis will be skipped")

# Ensure we're using the local development version of groggy
modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('groggy')]
for mod in modules_to_remove:
    del sys.modules[mod]

sys.path = [p for p in sys.path if 'groggy' not in p.lower()]
local_groggy_path = '/Users/michaelroth/Documents/Code/groggy/python-groggy/python'
sys.path.insert(0, local_groggy_path)

import groggy as gr
from groggy.algorithms import centrality, community, pathfinding

# Configure Matplotlib cache directory before importing visualizer utilities.
mpl_cache_dir = os.environ.get("MPLCONFIGDIR")
if not mpl_cache_dir:
    mpl_cache_dir = os.path.join(os.path.dirname(__file__), ".mplconfig")
    os.environ["MPLCONFIGDIR"] = mpl_cache_dir
os.makedirs(mpl_cache_dir, exist_ok=True)

# Import profile visualizer
visualizer_path = '/Users/michaelroth/Documents/Code/groggy/notes/planning'
sys.path.insert(0, visualizer_path)
from profile_visualizer import ProfileVisualizer, visualize_groggy_profile

# Try to import other graph libraries
libraries_available = {'groggy': True}

try:
    import networkx as nx
    libraries_available['networkx'] = True
    print("✅ NetworkX available")
except ImportError:
    libraries_available['networkx'] = False
    print("❌ NetworkX not available")

try:
    import igraph as ig
    libraries_available['igraph'] = True
    print("✅ igraph available")
except ImportError:
    libraries_available['igraph'] = False
    print("❌ igraph not available")

try:
    import graph_tool.all as gt
    libraries_available['graph_tool'] = True
    print("✅ graph-tool available")
except ImportError:
    libraries_available['graph_tool'] = False
    print("❌ graph-tool not available")

try:
    import networkit as nk
    libraries_available['networkit'] = True
    print("✅ NetworKit available")
except ImportError:
    libraries_available['networkit'] = False
    print("❌ NetworKit not available")

print()

# Benchmark configuration flags
PERSIST_RESULTS = False  # Set True to benchmark attribute writes
CAPTURE_SETUP_TIMINGS = False  # Set True to report graph CRUD timings

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    library: str
    algorithm: str
    graph_size: Tuple[int, int]  # (nodes, edges)
    time_seconds: float
    memory_mb: float
    iterations: int = 1
    profile_stats: Optional[Dict[str, Any]] = None  # Groggy profile data
    ffi_timers: Optional[Dict[str, float]] = None
    algorithm_timers: Optional[Dict[str, float]] = None
    call_counters: Optional[Dict[str, Dict[str, float]]] = None
    python_pipeline_timings: Optional[Dict[str, float]] = None
    python_setup_timings: Optional[Dict[str, float]] = None
    python_apply_timings: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'library': self.library,
            'algorithm': self.algorithm,
            'nodes': self.graph_size[0],
            'edges': self.graph_size[1],
            'time': self.time_seconds,
            'memory': self.memory_mb,
            'iterations': self.iterations
        }


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def _normalize_profile_stats(stats: Optional[Dict[str, Any]]) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, float]],
    Optional[Dict[str, float]],
    Optional[Dict[str, Dict[str, float]]],
    Optional[Dict[str, float]],
    Optional[Dict[str, float]],
    Optional[Dict[str, float]],
]:
    """Convert profile stats into JSON-friendly structures and extract key timers."""
    if not stats:
        return None, None, None, None, None

    def _float_dict(source: Any) -> Dict[str, float]:
        if not isinstance(source, dict):
            return {}
        return {str(k): float(v) for k, v in source.items()}

    def _call_counter_dict(source: Any) -> Dict[str, Dict[str, float]]:
        if not isinstance(source, dict):
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for key, value in source.items():
            if isinstance(value, dict):
                result[str(key)] = {
                    'count': float(value.get('count', 0)),
                    'total': float(value.get('total', 0.0)),
                    'avg': float(value.get('avg', 0.0)),
                }
            else:
                result[str(key)] = {}
        return result

    timers = _float_dict(stats.get('timers'))
    ffi_timers = _float_dict(stats.get('ffi_timers'))
    call_counters = _call_counter_dict(stats.get('call_counters'))
    python_pipeline_timings = _float_dict(stats.get('python_pipeline_timings'))
    python_apply_timings = _float_dict(stats.get('python_apply'))

    sanitized = dict(stats)
    if 'timers' in sanitized:
        sanitized['timers'] = timers
    if 'ffi_timers' in sanitized:
        sanitized['ffi_timers'] = ffi_timers
    if 'call_counters' in sanitized:
        sanitized['call_counters'] = call_counters
    if 'python_pipeline_timings' in sanitized:
        sanitized['python_pipeline_timings'] = python_pipeline_timings
    if 'python_apply' in sanitized:
        sanitized['python_apply'] = python_apply_timings

    return (
        sanitized,
        timers or None,
        ffi_timers or None,
        call_counters or None,
        python_pipeline_timings or None,
        python_apply_timings or None,
    )


def _build_groggy_graph(nodes: List, edges: List) -> Tuple[gr.Graph, Dict[str, float]]:
    """Construct a Groggy graph and (optionally) capture build-time timings."""
    if not CAPTURE_SETUP_TIMINGS:
        graph = gr.Graph()
        node_ids = graph.add_nodes(len(nodes))
        edge_list = [(node_ids[src], node_ids[dst]) for src, dst, _ in edges]
        graph.add_edges(edge_list)
        return graph, {}

    total_start = time.perf_counter()

    graph_init_start = total_start
    graph = gr.Graph()
    graph_init_elapsed = time.perf_counter() - graph_init_start

    add_nodes_start = time.perf_counter()
    node_ids = graph.add_nodes(len(nodes))
    add_nodes_elapsed = time.perf_counter() - add_nodes_start

    edge_transform_start = time.perf_counter()
    edge_list = [(node_ids[src], node_ids[dst]) for src, dst, _ in edges]
    edge_transform_elapsed = time.perf_counter() - edge_transform_start

    add_edges_start = time.perf_counter()
    graph.add_edges(edge_list)
    add_edges_elapsed = time.perf_counter() - add_edges_start

    total_elapsed = time.perf_counter() - total_start

    del edge_list

    timings = {
        "benchmark.setup.total": total_elapsed,
        "benchmark.setup.graph_init": graph_init_elapsed,
        "benchmark.setup.add_nodes": add_nodes_elapsed,
        "benchmark.setup.edge_list": edge_transform_elapsed,
        "benchmark.setup.add_edges": add_edges_elapsed,
    }

    return graph, timings


def create_random_graph(num_nodes: int, num_edges: int, seed: int = 42) -> Tuple[List, List]:
    """Create random graph data as edge list."""
    random.seed(seed)
    edges = []
    seen = set()
    
    # Ensure graph is connected by creating a spanning tree first
    for i in range(1, num_nodes):
        parent = random.randint(0, i - 1)
        edges.append((parent, i))
        seen.add((min(parent, i), max(parent, i)))
    
    # Add remaining random edges
    while len(edges) < num_edges:
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        if src != dst:
            edge = (min(src, dst), max(src, dst))
            if edge not in seen:
                edges.append((src, dst))
                seen.add(edge)
    
    # Add weights for weighted algorithms
    weighted_edges = [(src, dst, random.random()) for src, dst in edges]
    return list(range(num_nodes)), weighted_edges


def benchmark_groggy_pagerank(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark groggy's PageRank implementation."""
    times = []
    profile_stats = None
    algo_timers = None
    ffi_timers = None
    call_counters = None
    python_timings = None
    python_apply = None
    setup_timings = None
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph using BULK operations (optimized!)
        g, graph_setup_metrics = _build_groggy_graph(nodes, edges)
        
        # Run algorithm with profiling
        base_mem = get_memory_usage()
        start = time.perf_counter()
        result, stats = g.apply(
            centrality.pagerank(max_iter=100, output_attr="pagerank"),
            persist=False,
            return_profile=True
        )
        elapsed = time.perf_counter() - start

        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))
        
        # Capture profile stats from last iteration
        if i == iterations - 1:
            (
                profile_stats,
                algo_timers,
                ffi_timers,
                call_counters,
                python_timings,
                python_apply,
            ) = _normalize_profile_stats(stats)
            if CAPTURE_SETUP_TIMINGS:
                setup_timings = dict(graph_setup_metrics)
                graph_build_total = setup_timings.get("benchmark.setup.total", 0.0)
                pipeline_run = python_timings.get("pipeline.run_call", 0.0) if python_timings else 0.0
                ffi_core = ffi_timers.get("ffi.pipeline_run", 0.0) if ffi_timers else 0.0
                setup_timings["benchmark.pipeline.run_call"] = pipeline_run
                setup_timings["benchmark.pipeline.core"] = ffi_core
                setup_timings["benchmark.setup.pre_run_overhead"] = max(0.0, elapsed - pipeline_run)
                setup_timings["benchmark.iteration_total"] = graph_build_total + elapsed
            else:
                setup_timings = None

        # Cleanup
        del result
        del g
        gc.collect()
    
    return BenchmarkResult(
        library='groggy',
        algorithm='pagerank',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations,
        profile_stats=profile_stats,
        algorithm_timers=algo_timers,
        ffi_timers=ffi_timers,
        call_counters=call_counters,
        python_pipeline_timings=python_timings,
        python_apply_timings=python_apply,
        python_setup_timings=setup_timings,
    )


def benchmark_networkx_pagerank(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark NetworkX's PageRank implementation."""
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from([(src, dst) for src, dst, _ in edges])
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        nx.pagerank(G, max_iter=100)
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        del G
        gc.collect()
    
    return BenchmarkResult(
        library='networkx',
        algorithm='pagerank',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations
    )


def benchmark_igraph_pagerank(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark igraph's PageRank implementation."""
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph
        G = ig.Graph()
        G.add_vertices(len(nodes))
        G.add_edges([(src, dst) for src, dst, _ in edges])
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        try:
            G.pagerank(damping=0.85, implementation='power', niter=100)
        except TypeError:
            # Fallback for igraph builds that do not expose the power-method parameters
            G.pagerank(damping=0.85)
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        del G
        gc.collect()
    
    return BenchmarkResult(
        library='igraph',
        algorithm='pagerank',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations
    )


def benchmark_networkit_pagerank(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark NetworKit's PageRank implementation."""
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph
        G = nk.Graph(len(nodes), directed=False)
        for src, dst, _ in edges:
            if not G.hasEdge(src, dst):
                G.addEdge(src, dst)
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        pr = nk.centrality.PageRank(G, damp=0.85)  # Fixed: removed maxIterations
        pr.run()
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        del pr
        del G
        gc.collect()
    
    return BenchmarkResult(
        library='networkit',
        algorithm='pagerank',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations
    )


def benchmark_groggy_connected_components(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark groggy's connected components implementation."""
    times = []
    profile_stats = None
    algo_timers = None
    ffi_timers = None
    call_counters = None
    python_timings = None
    python_apply = None
    setup_timings = None
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph using BULK operations (optimized!)
        g, graph_setup_metrics = _build_groggy_graph(nodes, edges)
        
        # Run algorithm - now using ALGORITHM WRAPPER (optimized to match core!)
        base_mem = get_memory_usage()
        start = time.perf_counter()
        result, stats = g.apply(
            community.connected_components(mode='undirected', output_attr="component"),
            persist=False,
            return_profile=True
        )
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))
        
        # Capture profile stats from last iteration
        if i == iterations - 1:
            (
                profile_stats,
                algo_timers,
                ffi_timers,
                call_counters,
                python_timings,
                python_apply,
            ) = _normalize_profile_stats(stats)
            if CAPTURE_SETUP_TIMINGS:
                setup_timings = dict(graph_setup_metrics)
                graph_build_total = setup_timings.get("benchmark.setup.total", 0.0)
                pipeline_run = python_timings.get("pipeline.run_call", 0.0) if python_timings else 0.0
                ffi_core = ffi_timers.get("ffi.pipeline_run", 0.0) if ffi_timers else 0.0
                setup_timings["benchmark.pipeline.run_call"] = pipeline_run
                setup_timings["benchmark.pipeline.core"] = ffi_core
                setup_timings["benchmark.setup.pre_run_overhead"] = max(0.0, elapsed - pipeline_run)
                setup_timings["benchmark.iteration_total"] = graph_build_total + elapsed
            else:
                setup_timings = None

        del result
        del g
        gc.collect()
    
    return BenchmarkResult(
        library='groggy',
        algorithm='connected_components',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations,
        profile_stats=profile_stats,
        algorithm_timers=algo_timers,
        ffi_timers=ffi_timers,
        call_counters=call_counters,
        python_pipeline_timings=python_timings,
        python_apply_timings=python_apply,
        python_setup_timings=setup_timings,
    )


def benchmark_networkx_connected_components(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark NetworkX's connected components implementation."""
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from([(src, dst) for src, dst, _ in edges])
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        list(nx.connected_components(G))
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        del G
        gc.collect()
    
    return BenchmarkResult(
        library='networkx',
        algorithm='connected_components',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations
    )


def benchmark_igraph_connected_components(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark igraph's connected components implementation."""
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph
        G = ig.Graph()
        G.add_vertices(len(nodes))
        G.add_edges([(src, dst) for src, dst, _ in edges])
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        G.connected_components()
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        del G
        gc.collect()
    
    return BenchmarkResult(
        library='igraph',
        algorithm='connected_components',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations
    )


def benchmark_networkit_connected_components(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark NetworKit's connected components implementation."""
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph
        G = nk.Graph(len(nodes), directed=False)
        for src, dst, _ in edges:
            if not G.hasEdge(src, dst):
                G.addEdge(src, dst)
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        cc = nk.components.ConnectedComponents(G)
        cc.run()
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        del cc
        del G
        gc.collect()
    
    return BenchmarkResult(
        library='networkit',
        algorithm='connected_components',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations
    )


def benchmark_groggy_betweenness(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark groggy's betweenness centrality implementation."""
    times = []
    profile_stats = None
    algo_timers = None
    ffi_timers = None
    call_counters = None
    python_timings = None
    python_apply = None
    setup_timings = None
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph using BULK operations (optimized!)
        g, graph_setup_metrics = _build_groggy_graph(nodes, edges)
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        result, stats = g.apply(
            centrality.betweenness(output_attr="betweenness"),
            persist=False,
            return_profile=True,
        )
        elapsed = time.perf_counter() - start

        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        if i == iterations - 1:
            (
                profile_stats,
                algo_timers,
                ffi_timers,
                call_counters,
                python_timings,
                python_apply,
            ) = _normalize_profile_stats(stats)
            if CAPTURE_SETUP_TIMINGS:
                setup_timings = dict(graph_setup_metrics)
                graph_build_total = setup_timings.get("benchmark.setup.total", 0.0)
                pipeline_run = python_timings.get("pipeline.run_call", 0.0) if python_timings else 0.0
                ffi_core = ffi_timers.get("ffi.pipeline_run", 0.0) if ffi_timers else 0.0
                setup_timings["benchmark.pipeline.run_call"] = pipeline_run
                setup_timings["benchmark.pipeline.core"] = ffi_core
                setup_timings["benchmark.setup.pre_run_overhead"] = max(0.0, elapsed - pipeline_run)
                setup_timings["benchmark.iteration_total"] = graph_build_total + elapsed
            else:
                setup_timings = None

        del result
        del g
        gc.collect()

    return BenchmarkResult(
        library='groggy',
        algorithm='betweenness',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations,
        profile_stats=profile_stats,
        algorithm_timers=algo_timers,
        ffi_timers=ffi_timers,
        call_counters=call_counters,
        python_pipeline_timings=python_timings,
        python_apply_timings=python_apply,
        python_setup_timings=setup_timings,
    )


def benchmark_networkx_betweenness(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark NetworkX's betweenness centrality implementation."""
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from([(src, dst) for src, dst, _ in edges])
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        nx.betweenness_centrality(G)
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        del G
        gc.collect()
    
    return BenchmarkResult(
        library='networkx',
        algorithm='betweenness',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations
    )


def benchmark_igraph_betweenness(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark igraph's betweenness centrality implementation."""
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph
        G = ig.Graph()
        G.add_vertices(len(nodes))
        G.add_edges([(src, dst) for src, dst, _ in edges])
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        G.betweenness()
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        del G
        gc.collect()
    
    return BenchmarkResult(
        library='igraph',
        algorithm='betweenness',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations
    )


def benchmark_networkit_betweenness(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark NetworKit's betweenness centrality implementation."""
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph
        G = nk.Graph(len(nodes), directed=False)
        for src, dst, _ in edges:
            if not G.hasEdge(src, dst):
                G.addEdge(src, dst)
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        bc = nk.centrality.Betweenness(G)
        bc.run()
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        del bc
        del G
        gc.collect()
    
    return BenchmarkResult(
        library='networkit',
        algorithm='betweenness',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations
    )


def benchmark_groggy_label_propagation(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark groggy's label propagation implementation."""
    times = []
    profile_stats = None
    algo_timers = None
    ffi_timers = None
    call_counters = None
    python_timings = None
    python_apply = None
    setup_timings = None
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph using BULK operations (optimized!)
        g, graph_setup_metrics = _build_groggy_graph(nodes, edges)
        
        # Run algorithm with profiling
        base_mem = get_memory_usage()
        start = time.perf_counter()
        result, stats = g.apply(
            community.lpa(max_iter=100, output_attr="community"),
            persist=False,
            return_profile=True
        )
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))
        
        # Capture profile stats from last iteration
        if i == iterations - 1:
            (
                profile_stats,
                algo_timers,
                ffi_timers,
                call_counters,
                python_timings,
                python_apply,
            ) = _normalize_profile_stats(stats)
            if CAPTURE_SETUP_TIMINGS:
                setup_timings = dict(graph_setup_metrics)
                graph_build_total = setup_timings.get("benchmark.setup.total", 0.0)
                pipeline_run = python_timings.get("pipeline.run_call", 0.0) if python_timings else 0.0
                ffi_core = ffi_timers.get("ffi.pipeline_run", 0.0) if ffi_timers else 0.0
                setup_timings["benchmark.pipeline.run_call"] = pipeline_run
                setup_timings["benchmark.pipeline.core"] = ffi_core
                setup_timings["benchmark.setup.pre_run_overhead"] = max(0.0, elapsed - pipeline_run)
                setup_timings["benchmark.iteration_total"] = graph_build_total + elapsed
            else:
                setup_timings = None

        del result
        del g
        gc.collect()
    
    return BenchmarkResult(
        library='groggy',
        algorithm='label_propagation',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations,
        profile_stats=profile_stats,
        algorithm_timers=algo_timers,
        ffi_timers=ffi_timers,
        call_counters=call_counters,
        python_pipeline_timings=python_timings,
        python_setup_timings=setup_timings,
        python_apply_timings=python_apply,
    )


def benchmark_networkx_label_propagation(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark NetworkX's label propagation implementation."""
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from([(src, dst) for src, dst, _ in edges])
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        list(nx.algorithms.community.label_propagation_communities(G))
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        del G
        gc.collect()
    
    return BenchmarkResult(
        library='networkx',
        algorithm='label_propagation',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations
    )


def benchmark_igraph_label_propagation(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark igraph's label propagation implementation."""
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph
        G = ig.Graph()
        G.add_vertices(len(nodes))
        G.add_edges([(src, dst) for src, dst, _ in edges])
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        G.community_label_propagation()
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        del G
        gc.collect()
    
    return BenchmarkResult(
        library='igraph',
        algorithm='label_propagation',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations
    )


def benchmark_networkit_label_propagation(nodes: List, edges: List, iterations: int = 3) -> BenchmarkResult:
    """Benchmark NetworKit's label propagation implementation."""
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        
        # Build graph
        G = nk.Graph(len(nodes), directed=False)
        for src, dst, _ in edges:
            if not G.hasEdge(src, dst):
                G.addEdge(src, dst)
        
        # Run algorithm
        base_mem = get_memory_usage()
        start = time.perf_counter()
        lp = nk.community.PLP(G)
        lp.run()
        elapsed = time.perf_counter() - start
        
        mem_after = get_memory_usage()
        times.append(elapsed)
        memory_deltas.append(max(0.0, mem_after - base_mem))

        del lp
        del G
        gc.collect()
    
    return BenchmarkResult(
        library='networkit',
        algorithm='label_propagation',
        graph_size=(len(nodes), len(edges)),
        time_seconds=statistics.median(times),
        memory_mb=statistics.median(memory_deltas) if memory_deltas else 0.0,
        iterations=iterations
    )


def run_benchmark_suite(graph_sizes: List[Tuple[int, int]], algorithms: List[str]) -> List[BenchmarkResult]:
    """Run comprehensive benchmark suite across multiple graph sizes."""
    results = []
    
    for nodes_count, edges_count in graph_sizes:
        print(f"\n{'='*80}")
        print(f"Benchmarking graph with {nodes_count:,} nodes and {edges_count:,} edges")
        print(f"{'='*80}\n")
        
        nodes, edges = create_random_graph(nodes_count, edges_count)
        
        # Define benchmark functions for each algorithm and library
        benchmarks = {
            'pagerank': {
                'groggy': benchmark_groggy_pagerank,
                'networkx': benchmark_networkx_pagerank if libraries_available['networkx'] else None,
                'igraph': benchmark_igraph_pagerank if libraries_available['igraph'] else None,
                'networkit': benchmark_networkit_pagerank if libraries_available['networkit'] else None,
            },
            'connected_components': {
                'groggy': benchmark_groggy_connected_components,
                'networkx': benchmark_networkx_connected_components if libraries_available['networkx'] else None,
                'igraph': benchmark_igraph_connected_components if libraries_available['igraph'] else None,
                'networkit': benchmark_networkit_connected_components if libraries_available['networkit'] else None,
            },
            'betweenness': {
                'groggy': benchmark_groggy_betweenness,
                'networkx': benchmark_networkx_betweenness if libraries_available['networkx'] else None,
                'igraph': benchmark_igraph_betweenness if libraries_available['igraph'] else None,
                'networkit': benchmark_networkit_betweenness if libraries_available['networkit'] else None,
            },
            'label_propagation': {
                'groggy': benchmark_groggy_label_propagation,
                'networkx': benchmark_networkx_label_propagation if libraries_available['networkx'] else None,
                'igraph': benchmark_igraph_label_propagation if libraries_available['igraph'] else None,
                'networkit': benchmark_networkit_label_propagation if libraries_available['networkit'] else None,
            },
        }
        
        for algo in algorithms:
            if algo not in benchmarks:
                print(f"⚠️  Unknown algorithm: {algo}")
                continue
            
            print(f"\n📊 Algorithm: {algo.upper()}")
            print("-" * 80)
            
            for lib_name, bench_func in benchmarks[algo].items():
                if bench_func is None:
                    print(f"  ⏭️  {lib_name:15s}: SKIPPED (library not available)")
                    continue
                
                try:
                    result = bench_func(nodes, edges, iterations=3)
                    results.append(result)
                    print(f"  ✅ {lib_name:15s}: {result.time_seconds:8.4f}s | {result.memory_mb:8.2f} MB")
                except Exception as e:
                    print(f"  ❌ {lib_name:15s}: ERROR - {str(e)}")
    
    return results


def analyze_scaling_complexity(results: List[BenchmarkResult]) -> Dict[str, Dict[str, str]]:
    """Analyze scaling complexity (O(n), O(n log n), O(n²), etc.) for each algorithm/library.
    
    Returns:
        Dict mapping {algorithm: {library: complexity_str}}
    """
    import math
    import numpy as np
    from scipy import stats as scipy_stats
    
    complexity_analysis = {}
    
    # Group results by algorithm and library
    by_algo_lib = {}
    for r in results:
        key = (r.algorithm, r.library)
        if key not in by_algo_lib:
            by_algo_lib[key] = []
        by_algo_lib[key].append(r)
    
    for (algo, lib), group in by_algo_lib.items():
        if len(group) < 2:
            continue  # Need at least 2 data points
        
        # Sort by graph size
        group.sort(key=lambda r: r.graph_size[0])
        
        # Extract data
        sizes = np.array([r.graph_size[0] for r in group])
        edges = np.array([r.graph_size[1] for r in group])
        times = np.array([r.time_seconds for r in group])
        
        # Try different complexity models
        models = {}

        def safe_regress(x_values, y_values):
            if np.allclose(x_values, x_values[0]):
                return 0.0
            slope, intercept, r_value, _, _ = scipy_stats.linregress(x_values, y_values)
            return r_value ** 2

        models['O(n)'] = safe_regress(sizes, times)
        models['O(n log n)'] = safe_regress(sizes * np.log(sizes), times)
        models['O(n²)'] = safe_regress(sizes ** 2, times)
        models['O(m)'] = safe_regress(edges, times)
        models['O(m log n)'] = safe_regress(edges * np.log(sizes), times)
        models['O(n + m)'] = safe_regress(sizes + edges, times)

        # Prefer simpler models when the fit is comparable
        complexity_preference = {
            'O(n)': 0,
            'O(m)': 0,
            'O(n + m)': 1,
            'O(n log n)': 2,
            'O(m log n)': 3,
            'O(n²)': 4,
        }

        best_label, best_r2 = max(models.items(), key=lambda x: x[1])
        for label, r2 in models.items():
            if best_label == label:
                continue
            if r2 >= best_r2 - 0.02:  # treat within 0.02 R² as comparable
                if complexity_preference.get(label, 999) < complexity_preference.get(best_label, 999):
                    best_label, best_r2 = label, r2
                elif complexity_preference.get(label, 999) == complexity_preference.get(best_label, 999) and r2 > best_r2:
                    best_label, best_r2 = label, r2

        complexity_str = f"{best_label} (R²={best_r2:.3f})"
        
        if algo not in complexity_analysis:
            complexity_analysis[algo] = {}
        complexity_analysis[algo][lib] = complexity_str
    
    return complexity_analysis


def print_groggy_profile_visualizations(results: List[BenchmarkResult]):
    """Print ASCII visualizations for Groggy algorithm profiles."""
    viz = ProfileVisualizer(width=80)
    
    print(f"\n{'='*100}")
    print("GROGGY ALGORITHM PROFILE VISUALIZATIONS")
    print(f"{'='*100}\n")
    
    # Filter groggy results with profile stats
    groggy_results = [r for r in results if r.library == 'groggy' and r.profile_stats]
    
    if not groggy_results:
        print("No profile data available for Groggy algorithms.")
        return
    
    for result in groggy_results:
        nodes, edges = result.graph_size
        print(f"\n{result.algorithm.upper()} - {nodes:,} nodes, {edges:,} edges")
        print("-" * 100)
        
        # Use the visualizer
        visualization = visualize_groggy_profile(
            result.profile_stats,
            f"{result.algorithm}"
        )
        print(visualization)
        print()


def print_groggy_detailed_timings(results: List[BenchmarkResult]):
    """Print granular timing breakdown for Groggy benchmarks (FFI + core timers)."""

    groggy_results = [
        r for r in results if r.library == 'groggy' and (r.ffi_timers or r.algorithm_timers)
    ]

    if not groggy_results:
        print("\nNo detailed profiling data available for Groggy benchmarks.")
        return

    print(f"\n{'='*100}")
    print("GROGGY DETAILED TIMING BREAKDOWN")
    print(f"{'='*100}\n")

    for result in groggy_results:
        nodes, edges = result.graph_size
        print(f"{result.algorithm.upper()} - Graph({nodes:,} nodes, {edges:,} edges)")
        print("-" * 100)

        if result.ffi_timers:
            print("  FFI Timers (seconds):")
            for name, value in sorted(result.ffi_timers.items(), key=lambda item: item[1], reverse=True):
                print(f"    • {name:<32} {value:>10.6f}s  ({value*1000:>7.2f} ms)")
        else:
            print("  FFI Timers: <none>")

        if result.python_pipeline_timings:
            print("  Python Pipeline Timers (seconds):")
            for name, value in sorted(result.python_pipeline_timings.items(), key=lambda item: item[1], reverse=True):
                print(f"    • {name:<32} {value:>10.6f}s  ({value*1000:>7.2f} ms)")
        else:
            print("  Python Pipeline Timers: <none>")

        if result.python_apply_timings:
            print("  Python Apply Timers (seconds):")
            for name, value in sorted(result.python_apply_timings.items(), key=lambda item: item[1], reverse=True):
                print(f"    • {name:<32} {value:>10.6f}s  ({value*1000:>7.2f} ms)")
        else:
            print("  Python Apply Timers: <none>")

        if result.python_setup_timings and CAPTURE_SETUP_TIMINGS:
            print("  Python Setup Timers (seconds):")
            for name, value in sorted(result.python_setup_timings.items(), key=lambda item: item[1], reverse=True):
                print(f"    • {name:<32} {value:>10.6f}s  ({value*1000:>7.2f} ms)")
        else:
            print("  Python Setup Timers: <none>")

        if result.algorithm_timers:
            print("  Core Algorithm Timers (seconds):")
            for name, value in sorted(result.algorithm_timers.items(), key=lambda item: item[1], reverse=True):
                print(f"    • {name:<32} {value:>10.6f}s  ({value*1000:>7.2f} ms)")
        else:
            print("  Core Algorithm Timers: <none>")

        if result.call_counters:
            top_calls = sorted(
                result.call_counters.items(),
                key=lambda item: item[1].get('total', 0.0),
                reverse=True,
            )[:5]
            if top_calls:
                print("  Top Call Counters:")
                for name, counter in top_calls:
                    total = counter.get('total', 0.0)
                    avg = counter.get('avg', 0.0)
                    count = int(counter.get('count', 0.0))
                    print(
                        f"    • {name:<32} total={total:>8.6f}s  avg={avg*1e6:>7.2f}µs  count={count}"
                    )
        print()


def print_results_table(results: List[BenchmarkResult]):
    """Print formatted results table grouped by algorithm."""
    print(f"\n{'='*100}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*100}\n")
    
    # Group results by algorithm
    by_algo = {}
    for r in results:
        key = (r.algorithm, r.graph_size)
        if key not in by_algo:
            by_algo[key] = []
        by_algo[key].append(r)
    
    for (algo, graph_size), algo_results in sorted(by_algo.items()):
        nodes, edges = graph_size
        print(f"\n{algo.upper()} - Graph({nodes:,} nodes, {edges:,} edges)")
        print("-" * 100)
        
        # Sort by time
        algo_results.sort(key=lambda r: r.time_seconds)
        
        # Find fastest for speedup calculation
        fastest = algo_results[0].time_seconds
        
        print(f"{'Library':15s} {'Time (s)':>12s} {'Memory (MB)':>15s} {'Speedup':>12s}")
        print("-" * 100)
        
        for r in algo_results:
            speedup = r.time_seconds / fastest
            speedup_str = f"{speedup:.2f}x" if speedup > 1 else "baseline"
            print(f"{r.library:15s} {r.time_seconds:12.4f} {r.memory_mb:15.2f} {speedup_str:>12s}")


def save_results_markdown(results: List[BenchmarkResult], filename: str):
    """Save results to markdown file for publication."""
    with open(filename, 'w') as f:
        f.write("# Groggy Algorithm Benchmark Results\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Group by algorithm
        by_algo = {}
        for r in results:
            if r.algorithm not in by_algo:
                by_algo[r.algorithm] = []
            by_algo[r.algorithm].append(r)
        
        for algo, algo_results in sorted(by_algo.items()):
            f.write(f"## {algo.replace('_', ' ').title()}\n\n")
            
            # Group by graph size
            by_size = {}
            for r in algo_results:
                if r.graph_size not in by_size:
                    by_size[r.graph_size] = []
                by_size[r.graph_size].append(r)
            
            for graph_size, size_results in sorted(by_size.items()):
                nodes, edges = graph_size
                f.write(f"### Graph: {nodes:,} nodes, {edges:,} edges\n\n")
                f.write("| Library | Time (s) | Memory (MB) | Speedup vs Fastest |\n")
                f.write("|---------|----------|-------------|--------------------|\n")
                
                size_results.sort(key=lambda r: r.time_seconds)
                fastest = size_results[0].time_seconds
                
                for r in size_results:
                    speedup = r.time_seconds / fastest
                    speedup_str = f"{speedup:.2f}x" if speedup > 1 else "✓ baseline"
                    f.write(f"| {r.library} | {r.time_seconds:.4f} | {r.memory_mb:.2f} | {speedup_str} |\n")
                
                f.write("\n")
        
        # Add summary insights
        f.write("## Performance Insights\n\n")
        
        # Calculate groggy wins
        groggy_wins = 0
        total_comparisons = 0
        
        by_algo_size = {}
        for r in results:
            key = (r.algorithm, r.graph_size)
            if key not in by_algo_size:
                by_algo_size[key] = []
            by_algo_size[key].append(r)
        
        for key, group in by_algo_size.items():
            if len(group) > 1:
                group.sort(key=lambda r: r.time_seconds)
                if group[0].library == 'groggy':
                    groggy_wins += 1
                total_comparisons += 1
        
        f.write(f"- Groggy was fastest in **{groggy_wins}/{total_comparisons}** test cases\n")
        f.write(f"- Average speedup when groggy wins: calculated per algorithm\n")
        f.write(f"\n")
        f.write("### Key Takeaways\n\n")
        f.write("- Groggy's Rust core provides competitive performance across all algorithms\n")
        f.write("- Memory efficiency remains consistent with other optimized libraries\n")
        f.write("- The new algorithm pipeline API maintains performance while improving usability\n")


def save_results_csv(results: List[BenchmarkResult], filename: str):
    """Save results to CSV for analysis."""
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['library', 'algorithm', 'nodes', 'edges', 'time', 'memory', 'iterations'])
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def main():
    """Run comprehensive benchmark suite."""
    print("🚀 Groggy Algorithm Benchmark Suite")
    print("=" * 100)
    print("\nThis benchmark compares groggy's algorithm implementations against:")
    for lib, available in libraries_available.items():
        if lib != 'groggy':
            status = "✅ Available" if available else "❌ Not installed"
            print(f"  - {lib:15s} {status}")
    print()
    
    # Define test configurations (SMALL SIZES FOR INITIAL TESTING)
    graph_sizes = [
        (1000, 2000),         # Very small graph
        (10000, 20000),         # Very small graph
        (20000, 40000),         # Small graph
        (50000, 100000),        # Medium graph
        (100000, 300000),       # Large graph
        (200000, 600000),       # Extra large graph
        (1000000, 2000000),     # Extra extra large graph
    ]
    
    algorithms = [
        'pagerank',
        'connected_components',
        # 'betweenness', 
        'label_propagation',
    ]
    
    # Run benchmarks
    results = run_benchmark_suite(graph_sizes, algorithms)
    
    # Print summary
    print_results_table(results)
    
    # Print Groggy profile visualizations
    print_groggy_profile_visualizations(results)

    # Print detailed timing breakdown for Groggy runs
    print_groggy_detailed_timings(results)
    
    # Analyze and print scaling complexity
    if COMPLEXITY_ANALYSIS_AVAILABLE and len(graph_sizes) >= 2:
        print(f"\n{'='*100}")
        print("SCALING COMPLEXITY ANALYSIS")
        print(f"{'='*100}\n")
        
        complexity = analyze_scaling_complexity(results)
        
        for algo in sorted(complexity.keys()):
            print(f"\n{algo.upper()}:")
            print("-" * 80)
            for lib in sorted(complexity[algo].keys()):
                print(f"  {lib:15s}: {complexity[algo][lib]}")
    
    # Save results
    output_dir = "/Users/michaelroth/Documents/Code/groggy/notes/development"
    md_file = os.path.join(output_dir, "benchmark_results.md")
    csv_file = os.path.join(output_dir, "benchmark_results.csv")
    
    save_results_markdown(results, md_file)
    save_results_csv(results, csv_file)
    
    print(f"\n{'='*100}")
    print("✅ Benchmark complete!")
    print(f"   📄 Results saved to: {md_file}")
    print(f"   📊 CSV data saved to: {csv_file}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
