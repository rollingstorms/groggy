#!/usr/bin/env python3
"""
🚀 FOCUSED GROGGY VS NETWORKX BENCHMARK
======================================

Focus on operations where Groggy's Rust backend should excel:
- Filtering operations (where we've optimized)
- Numerical comparisons
- Multi-criteria queries
- Graph traversals
"""

import time
import random
import sys
from collections import defaultdict

# Add groggy to path
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')

import groggy as gr

# Try to import NetworkX
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("❌ NetworkX not available")


def time_operation(func, *args, **kwargs):
    """Time an operation"""
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        return elapsed, result, True, ""
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        return elapsed, None, False, str(e)


def create_test_graphs(n_nodes=10000, n_edges=5000):
    """Create test graphs for benchmarking"""
    print(f"Creating test graphs ({n_nodes:,} nodes, {n_edges:,} edges)...")
    
    # Create nodes with rich attributes
    nodes_data = []
    for i in range(n_nodes):
        nodes_data.append({
            'id': f'n{i}',
            'salary': random.randint(40000, 200000),
            'age': random.randint(22, 65),
            'experience': random.randint(0, 40),
            'rating': round(random.uniform(1.0, 5.0), 2),
            'role': random.choice(['engineer', 'manager', 'analyst', 'director']),
            'department': random.choice(['Engineering', 'Sales', 'Marketing', 'Finance']),
            'location': random.choice(['San Francisco', 'New York', 'Austin', 'Seattle']),
            'active': random.choice([True, False]),
        })
    
    # Create edges
    edges_data = []
    for _ in range(n_edges):
        source = random.randint(0, n_nodes - 1)
        target = random.randint(0, n_nodes - 1)
        if source != target:
            edges_data.append({
                'source': f'n{source}',
                'target': f'n{target}',
                'weight': round(random.uniform(0.1, 1.0), 3),
                'relationship': random.choice(['collaborates', 'reports_to', 'mentors']),
                'strength': round(random.uniform(0.1, 1.0), 2),
            })
    
    # Create Groggy graph
    start = time.time()
    groggy_graph = gr.Graph(backend='rust')
    groggy_graph.add_nodes(nodes_data)
    for edge in edges_data:
        source = edge['source']
        target = edge['target']
        attrs = {k: v for k, v in edge.items() if k not in ['source', 'target']}
        groggy_graph.add_edge(source, target, **attrs)
    groggy_time = time.time() - start
    
    # Create NetworkX graph
    nx_graph = None
    nx_time = 0
    if NETWORKX_AVAILABLE:
        start = time.time()
        nx_graph = nx.DiGraph()
        for node in nodes_data:
            node_id = node['id']
            attrs = {k: v for k, v in node.items() if k != 'id'}
            nx_graph.add_node(node_id, **attrs)
        for edge in edges_data:
            attrs = {k: v for k, v in edge.items() if k not in ['source', 'target']}
            nx_graph.add_edge(edge['source'], edge['target'], **attrs)
        nx_time = time.time() - start
    
    print(f"  Groggy creation: {groggy_time:.2f}s")
    if NETWORKX_AVAILABLE:
        print(f"  NetworkX creation: {nx_time:.2f}s")
    
    return groggy_graph, nx_graph


def benchmark_filtering_operations(groggy_graph, nx_graph):
    """Benchmark filtering operations - Groggy's strength"""
    print("\n🔍 FILTERING BENCHMARKS")
    print("=" * 50)
    
    results = []
    
    # Test cases that exercise Groggy's optimized filtering
    test_cases = [
        ("Exact role filter", {'role': 'engineer'}),
        ("Salary threshold", {'salary': ('>', 80000)}),
        ("Age filter", {'age': ('>', 30)}),
        ("Multi-numeric filter", {'salary': ('>', 80000), 'age': ('>', 30)}),
        ("Complex filter", {'role': 'engineer', 'salary': ('>', 100000), 'experience': ('>=', 5)}),
        ("Department filter", {'department': 'Engineering'}),
        ("Boolean filter", {'active': True}),
        ("High experience", {'experience': ('>', 15)}),
        ("Rating threshold", {'rating': ('>', 4.0)}),
        ("Multi-criteria", {'role': 'engineer', 'department': 'Engineering', 'salary': ('>', 120000)}),
    ]
    
    for test_name, filter_dict in test_cases:
        print(f"  🔄 {test_name}")
        
        # Test Groggy
        elapsed, result, success, error = time_operation(lambda: groggy_graph.filter_nodes(filter_dict))
        groggy_time = elapsed
        groggy_count = len(result) if success else 0
        
        # Test NetworkX
        nx_time = 0
        nx_count = 0
        if NETWORKX_AVAILABLE and nx_graph:
            def nx_filter():
                def matches_filter(node_id, attrs):
                    for attr_name, condition in filter_dict.items():
                        if attr_name not in attrs:
                            return False
                        
                        value = attrs[attr_name]
                        if isinstance(condition, tuple):
                            op, target = condition
                            if op == '>':
                                if not (value > target):
                                    return False
                            elif op == '>=':
                                if not (value >= target):
                                    return False
                            elif op == '<':
                                if not (value < target):
                                    return False
                            elif op == '<=':
                                if not (value <= target):
                                    return False
                        else:
                            if value != condition:
                                return False
                    return True
                
                return [node for node, attrs in nx_graph.nodes(data=True) if matches_filter(node, attrs)]
            
            elapsed, result, success, error = time_operation(nx_filter)
            nx_time = elapsed
            nx_count = len(result) if success else 0
        
        # Compare results
        if nx_time > 0:
            speedup = nx_time / groggy_time
            comparison = f"{speedup:.1f}x" if speedup >= 1.1 else "similar"
        else:
            comparison = "N/A"
        
        print(f"    Groggy:   {groggy_time:6.2f}ms ({groggy_count:,} results)")
        if NETWORKX_AVAILABLE:
            print(f"    NetworkX: {nx_time:6.2f}ms ({nx_count:,} results) - Groggy {comparison}")
        
        results.append({
            'test': test_name,
            'groggy_time': groggy_time,
            'nx_time': nx_time,
            'groggy_count': groggy_count,
            'nx_count': nx_count
        })
    
    return results


def benchmark_edge_operations(groggy_graph, nx_graph):
    """Benchmark edge operations"""
    print("\n🔗 EDGE OPERATION BENCHMARKS")
    print("=" * 50)
    
    results = []
    
    edge_tests = [
        ("Relationship filter", {'relationship': 'collaborates'}),
        ("Weight threshold", {'weight': ('>', 0.5)}),
        ("Strength filter", {'strength': ('>', 0.7)}),
        ("Complex edge filter", {'relationship': 'collaborates', 'weight': ('>', 0.6)}),
    ]
    
    for test_name, filter_dict in edge_tests:
        print(f"  🔄 {test_name}")
        
        # Test Groggy
        elapsed, result, success, error = time_operation(lambda: groggy_graph.filter_edges(filter_dict))
        groggy_time = elapsed
        groggy_count = len(result) if success else 0
        
        # Test NetworkX
        nx_time = 0
        nx_count = 0
        if NETWORKX_AVAILABLE and nx_graph:
            def nx_edge_filter():
                def matches_filter(attrs):
                    for attr_name, condition in filter_dict.items():
                        if attr_name not in attrs:
                            return False
                        
                        value = attrs[attr_name]
                        if isinstance(condition, tuple):
                            op, target = condition
                            if op == '>':
                                if not (value > target):
                                    return False
                        else:
                            if value != condition:
                                return False
                    return True
                
                return [(u, v) for u, v, attrs in nx_graph.edges(data=True) if matches_filter(attrs)]
            
            elapsed, result, success, error = time_operation(nx_edge_filter)
            nx_time = elapsed
            nx_count = len(result) if success else 0
        
        # Compare results
        if nx_time > 0:
            speedup = nx_time / groggy_time
            comparison = f"{speedup:.1f}x" if speedup >= 1.1 else "similar"
        else:
            comparison = "N/A"
        
        print(f"    Groggy:   {groggy_time:6.2f}ms ({groggy_count:,} results)")
        if NETWORKX_AVAILABLE:
            print(f"    NetworkX: {nx_time:6.2f}ms ({nx_count:,} results) - Groggy {comparison}")
        
        results.append({
            'test': test_name,
            'groggy_time': groggy_time,
            'nx_time': nx_time,
            'groggy_count': groggy_count,
            'nx_count': nx_count
        })
    
    return results


def benchmark_basic_operations(groggy_graph, nx_graph):
    """Benchmark basic graph operations"""
    print("\n🔧 BASIC OPERATION BENCHMARKS")
    print("=" * 50)
    
    results = []
    
    # Test basic operations
    operations = [
        ("Node count", lambda g: g.node_count(), lambda g: g.number_of_nodes()),
        ("Edge count", lambda g: g.edge_count(), lambda g: g.number_of_edges()),
        ("Has node", lambda g: g.has_node('n100'), lambda g: g.has_node('n100')),
        ("Has edge", lambda g: g.has_edge('n100', 'n200'), lambda g: g.has_edge('n100', 'n200')),
        ("Get neighbors", lambda g: g.get_neighbors('n100'), lambda g: list(g.neighbors('n100'))),
    ]
    
    for op_name, groggy_func, nx_func in operations:
        print(f"  🔄 {op_name}")
        
        # Test Groggy
        elapsed, result, success, error = time_operation(groggy_func, groggy_graph)
        groggy_time = elapsed
        groggy_result = result
        
        # Test NetworkX  
        nx_time = 0
        nx_result = None
        if NETWORKX_AVAILABLE and nx_graph:
            elapsed, result, success, error = time_operation(nx_func, nx_graph)
            nx_time = elapsed
            nx_result = result
        
        # Compare results
        if nx_time > 0:
            speedup = nx_time / groggy_time if groggy_time > 0 else float('inf')
            comparison = f"{speedup:.1f}x" if speedup >= 1.1 else "similar"
        else:
            comparison = "N/A"
        
        print(f"    Groggy:   {groggy_time:6.2f}ms")
        if NETWORKX_AVAILABLE:
            print(f"    NetworkX: {nx_time:6.2f}ms - Groggy {comparison}")
        
        results.append({
            'test': op_name,
            'groggy_time': groggy_time,
            'nx_time': nx_time
        })
    
    return results


def print_summary(filtering_results, edge_results, basic_results):
    """Print performance summary"""
    print("\n" + "=" * 80)
    print("🏆 PERFORMANCE SUMMARY")
    print("=" * 80)
    
    all_results = filtering_results + edge_results + basic_results
    
    # Count wins
    wins = {"Groggy": 0, "NetworkX": 0, "Similar": 0}
    total_speedup = 0
    comparisons = 0
    
    for result in all_results:
        if result['groggy_time'] > 0 and result['nx_time'] > 0:
            speedup = result['nx_time'] / result['groggy_time']
            total_speedup += speedup
            comparisons += 1
            
            if speedup >= 1.3:
                wins["Groggy"] += 1
            elif speedup <= 0.77:  # 1/1.3
                wins["NetworkX"] += 1
            else:
                wins["Similar"] += 1
    
    avg_speedup = total_speedup / comparisons if comparisons > 0 else 1
    
    print(f"🎯 Overall Performance:")
    print(f"   Groggy faster: {wins['Groggy']} tests")
    print(f"   NetworkX faster: {wins['NetworkX']} tests") 
    print(f"   Similar performance: {wins['Similar']} tests")
    print(f"   Average Groggy speedup: {avg_speedup:.1f}x")
    
    # Highlight best performing categories
    print(f"\n🚀 Best Groggy Performance:")
    top_speedups = []
    for result in all_results:
        if result['groggy_time'] > 0 and result['nx_time'] > 0:
            speedup = result['nx_time'] / result['groggy_time']
            if speedup >= 2.0:
                top_speedups.append((result['test'], speedup))
    
    top_speedups.sort(key=lambda x: x[1], reverse=True)
    for test, speedup in top_speedups[:5]:
        print(f"   {test}: {speedup:.1f}x faster")
    
    if not top_speedups:
        print("   No major speedups found (>2x)")


def main():
    """Run focused benchmark"""
    print("🚀 FOCUSED GROGGY VS NETWORKX BENCHMARK")
    print("=" * 50)
    print(f"NetworkX available: {'✅' if NETWORKX_AVAILABLE else '❌'}")
    
    # Create test graphs
    groggy_graph, nx_graph = create_test_graphs(n_nodes=10000, n_edges=5000)
    
    # Run benchmarks
    filtering_results = benchmark_filtering_operations(groggy_graph, nx_graph)
    edge_results = benchmark_edge_operations(groggy_graph, nx_graph)
    basic_results = benchmark_basic_operations(groggy_graph, nx_graph)
    
    # Print summary
    print_summary(filtering_results, edge_results, basic_results)


if __name__ == "__main__":
    main()
