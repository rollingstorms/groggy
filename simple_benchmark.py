#!/usr/bin/env python3
"""
🚀 SIMPLE GROGGY VS NETWORKX BENCHMARK
======================================

Focused benchmark testing the key operations:
- Graph creation with batch operations
- Basic operations
- Filtering operations
- Memory usage

This is a simplified, debuggable version.
"""

import time
import random
import sys

# Add groggy to path
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python')

import groggy as gr

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
    print("✅ NetworkX available")
except ImportError:
    NETWORKX_AVAILABLE = False
    print("❌ NetworkX not available")


def time_function(func, description):
    """Time a function and return results"""
    print(f"  🔄 {description}...", end=" ")
    start_time = time.time()
    try:
        result = func()
        elapsed = (time.time() - start_time) * 1000
        result_size = len(result) if hasattr(result, '__len__') else (1 if result is not None else 0)
        print(f"✅ {elapsed:.2f}ms ({result_size:,} results)")
        return elapsed, result_size, True, ""
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        print(f"❌ {elapsed:.2f}ms - Error: {str(e)}")
        return elapsed, 0, False, str(e)


def create_test_data(n_nodes, n_edges):
    """Create test data"""
    print(f"Creating test data: {n_nodes:,} nodes, {n_edges:,} edges")
    
    nodes_data = []
    for i in range(n_nodes):
        nodes_data.append({
            'id': f'n{i}',
            'salary': random.randint(50000, 150000),
            'age': random.randint(25, 65),
            'role': random.choice(['engineer', 'manager', 'analyst']),
            'department': random.choice(['Engineering', 'Sales', 'Marketing']),
            'active': random.choice([True, False])
        })
    
    edges_data = []
    for _ in range(n_edges):
        source = random.randint(0, n_nodes - 1)
        target = random.randint(0, n_nodes - 1)
        if source != target:
            edges_data.append({
                'source': f'n{source}',
                'target': f'n{target}',
                'weight': round(random.uniform(0.1, 1.0), 3),
                'relationship': random.choice(['collaborates', 'reports_to', 'mentors'])
            })
    
    return nodes_data, edges_data


def benchmark_graph_creation():
    """Test graph creation performance"""
    print("\n🏗️ GRAPH CREATION BENCHMARK")
    print("=" * 50)
    
    sizes = [(1000, 500), (2500, 1250), (5000, 2500)]
    
    for n_nodes, n_edges in sizes:
        print(f"\n📊 Testing {n_nodes:,} nodes, {n_edges:,} edges")
        nodes_data, edges_data = create_test_data(n_nodes, n_edges)
        
        # Test Groggy
        def create_groggy():
            graph = gr.Graph(backend='rust')
            graph.add_nodes(nodes_data)
            graph.add_edges(edges_data)  # Batch operation
            return graph
        
        groggy_time, groggy_size, groggy_success, groggy_error = time_function(
            create_groggy, f"Groggy create ({n_nodes:,} nodes)"
        )
        
        # Test NetworkX
        if NETWORKX_AVAILABLE:
            def create_networkx():
                graph = nx.DiGraph()
                # Batch operations
                nodes_for_nx = [(node['id'], {k: v for k, v in node.items() if k != 'id'}) for node in nodes_data]
                graph.add_nodes_from(nodes_for_nx)
                edges_for_nx = [(edge['source'], edge['target'], {k: v for k, v in edge.items() if k not in ['source', 'target']}) for edge in edges_data]
                graph.add_edges_from(edges_for_nx)
                return graph
            
            nx_time, nx_size, nx_success, nx_error = time_function(
                create_networkx, f"NetworkX create ({n_nodes:,} nodes)"
            )
            
            if groggy_success and nx_success:
                if nx_time > 0:
                    speedup = nx_time / groggy_time
                    if speedup > 1.1:
                        print(f"    🚀 Groggy is {speedup:.1f}x faster")
                    elif speedup < 0.9:
                        print(f"    📊 NetworkX is {1/speedup:.1f}x faster")
                    else:
                        print(f"    ⚖️ Similar performance")


def benchmark_operations():
    """Test basic operations"""
    print("\n🔧 BASIC OPERATIONS BENCHMARK")
    print("=" * 50)
    
    # Create test graphs
    nodes_data, edges_data = create_test_data(5000, 2500)
    
    print("Creating graphs for testing...")
    
    # Create Groggy graph
    groggy_graph = gr.Graph(backend='rust')
    groggy_graph.add_nodes(nodes_data)
    groggy_graph.add_edges(edges_data)
    print(f"✅ Groggy graph: {groggy_graph.node_count()} nodes, {groggy_graph.edge_count()} edges")
    
    # Create NetworkX graph
    nx_graph = None
    if NETWORKX_AVAILABLE:
        nx_graph = nx.DiGraph()
        nodes_for_nx = [(node['id'], {k: v for k, v in node.items() if k != 'id'}) for node in nodes_data]
        nx_graph.add_nodes_from(nodes_for_nx)
        edges_for_nx = [(edge['source'], edge['target'], {k: v for k, v in edge.items() if k not in ['source', 'target']}) for edge in edges_data]
        nx_graph.add_edges_from(edges_for_nx)
        print(f"✅ NetworkX graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
    
    print("\nTesting operations:")
    
    # Node count
    groggy_time, _, _, _ = time_function(lambda: groggy_graph.node_count(), "Groggy node count")
    if nx_graph:
        nx_time, _, _, _ = time_function(lambda: nx_graph.number_of_nodes(), "NetworkX node count")
    
    # Get neighbors
    test_node = 'n100'
    groggy_time, groggy_size, _, _ = time_function(lambda: groggy_graph.get_neighbors(test_node), f"Groggy neighbors of {test_node}")
    if nx_graph:
        nx_time, nx_size, _, _ = time_function(lambda: list(nx_graph.neighbors(test_node)), f"NetworkX neighbors of {test_node}")
        print(f"    Results match: {groggy_size == nx_size}")


def benchmark_filtering():
    """Test filtering operations"""
    print("\n🔍 FILTERING BENCHMARK")
    print("=" * 50)
    
    # Create test graphs
    nodes_data, edges_data = create_test_data(10000, 5000)
    
    print("Creating graphs for filtering tests...")
    
    # Create Groggy graph
    groggy_graph = gr.Graph(backend='rust')
    groggy_graph.add_nodes(nodes_data)
    groggy_graph.add_edges(edges_data)
    
    # Create NetworkX graph
    nx_graph = None
    if NETWORKX_AVAILABLE:
        nx_graph = nx.DiGraph()
        nodes_for_nx = [(node['id'], {k: v for k, v in node.items() if k != 'id'}) for node in nodes_data]
        nx_graph.add_nodes_from(nodes_for_nx)
        edges_for_nx = [(edge['source'], edge['target'], {k: v for k, v in edge.items() if k not in ['source', 'target']}) for edge in edges_data]
        nx_graph.add_edges_from(edges_for_nx)
    
    print("\nTesting node filters:")
    
    # Simple role filter
    groggy_time, groggy_size, groggy_success, _ = time_function(
        lambda: groggy_graph.filter_nodes({'role': 'engineer'}), 
        "Groggy role filter"
    )
    
    if nx_graph:
        def nx_role_filter():
            return [node for node, attrs in nx_graph.nodes(data=True) if attrs.get('role') == 'engineer']
        
        nx_time, nx_size, nx_success, _ = time_function(nx_role_filter, "NetworkX role filter")
        
        if groggy_success and nx_success:
            print(f"    Results: Groggy {groggy_size}, NetworkX {nx_size}")
            if nx_time > 0:
                speedup = nx_time / groggy_time
                if speedup > 1.1:
                    print(f"    🚀 Groggy is {speedup:.1f}x faster")
                elif speedup < 0.9:
                    print(f"    📊 NetworkX is {1/speedup:.1f}x faster")
    
    # Salary range filter
    groggy_time, groggy_size, groggy_success, _ = time_function(
        lambda: groggy_graph.filter_nodes({'salary': ('>', 100000)}), 
        "Groggy salary filter"
    )
    
    if nx_graph:
        def nx_salary_filter():
            return [node for node, attrs in nx_graph.nodes(data=True) if attrs.get('salary', 0) > 100000]
        
        nx_time, nx_size, nx_success, _ = time_function(nx_salary_filter, "NetworkX salary filter")
        
        if groggy_success and nx_success:
            print(f"    Results: Groggy {groggy_size}, NetworkX {nx_size}")
            if nx_time > 0:
                speedup = nx_time / groggy_time
                if speedup > 1.1:
                    print(f"    🚀 Groggy is {speedup:.1f}x faster")
                elif speedup < 0.9:
                    print(f"    📊 NetworkX is {1/speedup:.1f}x faster")
    
    print("\nTesting edge filters:")
    
    # Relationship filter
    groggy_time, groggy_size, groggy_success, _ = time_function(
        lambda: groggy_graph.filter_edges({'relationship': 'collaborates'}), 
        "Groggy edge filter"
    )
    
    if nx_graph:
        def nx_edge_filter():
            return [(u, v) for u, v, attrs in nx_graph.edges(data=True) if attrs.get('relationship') == 'collaborates']
        
        nx_time, nx_size, nx_success, _ = time_function(nx_edge_filter, "NetworkX edge filter")
        
        if groggy_success and nx_success:
            print(f"    Results: Groggy {groggy_size}, NetworkX {nx_size}")
            if nx_time > 0:
                speedup = nx_time / groggy_time
                if speedup > 1.1:
                    print(f"    🚀 Groggy is {speedup:.1f}x faster")
                elif speedup < 0.9:
                    print(f"    📊 NetworkX is {1/speedup:.1f}x faster")


def main():
    """Run the simple benchmark suite"""
    print("🚀 SIMPLE GROGGY VS NETWORKX BENCHMARK")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"NetworkX available: {'✅' if NETWORKX_AVAILABLE else '❌'}")
    
    try:
        benchmark_graph_creation()
        benchmark_operations()  
        benchmark_filtering()
        
        print("\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
