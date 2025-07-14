#!/usr/bin/env python3
"""
Focused performance test for Groggy bulk operations vs NetworkX
Tests bulk node creation, bulk attribute retrieval, and DataFrame conversion
"""
import time
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import groggy
import networkx as nx

def timer(func):
    """Simple timing decorator"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    return wrapper

@timer
def create_groggy_graph_bulk(nodes_data, edges_data):
    """Create Groggy graph using bulk operations"""
    g = groggy.Graph(backend='rust')
    
    # Convert tuple format to dict format for add_nodes
    nodes_dict_format = []
    for node_id, attrs in nodes_data:
        node_dict = {'id': node_id}
        node_dict.update(attrs)
        nodes_dict_format.append(node_dict)
    
    g.add_nodes(nodes_dict_format)
    
    # Convert edges format for add_edges
    if hasattr(g, 'add_edges'):
        edges_dict_format = []
        for source, target, attrs in edges_data:
            edge_dict = {'source': source, 'target': target}
            edge_dict.update(attrs)
            edges_dict_format.append(edge_dict)
        g.add_edges(edges_dict_format)
    else:
        # Fallback to individual edge addition
        for source, target, attrs in edges_data:
            g.add_edge(source, target, **attrs)
    
    return g

@timer 
def create_networkx_graph_bulk(nodes_data, edges_data):
    """Create NetworkX graph using bulk operations"""
    g = nx.Graph()
    
    # NetworkX bulk node creation
    g.add_nodes_from(nodes_data)
    g.add_edges_from(edges_data)
    
    return g

@timer
def groggy_bulk_attribute_retrieval(g, node_ids, attr_names):
    """Test Groggy bulk attribute retrieval"""
    results = {}
    
    # Test single attribute bulk retrieval
    for attr_name in attr_names:
        if hasattr(g, 'get_nodes_attribute'):
            results[attr_name] = g.get_nodes_attribute(node_ids, attr_name)
        else:
            # Fallback to individual retrieval
            results[attr_name] = {nid: g.get_node_attribute(nid, attr_name) 
                                  for nid in node_ids 
                                  if g.get_node_attribute(nid, attr_name) is not None}
    
    # Test multi-attribute bulk retrieval
    if hasattr(g, 'get_nodes_attributes'):
        multi_attrs = g.get_nodes_attributes(node_ids)
        results['multi_attrs'] = multi_attrs
    
    return results

@timer
def networkx_bulk_attribute_retrieval(g, node_ids, attr_names):
    """Test NetworkX bulk attribute retrieval"""
    results = {}
    
    # NetworkX doesn't have true bulk retrieval - simulate it
    for attr_name in attr_names:
        results[attr_name] = {nid: g.nodes[nid].get(attr_name) 
                              for nid in node_ids 
                              if nid in g.nodes and attr_name in g.nodes[nid]}
    
    # Multi-attribute retrieval
    multi_attrs = {nid: dict(g.nodes[nid]) for nid in node_ids if nid in g.nodes}
    results['multi_attrs'] = multi_attrs
    
    return results

@timer
def groggy_dataframe_conversion(g, attr_names):
    """Test Groggy DataFrame conversion if available"""
    if hasattr(g, 'to_dataframe'):
        try:
            return g.to_dataframe(attr_names=attr_names, library='pandas')
        except ImportError:
            print("pandas not available, skipping DataFrame conversion")
            return None
    else:
        print("to_dataframe method not available")
        return None

def generate_test_data(num_nodes, num_edges, num_attributes):
    """Generate test data for bulk operations"""
    print(f"Generating test data: {num_nodes} nodes, {num_edges} edges, {num_attributes} attributes")
    
    # Generate node data with attributes
    nodes_data = []
    for i in range(num_nodes):
        node_attrs = {}
        for j in range(num_attributes):
            if j % 3 == 0:  # numeric
                node_attrs[f'attr_num_{j}'] = random.uniform(0, 1000)
            elif j % 3 == 1:  # string
                node_attrs[f'attr_str_{j}'] = f"value_{random.randint(0, 100)}"
            else:  # boolean
                node_attrs[f'attr_bool_{j}'] = random.choice([True, False])
        
        nodes_data.append((f"node_{i}", node_attrs))
    
    # Generate edge data
    edges_data = []
    node_ids = [f"node_{i}" for i in range(num_nodes)]
    for _ in range(num_edges):
        source = random.choice(node_ids)
        target = random.choice(node_ids)
        edge_attrs = {
            'weight': random.uniform(0, 10),
            'type': random.choice(['A', 'B', 'C'])
        }
        edges_data.append((source, target, edge_attrs))
    
    return nodes_data, edges_data

def run_performance_test(num_nodes, num_edges, num_attributes):
    """Run comprehensive performance test"""
    print(f"\n{'='*60}")
    print(f"PERFORMANCE TEST: {num_nodes} nodes, {num_edges} edges, {num_attributes} attributes")
    print(f"{'='*60}")
    
    # Generate test data
    nodes_data, edges_data = generate_test_data(num_nodes, num_edges, num_attributes)
    
    # Test graph creation
    print("\n1. GRAPH CREATION:")
    print("-" * 30)
    
    groggy_graph, groggy_time = create_groggy_graph_bulk(nodes_data, edges_data)
    print(f"Groggy (bulk):   {groggy_time:.4f}s")
    
    nx_graph, nx_time = create_networkx_graph_bulk(nodes_data, edges_data)
    print(f"NetworkX (bulk): {nx_time:.4f}s")
    
    speedup = nx_time / groggy_time if groggy_time > 0 else float('inf')
    print(f"NetworkX is {speedup:.2f}x faster at graph creation")
    
    # Test bulk attribute retrieval
    print("\n2. BULK ATTRIBUTE RETRIEVAL:")
    print("-" * 30)
    
    # Test on subset of nodes
    test_node_ids = [f"node_{i}" for i in range(0, min(num_nodes, 1000), 10)]
    attr_names = [f'attr_num_{i}' for i in range(0, min(num_attributes, 5))]
    
    groggy_results, groggy_attr_time = groggy_bulk_attribute_retrieval(groggy_graph, test_node_ids, attr_names)
    print(f"Groggy bulk attr:   {groggy_attr_time:.4f}s")
    
    nx_results, nx_attr_time = networkx_bulk_attribute_retrieval(nx_graph, test_node_ids, attr_names)
    print(f"NetworkX sim bulk:  {nx_attr_time:.4f}s")
    
    attr_speedup = nx_attr_time / groggy_attr_time if groggy_attr_time > 0 else float('inf')
    print(f"NetworkX is {attr_speedup:.2f}x faster at attribute retrieval")
    
    # Test DataFrame conversion
    print("\n3. DATAFRAME CONVERSION:")
    print("-" * 30)
    
    df_result, df_time = groggy_dataframe_conversion(groggy_graph, attr_names)
    if df_result is not None:
        print(f"Groggy DataFrame:   {df_time:.4f}s")
        print(f"DataFrame shape:    {df_result.shape}")
    else:
        print("DataFrame conversion not available or failed")
    
    # Memory usage comparison
    print("\n4. MEMORY ANALYSIS:")
    print("-" * 30)
    
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Current memory usage: {memory_mb:.1f} MB")
    except ImportError:
        print("psutil not available for memory analysis")
    
    # Storage efficiency check
    if hasattr(groggy_graph, '_rust_core') and hasattr(groggy_graph._rust_core, 'columnar_store'):
        stats = groggy_graph._rust_core.columnar_store.get_stats()
        print(f"Groggy storage stats: {stats}")
    
    return {
        'graph_creation_speedup': speedup,
        'attr_retrieval_speedup': attr_speedup,
        'groggy_creation_time': groggy_time,
        'nx_creation_time': nx_time,
        'groggy_attr_time': groggy_attr_time,
        'nx_attr_time': nx_attr_time
    }

def test_scalability():
    """Test how performance scales with graph size"""
    print("\n" + "="*60)
    print("SCALABILITY ANALYSIS")
    print("="*60)
    
    test_sizes = [
        (100, 200, 5),      # Small
        (1000, 2000, 10),   # Medium
        (5000, 10000, 15),  # Large
        (10000, 20000, 20), # X-Large
    ]
    
    results = []
    for num_nodes, num_edges, num_attrs in test_sizes:
        try:
            result = run_performance_test(num_nodes, num_edges, num_attrs)
            result['size'] = (num_nodes, num_edges, num_attrs)
            results.append(result)
        except Exception as e:
            print(f"Error with size {num_nodes}x{num_edges}: {e}")
            break
    
    # Print summary
    print("\n" + "="*60)
    print("SCALABILITY SUMMARY")
    print("="*60)
    print(f"{'Size (N,E,A)':<15} {'Creation':<12} {'Attr Retr':<12} {'Groggy Time':<12} {'NX Time':<10}")
    print("-" * 70)
    
    for r in results:
        size_str = f"{r['size'][0]},{r['size'][1]},{r['size'][2]}"
        print(f"{size_str:<15} {r['graph_creation_speedup']:<12.2f} {r['attr_retrieval_speedup']:<12.2f} "
              f"{r['groggy_creation_time']:<12.4f} {r['nx_creation_time']:<10.4f}")
    
    return results

def identify_bottlenecks():
    """Run targeted tests to identify specific bottlenecks"""
    print("\n" + "="*60)
    print("BOTTLENECK ANALYSIS")
    print("="*60)
    
    # Test 1: Pure node creation without attributes
    print("\n1. Pure node creation (no attributes):")
    nodes_simple = [(f"node_{i}", {}) for i in range(1000)]
    edges_simple = []
    
    g1, t1 = create_groggy_graph_bulk(nodes_simple, edges_simple)
    nx1, nt1 = create_networkx_graph_bulk(nodes_simple, edges_simple)
    print(f"   Groggy: {t1:.4f}s, NetworkX: {nt1:.4f}s, Ratio: {nt1/t1:.2f}x")
    
    # Test 2: Node creation with many attributes
    print("\n2. Node creation with 20 attributes:")
    nodes_attrs = []
    for i in range(1000):
        attrs = {f"attr_{j}": random.random() for j in range(20)}
        nodes_attrs.append((f"node_{i}", attrs))
    
    g2, t2 = create_groggy_graph_bulk(nodes_attrs, [])
    nx2, nt2 = create_networkx_graph_bulk(nodes_attrs, [])
    print(f"   Groggy: {t2:.4f}s, NetworkX: {nt2:.4f}s, Ratio: {nt2/t2:.2f}x")
    
    # Test 3: Single attribute retrieval scaling
    print("\n3. Single attribute retrieval scaling:")
    node_ids = [f"node_{i}" for i in range(100)]
    
    _, single_t = groggy_bulk_attribute_retrieval(g2, node_ids, ['attr_0'])
    _, single_nt = networkx_bulk_attribute_retrieval(nx2, node_ids, ['attr_0'])
    print(f"   Groggy: {single_t:.4f}s, NetworkX: {single_nt:.4f}s, Ratio: {single_nt/single_t:.2f}x")

if __name__ == "__main__":
    print("GROGGY BULK PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Run scalability tests
    scalability_results = test_scalability()
    
    # Run bottleneck analysis
    identify_bottlenecks()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print("1. Graph creation performance ratio at different scales")
    print("2. Bulk attribute retrieval performance comparison") 
    print("3. Identification of specific bottlenecks")
    print("4. Memory usage patterns")
    print("\nNext steps: Focus optimization on the biggest bottlenecks identified above.")
