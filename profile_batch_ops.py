#!/usr/bin/env python3
"""
Profile batch operations to identify specific bottlenecks.
"""

import cProfile
import pstats
import io
import time
import random
import string
from python.groggy import Graph

def generate_node_data(n_nodes, n_attrs_per_node=3):
    """Generate test data for batch node addition."""
    nodes = []
    for i in range(n_nodes):
        node_data = {"id": f"node_{i}"}
        
        # Generate attributes
        for j in range(n_attrs_per_node):
            attr_name = f"attr_{j}"
            # Mix of attribute types
            if j % 3 == 0:
                node_data[attr_name] = random.randint(0, 1000)
            elif j % 3 == 1:
                node_data[attr_name] = random.random() * 100.0
            else:
                node_data[attr_name] = ''.join(random.choices(string.ascii_letters, k=10))
        
        nodes.append(node_data)
    
    return nodes

def generate_edge_data(n_edges, n_nodes, n_attrs_per_edge=2):
    """Generate test data for batch edge addition."""
    edges = []
    for i in range(n_edges):
        edge_data = {
            "source": f"node_{random.randint(0, n_nodes-1)}",
            "target": f"node_{random.randint(0, n_nodes-1)}"
        }
        
        # Generate attributes
        for j in range(n_attrs_per_edge):
            attr_name = f"edge_attr_{j}"
            if j % 2 == 0:
                edge_data[attr_name] = random.random()
            else:
                edge_data[attr_name] = random.randint(0, 100)
        
        edges.append(edge_data)
    
    return edges

def profile_batch_nodes():
    """Profile batch node addition."""
    print("Profiling batch node addition...")
    
    # Generate test data
    n_nodes = 10000
    nodes_data = generate_node_data(n_nodes, n_attrs_per_node=5)
    
    g = Graph()
    
    # Profile the batch addition
    pr = cProfile.Profile()
    pr.enable()
    
    g.add_nodes(nodes_data)
    
    pr.disable()
    
    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("Batch node addition profile:")
    print(s.getvalue())

def profile_batch_edges():
    """Profile batch edge addition."""
    print("Profiling batch edge addition...")
    
    # Generate test data
    n_nodes = 5000
    n_edges = 20000
    
    # First add nodes
    nodes_data = generate_node_data(n_nodes, n_attrs_per_node=3)
    g = Graph()
    g.add_nodes(nodes_data)
    
    # Generate edges
    edges_data = generate_edge_data(n_edges, n_nodes, n_attrs_per_edge=3)
    
    # Profile the batch addition
    pr = cProfile.Profile()
    pr.enable()
    
    g.add_edges(edges_data)
    
    pr.disable()
    
    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("Batch edge addition profile:")
    print(s.getvalue())

def profile_python_conversion():
    """Profile just the Python → JSON conversion overhead."""
    print("Profiling Python → JSON conversion...")
    
    # Generate test data
    test_attrs = []
    for i in range(10000):
        attrs = {
            f"attr_{j}": random.choice([
                random.randint(0, 1000),
                random.random() * 100.0,
                ''.join(random.choices(string.ascii_letters, k=10)),
                [1, 2, 3],
                {"nested": "value"}
            ])
            for j in range(5)
        }
        test_attrs.append(attrs)
    
    # Profile conversion
    pr = cProfile.Profile()
    pr.enable()
    
    # Simulate the conversion that happens in batch operations
    from python.groggy.graph.core import Graph as CoreGraph
    core_graph = CoreGraph()
    
    # This should trigger the conversion logic
    for attrs in test_attrs:
        # Call internal conversion (if accessible)
        pass  # We'd need to expose the conversion function
    
    pr.disable()
    
    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    
    print("Python conversion profile:")
    print(s.getvalue())

def timing_comparison():
    """Compare timing of different batch sizes."""
    print("Timing comparison for different batch sizes...")
    
    batch_sizes = [100, 500, 1000, 5000, 10000]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Node timing
        nodes_data = generate_node_data(batch_size, n_attrs_per_node=3)
        g = Graph()
        
        start = time.time()
        g.add_nodes(nodes_data)
        node_time = time.time() - start
        
        print(f"  Nodes: {node_time:.4f}s ({node_time*1e6/batch_size:.2f}μs per node)")
        
        # Edge timing
        n_edges = batch_size * 2
        edges_data = generate_edge_data(n_edges, batch_size, n_attrs_per_edge=2)
        
        start = time.time()
        g.add_edges(edges_data)
        edge_time = time.time() - start
        
        print(f"  Edges: {edge_time:.4f}s ({edge_time*1e6/n_edges:.2f}μs per edge)")

if __name__ == "__main__":
    print("=== Batch Operations Profiling ===\n")
    
    # Run timing comparison first
    timing_comparison()
    
    print("\n" + "="*50 + "\n")
    
    # Run detailed profiling
    profile_batch_nodes()
    
    print("\n" + "="*50 + "\n")
    
    profile_batch_edges()
