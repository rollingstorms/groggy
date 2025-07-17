#!/usr/bin/env python3
"""
Memory diagnostic script to analyze Groggy's memory usage patterns
"""

import sys
import os
import time
import random
import gc
import tracemalloc
import psutil
from typing import Dict, List, Any

# Remove any existing groggy from the module cache
modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('groggy')]
for mod in modules_to_remove:
    del sys.modules[mod]

# Add local development version
local_groggy_path = '/Users/michaelroth/Documents/Code/groggy/python'
sys.path.insert(0, local_groggy_path)

import groggy as gr

try:
    import objgraph
    HAS_OBJGRAPH = True
except ImportError:
    HAS_OBJGRAPH = False
    print("objgraph not available - install with: pip install objgraph")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def format_memory(mb):
    """Format memory in MB with appropriate units"""
    if mb < 1024:
        return f"{mb:.1f} MB"
    else:
        return f"{mb/1024:.2f} GB"

def create_test_data(num_nodes: int, num_edges: int):
    """Create test data"""
    print(f"Generating test data: {num_nodes:,} nodes, {num_edges:,} edges...")
    
    nodes_data = []
    for i in range(num_nodes):
        nodes_data.append({
            'id': f"n{i}",
            'role': random.choice(['engineer', 'manager', 'designer', 'analyst']),
            'department': random.choice(['AI', 'Web', 'Mobile', 'Data']),
            'salary': random.randint(50000, 150000),
            'active': random.choice([True, False]),
            'level': random.randint(1, 5),
            'value': random.randint(1, 1000)
        })
    
    edges_data = []
    for i in range(num_edges):
        src = random.randint(0, num_nodes-1)
        tgt = random.randint(0, num_nodes-1)
        if src != tgt:
            edges_data.append({
                'source': f"n{src}",
                'target': f"n{tgt}",
                'relationship': random.choice(['reports_to', 'collaborates', 'manages']),
                'strength': random.uniform(0.1, 1.0),
                'weight': random.uniform(0.1, 1.0)
            })
    
    return nodes_data, edges_data

def analyze_memory_growth():
    """Analyze memory growth patterns during graph creation"""
    print("="*70)
    print("MEMORY GROWTH ANALYSIS")
    print("="*70)
    
    # Start tracemalloc
    tracemalloc.start()
    
    # Initial memory snapshot
    initial_memory = get_memory_usage()
    print(f"Initial memory: {format_memory(initial_memory)}")
    
    # Create test data
    nodes_data, edges_data = create_test_data(5000, 5000)
    after_data_memory = get_memory_usage()
    print(f"After creating test data: {format_memory(after_data_memory)} (+{format_memory(after_data_memory - initial_memory)})")
    
    # Create empty graph
    graph = gr.Graph()
    after_graph_memory = get_memory_usage()
    print(f"After creating empty graph: {format_memory(after_graph_memory)} (+{format_memory(after_graph_memory - after_data_memory)})")
    
    # Add nodes
    print("\nAdding nodes...")
    graph.nodes.add(nodes_data)
    after_nodes_memory = get_memory_usage()
    print(f"After adding nodes: {format_memory(after_nodes_memory)} (+{format_memory(after_nodes_memory - after_graph_memory)})")
    
    # Add edges
    print("Adding edges...")
    graph.edges.add(edges_data)
    after_edges_memory = get_memory_usage()
    print(f"After adding edges: {format_memory(after_edges_memory)} (+{format_memory(after_edges_memory - after_nodes_memory)})")
    
    # Get tracemalloc snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("\n" + "="*50)
    print("TOP MEMORY ALLOCATIONS")
    print("="*50)
    for index, stat in enumerate(top_stats[:10]):
        print(f"{index+1}. {stat}")
    
    # Clean up data references
    print("\nCleaning up data references...")
    del nodes_data, edges_data
    gc.collect()
    
    after_cleanup_memory = get_memory_usage()
    print(f"After cleanup: {format_memory(after_cleanup_memory)} (-{format_memory(after_edges_memory - after_cleanup_memory)})")
    
    # Check what's still in memory
    if HAS_OBJGRAPH:
        print("\n" + "="*50)
        print("OBJECT COUNTS AFTER CLEANUP")
        print("="*50)
        objgraph.show_most_common_types(limit=15)
        
        # Look for potential leaks
        print("\nLooking for potential groggy objects...")
        for obj_type in ['Graph', 'NodeCollection', 'EdgeCollection']:
            count = len(objgraph.by_type(obj_type))
            if count > 0:
                print(f"{obj_type}: {count} objects")
    
    # Check graph info
    print(f"\nGraph info: {graph.info()}")
    print(f"Node count: {len(graph.nodes)}")
    print(f"Edge count: {len(graph.edges)}")
    
    return graph

def compare_with_networkx():
    """Compare memory usage with NetworkX"""
    if not HAS_NETWORKX:
        print("NetworkX not available for comparison")
        return
    
    print("\n" + "="*70)
    print("COMPARISON WITH NETWORKX")
    print("="*70)
    
    # Create same test data
    nodes_data, edges_data = create_test_data(5000, 5000)
    
    # Test Groggy
    print("\nTesting Groggy:")
    gc.collect()
    groggy_before = get_memory_usage()
    
    groggy_graph = gr.Graph()
    groggy_graph.nodes.add(nodes_data)
    groggy_graph.edges.add(edges_data)
    
    groggy_after = get_memory_usage()
    groggy_usage = groggy_after - groggy_before
    print(f"Groggy memory usage: {format_memory(groggy_usage)}")
    
    # Test NetworkX
    print("\nTesting NetworkX:")
    gc.collect()
    nx_before = get_memory_usage()
    
    nx_graph = nx.DiGraph()
    for node in nodes_data:
        node_id = node['id']
        attrs = {k: v for k, v in node.items() if k != 'id'}
        nx_graph.add_node(node_id, **attrs)
    
    for edge in edges_data:
        attrs = {k: v for k, v in edge.items() if k not in ['source', 'target']}
        nx_graph.add_edge(edge['source'], edge['target'], **attrs)
    
    nx_after = get_memory_usage()
    nx_usage = nx_after - nx_before
    print(f"NetworkX memory usage: {format_memory(nx_usage)}")
    
    # Compare
    ratio = groggy_usage / nx_usage if nx_usage > 0 else 0
    print(f"\nGroggy uses {ratio:.1f}x more memory than NetworkX")
    
    # Clean up
    del nodes_data, edges_data, groggy_graph, nx_graph
    gc.collect()

def detailed_object_analysis():
    """Detailed analysis of object types and references"""
    if not HAS_OBJGRAPH:
        print("objgraph not available - skipping detailed analysis")
        return
    
    print("\n" + "="*70)
    print("DETAILED OBJECT ANALYSIS")
    print("="*70)
    
    # Take baseline
    gc.collect()
    baseline_counts = {}
    for obj_type in ['dict', 'list', 'tuple', 'str', 'int', 'float']:
        baseline_counts[obj_type] = len(objgraph.by_type(obj_type))
    
    print("Baseline object counts:")
    for obj_type, count in baseline_counts.items():
        print(f"  {obj_type}: {count}")
    
    # Create small graph
    print("\nCreating small graph (1000 nodes, 1000 edges)...")
    nodes_data, edges_data = create_test_data(1000, 1000)
    
    graph = gr.Graph()
    graph.nodes.add(nodes_data)
    graph.edges.add(edges_data)
    
    # Check new object counts
    print("\nObject counts after graph creation:")
    for obj_type in ['dict', 'list', 'tuple', 'str', 'int', 'float']:
        current_count = len(objgraph.by_type(obj_type))
        diff = current_count - baseline_counts[obj_type]
        print(f"  {obj_type}: {current_count} (+{diff})")
    
    # Look for references to our data
    print("\nLooking for references to test data...")
    if nodes_data:
        refs = objgraph.find_backref_chain(
            nodes_data[0], 
            objgraph.is_proper_module
        )
        if refs:
            print("Found reference chain to first node:")
            for i, ref in enumerate(refs):
                print(f"  {i}: {ref}")
    
    # Clean up and check again
    del nodes_data, edges_data
    gc.collect()
    
    print("\nObject counts after cleanup:")
    for obj_type in ['dict', 'list', 'tuple', 'str', 'int', 'float']:
        current_count = len(objgraph.by_type(obj_type))
        diff = current_count - baseline_counts[obj_type]
        print(f"  {obj_type}: {current_count} (+{diff})")

def main():
    """Main diagnostic function"""
    print("🔍 GROGGY MEMORY DIAGNOSTIC")
    print("="*70)
    
    # Basic memory analysis
    graph = analyze_memory_growth()
    
    # Compare with NetworkX
    compare_with_networkx()
    
    # Detailed object analysis
    detailed_object_analysis()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()