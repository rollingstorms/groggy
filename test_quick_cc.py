#!/usr/bin/env python3

"""
Quick test of the optimized connected_components
"""

import time
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def create_test_data(num_nodes, num_edges):
    """Create test data using the same method as benchmark"""
    import random
    
    # Create nodes with attributes
    nodes_data = []
    for i in range(num_nodes):
        nodes_data.append({
            'id': f'node_{i}',
            'department': random.choice(['Engineering', 'Marketing', 'Sales', 'HR']),
            'salary': random.randint(30000, 150000),
            'active': random.choice([True, False]),
            'performance': random.uniform(1.0, 5.0),
        })
    
    # Create edges
    edges_data = []
    for i in range(num_edges):
        source_idx = random.randint(0, num_nodes - 1)
        target_idx = random.randint(0, num_nodes - 1)
        if source_idx != target_idx:
            edges_data.append({
                'source': f'node_{source_idx}',
                'target': f'node_{target_idx}',
            })
    
    return nodes_data[:num_nodes], edges_data[:num_edges]

def quick_test_cc():
    """Quick test of connected_components optimization"""
    import groggy as gr
    
    sizes = [1000, 5000, 10000, 25000, 50000]
    
    for num_nodes in sizes:
        num_edges = num_nodes  # Same ratio as benchmark
        print(f"\nTesting {num_nodes} nodes, {num_edges} edges...")
        
        # Create test data
        nodes_data, edges_data = create_test_data(num_nodes, num_edges)
        
        # Create graph quickly
        graph = gr.Graph()
        bulk_node_ids = graph.add_nodes(num_nodes)
        
        # Create node mapping
        node_id_map = {nodes_data[i]['id']: bulk_node_ids[i] for i in range(num_nodes)}
        
        # Add edges
        edge_specs = []
        for edge in edges_data:
            if edge['source'] in node_id_map and edge['target'] in node_id_map:
                source_id = node_id_map[edge['source']]
                target_id = node_id_map[edge['target']]
                edge_specs.append((source_id, target_id))
        
        graph.add_edges(edge_specs)
        
        print(f"  Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")
        
        # Test connected_components
        start_time = time.time()
        components = graph.connected_components()
        cc_time = time.time() - start_time
        
        print(f"  Connected components: {len(components)} found")
        print(f"  Time: {cc_time:.3f}s")
        print(f"  Rate: {graph.node_count() / cc_time:.0f} nodes/sec")
        
        # Check if it's getting too slow to continue
        if cc_time > 30:  # More than 30 seconds
            print(f"  Too slow, stopping here")
            break

if __name__ == "__main__":
    quick_test_cc()
