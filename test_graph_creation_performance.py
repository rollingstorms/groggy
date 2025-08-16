#!/usr/bin/env python3

"""
Test graph creation performance specifically
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
            'name': f'Node {i}',
            'department': random.choice(['Engineering', 'Marketing', 'Sales', 'HR']),
            'salary': random.randint(30000, 150000),
            'age': random.randint(22, 65),
            'active': random.choice([True, False]),
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
                'relationship': random.choice(['reports_to', 'collaborates', 'mentor']),
                'weight': random.uniform(0.1, 1.0),
                'strength': random.uniform(0.0, 10.0),
            })
    
    return nodes_data, edges_data

def test_graph_creation_performance():
    """Test graph creation at different scales"""
    import groggy as gr
    
    sizes = [
        (1000, 1000),
        (5000, 5000),
        (10000, 10000),
        (25000, 25000),
    ]
    
    for num_nodes, num_edges in sizes:
        print(f"\nTesting graph creation: {num_nodes} nodes, {num_edges} edges")
        
        # Create test data
        print("  Creating test data...")
        start_time = time.time()
        nodes_data, edges_data = create_test_data(num_nodes, num_edges)
        data_creation_time = time.time() - start_time
        print(f"  Test data created in {data_creation_time:.3f}s")
        
        # Create graph using the same method as benchmark
        print("  Creating graph...")
        start_time = time.time()
        
        graph = gr.Graph()
        
        # Create nodes in bulk - much faster!
        bulk_node_ids = graph.add_nodes(num_nodes)
        
        # Create node ID mapping
        node_id_map = {}
        for i, node_data in enumerate(nodes_data):
            node_id_map[node_data['id']] = bulk_node_ids[i]
        
        node_creation_time = time.time() - start_time
        print(f"  Nodes created in {node_creation_time:.3f}s")
        
        # Set node attributes in bulk
        start_time = time.time()
        
        bulk_attrs_dict = {}
        for attr_name in ['name', 'department', 'salary', 'age', 'active']:
            values_list = [node[attr_name] for node in nodes_data]
            
            # Determine value type
            if attr_name in ['salary', 'age']:
                value_type = 'int'
            elif attr_name == 'active':
                value_type = 'bool'
            else:
                value_type = 'text'
            
            bulk_attrs_dict[attr_name] = {
                'nodes': bulk_node_ids,
                'values': values_list,
                'value_type': value_type
            }
        
        if bulk_attrs_dict:
            graph.set_node_attributes(bulk_attrs_dict)
        
        attr_setting_time = time.time() - start_time
        print(f"  Node attributes set in {attr_setting_time:.3f}s")
        
        # Add edges
        start_time = time.time()
        edge_specs = []
        for edge in edges_data:
            if edge['source'] in node_id_map and edge['target'] in node_id_map:
                graph_source = node_id_map[edge['source']]
                graph_target = node_id_map[edge['target']]
                edge_specs.append((graph_source, graph_target))
        
        edge_ids = graph.add_edges(edge_specs)
        edge_creation_time = time.time() - start_time
        print(f"  Edges created in {edge_creation_time:.3f}s")
        
        total_time = node_creation_time + attr_setting_time + edge_creation_time
        print(f"  Total graph creation: {total_time:.3f}s")
        print(f"  Rate: {num_nodes / total_time:.0f} nodes/sec")
        
        # Check if getting too slow
        if total_time > 5.0:
            print(f"  Breaking - getting too slow")
            break

if __name__ == "__main__":
    test_graph_creation_performance()
