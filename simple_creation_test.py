#!/usr/bin/env python3
"""
Simplified version of what the benchmark does for graph creation
"""

import time
import random
import groggy as gr

def create_benchmark_style_graph():
    """Create graph exactly like the benchmark does"""
    print("Creating graph like benchmark...")
    
    # Same parameters as benchmark
    num_nodes = 50000
    num_edges = 25000
    
    print("1. Creating test data...")
    start_data = time.time()
    
    # Generate nodes with rich attributes (same as benchmark)
    nodes_data = []
    departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations']
    
    for i in range(num_nodes):
        node = {
            'id': i,
            'department': random.choice(departments),
            'salary': random.randint(40000, 200000),
            'active': random.choice([True, False]),
            'performance': random.uniform(1.0, 5.0)
        }
        nodes_data.append(node)
    
    # Generate edges data
    edges_data = []
    relationships = ['reports_to', 'collaborates_with', 'mentors', 'manages', 'works_with']
    
    for i in range(num_edges):
        src = random.randint(0, num_nodes-1)
        tgt = random.randint(0, num_nodes-1)
        if src != tgt:
            edge = {
                'source': src,
                'target': tgt,
                'relationship': random.choice(relationships),
                'weight': random.uniform(0.1, 2.0)
            }
            edges_data.append(edge)
    
    data_time = time.time() - start_data
    print(f"   Generated test data in {data_time:.4f}s")
    
    print("2. Creating graph...")
    start_graph = time.time()
    
    graph = gr.Graph()
    
    # Use bulk node creation for better performance (like benchmark)
    print("   Adding nodes...")
    bulk_node_ids = graph.add_nodes(num_nodes)
    
    # Create mapping from original ID to graph ID (like benchmark)
    node_id_map = {}
    for i, node in enumerate(nodes_data):
        original_id = node['id']
        graph_node_id = bulk_node_ids[i]
        node_id_map[original_id] = graph_node_id
    
    print("   Setting node attributes...")
    # Use optimized bulk attribute setting (like benchmark)
    essential_attributes = ['department', 'salary', 'active', 'performance']
    bulk_attrs_dict = {}
    
    for attr_name in essential_attributes:
        values_list = []
        
        for i, node in enumerate(nodes_data):
            values_list.append(node[attr_name])
        
        # Determine value type for bulk API
        if attr_name == 'department':
            value_type = 'text'
        elif attr_name == 'salary':
            value_type = 'int'
        elif attr_name == 'active':
            value_type = 'bool'
        elif attr_name == 'performance':
            value_type = 'float'
        
        bulk_attrs_dict[attr_name] = {
            'nodes': bulk_node_ids,
            'values': values_list,
            'value_type': value_type
        }
    
    # Set ALL node attributes in a single optimized bulk operation
    graph.set_node_attributes(bulk_attrs_dict)
    
    print("   Adding edges...")
    # Use bulk edge creation (like benchmark)
    edge_specs = []
    for edge in edges_data:
        graph_source = node_id_map[edge['source']]
        graph_target = node_id_map[edge['target']]
        edge_specs.append((graph_source, graph_target))
    
    bulk_edge_ids = graph.add_edges(edge_specs)
    
    print("   Setting edge attributes...")
    # Use optimized bulk edge attribute setting (like benchmark)
    essential_edge_attributes = ['relationship', 'weight']
    bulk_edge_attrs_dict = {}
    
    for attr_name in essential_edge_attributes:
        values_list = []
        
        for i, edge in enumerate(edges_data):
            values_list.append(edge[attr_name])
        
        # Determine value type for bulk API
        if attr_name == 'relationship':
            value_type = 'text'
        elif attr_name == 'weight':
            value_type = 'float'
        
        bulk_edge_attrs_dict[attr_name] = {
            'edges': bulk_edge_ids,
            'values': values_list,
            'value_type': value_type
        }
    
    # Set ALL edge attributes in a single optimized bulk operation
    graph.set_edge_attributes(bulk_edge_attrs_dict)
    
    creation_time = time.time() - start_graph
    total_time = data_time + creation_time
    
    print(f"\n=== Results ===")
    print(f"Data generation: {data_time:.4f}s")
    print(f"Graph creation: {creation_time:.4f}s")
    print(f"Total time: {total_time:.4f}s")
    print(f"Graph: {graph}")
    
    return graph, creation_time

if __name__ == "__main__":
    create_benchmark_style_graph()