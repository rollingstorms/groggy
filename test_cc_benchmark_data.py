#!/usr/bin/env python3

"""
Test connected components performance using benchmark's create_test_data method
"""

import time
import sys
import os

# Add the benchmark path so we can import create_test_data
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy')

from benchmark_graph_libraries import create_test_data
import groggy as gr

def test_connected_components_scaling():
    """Test connected components with different scales using benchmark data"""
    
    # Test different scales that match the benchmark
    test_cases = [
        (1000, 1000, "Small"),
        (5000, 5000, "Medium-Small"), 
        (10000, 10000, "Medium"),
        (25000, 25000, "Large"), 
        (50000, 50000, "X-Large"),  # This is where it hangs in benchmark
    ]
    
    for num_nodes, num_edges, scale_name in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing {scale_name}: {num_nodes:,} nodes, {num_edges:,} edges")
        print(f"{'='*60}")
        
        try:
            # Create test data using benchmark method
            print("Creating test data...")
            data_start = time.time()
            nodes_data, edges_data = create_test_data(num_nodes, num_edges)
            data_time = time.time() - data_start
            print(f"Test data created in {data_time:.3f}s")
            
            # Create graph using the same method as benchmark
            print("Creating graph...")
            graph_start = time.time()
            
            graph = gr.Graph()
            
            # Use bulk node creation
            bulk_node_ids = graph.add_nodes(num_nodes)
            
            # Create node mapping
            node_id_map = {}
            for i, node in enumerate(nodes_data):
                original_id = node['id']
                graph_node_id = bulk_node_ids[i]
                node_id_map[original_id] = graph_node_id
            
            # Set essential attributes only (like benchmark)
            essential_attributes = ['department', 'salary', 'active', 'performance']
            
            bulk_attrs_dict = {}
            for attr_name in essential_attributes:
                values_list = []
                
                for i, node in enumerate(nodes_data):
                    if attr_name == 'department':
                        values_list.append(node['department'])
                    elif attr_name == 'salary':
                        values_list.append(node['salary'])
                    elif attr_name == 'active':
                        values_list.append(node['active'])
                    elif attr_name == 'performance':
                        values_list.append(node['performance'])
                
                # Determine value type
                if attr_name in ['department']:
                    value_type = 'text'
                elif attr_name in ['salary']:
                    value_type = 'int'
                elif attr_name in ['active']:
                    value_type = 'bool'
                elif attr_name in ['performance']:
                    value_type = 'float'
                
                bulk_attrs_dict[attr_name] = {
                    'nodes': bulk_node_ids,
                    'values': values_list,
                    'value_type': value_type
                }
            
            # Set node attributes
            if bulk_attrs_dict:
                graph.set_node_attributes(bulk_attrs_dict)
            
            # Create edges
            edge_specs = []
            for edge in edges_data:
                graph_source = node_id_map[edge['source']]
                graph_target = node_id_map[edge['target']]
                edge_specs.append((graph_source, graph_target))
            
            bulk_edge_ids = graph.add_edges(edge_specs)
            
            graph_time = time.time() - graph_start
            print(f"Graph created in {graph_time:.3f}s")
            print(f"Final graph: {graph.node_count()} nodes, {graph.edge_count()} edges")
            
            # Test connected components with timeout
            print("Testing connected_components()...")
            
            # Set timeout for large tests
            timeout_seconds = 30 if num_nodes >= 25000 else 10
            
            cc_start = time.time()
            try:
                components = graph.connected_components()
                cc_time = time.time() - cc_start
                
                print(f"✅ Connected components: {len(components)} found in {cc_time:.3f}s")
                print(f"Rate: {graph.node_count() / cc_time:.0f} nodes/sec")
                
                # Check if it's getting too slow
                if cc_time > timeout_seconds:
                    print(f"⚠️ Warning: Taking {cc_time:.1f}s for {num_nodes:,} nodes")
                    print(f"   This suggests O(n²) or worse scaling")
                    
            except KeyboardInterrupt:
                cc_time = time.time() - cc_start
                print(f"❌ Interrupted after {cc_time:.1f}s - too slow!")
                break
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Check if we should continue
        if num_nodes >= 50000:
            response = input("\nContinue with larger tests? (y/n): ")
            if response.lower() != 'y':
                break

def test_connected_components_simple():
    """Simple test with known good size"""
    print("Testing with 5000 nodes (known good size)...")
    
    nodes_data, edges_data = create_test_data(5000, 5000)
    
    graph = gr.Graph()
    bulk_node_ids = graph.add_nodes(5000)
    
    # Simple node mapping
    node_id_map = {i: bulk_node_ids[i] for i in range(5000)}
    
    # Add edges
    edge_specs = []
    for edge in edges_data:
        edge_specs.append((node_id_map[edge['source']], node_id_map[edge['target']]))
    
    graph.add_edges(edge_specs)
    
    print(f"Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")
    
    start_time = time.time()
    components = graph.connected_components()
    cc_time = time.time() - start_time
    
    print(f"Connected components: {len(components)} in {cc_time:.4f}s")
    print(f"Rate: {graph.node_count() / cc_time:.0f} nodes/sec")

if __name__ == "__main__":
    print("Connected Components Scaling Test")
    print("Using benchmark's create_test_data method")
    print("="*60)
    
    # First try a simple test
    test_connected_components_simple()
    
    # Then try scaling test
    test_connected_components_scaling()
