#!/usr/bin/env python3
"""
Break down exactly what the benchmark is doing step by step
"""

import time
import random
import groggy as gr

def time_step(func, name):
    """Time a step and print results"""
    start = time.time()
    result = func()
    elapsed = time.time() - start
    print(f"  {name}: {elapsed:.4f}s")
    return result, elapsed

def benchmark_breakdown():
    """Break down the benchmark creation process"""
    
    print("=== Benchmark Creation Breakdown ===")
    
    num_nodes = 50000
    num_edges = 25000
    
    # Step 1: Graph initialization
    def create_graph():
        return gr.Graph()
    
    graph, init_time = time_step(create_graph, "Graph initialization")
    
    # Step 2: Bulk node creation  
    def create_nodes():
        return graph.add_nodes(num_nodes)
    
    bulk_node_ids, node_time = time_step(create_nodes, f"Bulk node creation ({num_nodes})")
    
    # Step 3: Prepare attribute data (this happens in the benchmark)
    def prepare_attr_data():
        departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations']
        
        bulk_attrs_dict = {}
        
        # Department attribute (text)
        dept_values = [random.choice(departments) for _ in range(num_nodes)]
        bulk_attrs_dict['department'] = {
            'nodes': bulk_node_ids,
            'values': dept_values,
            'value_type': 'text'
        }
        
        # Salary attribute (int) 
        salary_values = [random.randint(40000, 200000) for _ in range(num_nodes)]
        bulk_attrs_dict['salary'] = {
            'nodes': bulk_node_ids,
            'values': salary_values,
            'value_type': 'int'
        }
        
        # Performance attribute (float)
        perf_values = [random.uniform(1.0, 5.0) for _ in range(num_nodes)]
        bulk_attrs_dict['performance'] = {
            'nodes': bulk_node_ids,
            'values': perf_values,
            'value_type': 'float'
        }
        
        # Active attribute (bool)
        active_values = [random.choice([True, False]) for _ in range(num_nodes)]
        bulk_attrs_dict['active'] = {
            'nodes': bulk_node_ids,
            'values': active_values,
            'value_type': 'bool'
        }
        
        return bulk_attrs_dict
    
    bulk_attrs_dict, attr_prep_time = time_step(prepare_attr_data, "Prepare attribute data")
    
    # Step 4: Set node attributes in bulk
    def set_node_attrs():
        return graph.set_node_attributes(bulk_attrs_dict)
    
    _, attr_set_time = time_step(set_node_attrs, "Set node attributes (bulk)")
    
    # Step 5: Prepare edge data
    def prepare_edge_data():
        relationships = ['reports_to', 'collaborates_with', 'mentors', 'manages', 'works_with']
        
        edge_specs = []
        edges_data = []
        
        for i in range(num_edges):
            src = random.randint(0, num_nodes-1)
            tgt = random.randint(0, num_nodes-1)
            if src != tgt:
                edge_specs.append((bulk_node_ids[src], bulk_node_ids[tgt]))
                edges_data.append({
                    'relationship': random.choice(relationships),
                    'weight': random.uniform(0.1, 2.0)
                })
        
        return edge_specs, edges_data
    
    (edge_specs, edges_data), edge_prep_time = time_step(prepare_edge_data, "Prepare edge data")
    
    # Step 6: Create edges in bulk
    def create_edges():
        return graph.add_edges(edge_specs)
    
    bulk_edge_ids, edge_create_time = time_step(create_edges, f"Create edges ({len(edge_specs)})")
    
    # Step 7: Prepare edge attribute data
    def prepare_edge_attrs():
        bulk_edge_attrs_dict = {}
        
        # Relationship attribute
        rel_values = [edge['relationship'] for edge in edges_data]
        bulk_edge_attrs_dict['relationship'] = {
            'edges': bulk_edge_ids,
            'values': rel_values,
            'value_type': 'text'
        }
        
        # Weight attribute
        weight_values = [edge['weight'] for edge in edges_data]
        bulk_edge_attrs_dict['weight'] = {
            'edges': bulk_edge_ids,
            'values': weight_values,
            'value_type': 'float'
        }
        
        return bulk_edge_attrs_dict
    
    bulk_edge_attrs_dict, edge_attr_prep_time = time_step(prepare_edge_attrs, "Prepare edge attribute data")
    
    # Step 8: Set edge attributes in bulk
    def set_edge_attrs():
        return graph.set_edge_attributes(bulk_edge_attrs_dict)
    
    _, edge_attr_set_time = time_step(set_edge_attrs, "Set edge attributes (bulk)")
    
    # Summary
    total_time = (init_time + node_time + attr_prep_time + attr_set_time + 
                 edge_prep_time + edge_create_time + edge_attr_prep_time + edge_attr_set_time)
    
    print(f"\n=== Summary ===")
    print(f"Total time: {total_time:.4f}s")
    print(f"Final graph: {graph}")
    
    print(f"\n=== Time Breakdown ===")
    times = [
        ("Graph init", init_time),
        ("Node creation", node_time),
        ("Attr data prep", attr_prep_time),
        ("Node attr set", attr_set_time),
        ("Edge data prep", edge_prep_time), 
        ("Edge creation", edge_create_time),
        ("Edge attr prep", edge_attr_prep_time),
        ("Edge attr set", edge_attr_set_time)
    ]
    
    for name, t in times:
        percentage = (t / total_time) * 100
        print(f"{name}: {t:.4f}s ({percentage:.1f}%)")

if __name__ == "__main__":
    benchmark_breakdown()