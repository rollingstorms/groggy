#!/usr/bin/env python3
"""
Test bulk attribute setting performance
"""

import time
import groggy as gr

def test_bulk_attribute_performance():
    """Test bulk attribute setting performance"""
    print("Testing bulk attribute setting performance...")
    
    # Create a graph with nodes
    graph = gr.Graph()
    
    print("\n=== Step 1: Create nodes ===")
    start = time.time()
    node_ids = graph.add_nodes(50000)  # Same size as benchmark
    node_creation_time = time.time() - start
    print(f"Created {len(node_ids)} nodes in {node_creation_time:.4f}s")
    
    # Test bulk attribute setting like the benchmark does
    print("\n=== Step 2: Set attributes using bulk API ===")
    
    # Generate test data like the benchmark
    departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations']
    import random
    
    # Prepare attribute data like the benchmark
    bulk_attrs_dict = {}
    
    # Department attribute (text)
    dept_values = [random.choice(departments) for _ in range(len(node_ids))]
    bulk_attrs_dict['department'] = {
        'nodes': node_ids,
        'values': dept_values,
        'value_type': 'text'
    }
    
    # Salary attribute (int) 
    salary_values = [random.randint(40000, 200000) for _ in range(len(node_ids))]
    bulk_attrs_dict['salary'] = {
        'nodes': node_ids,
        'values': salary_values,
        'value_type': 'int'
    }
    
    # Performance attribute (float)
    perf_values = [random.uniform(1.0, 5.0) for _ in range(len(node_ids))]
    bulk_attrs_dict['performance'] = {
        'nodes': node_ids,
        'values': perf_values,
        'value_type': 'float'
    }
    
    # Active attribute (bool)
    active_values = [random.choice([True, False]) for _ in range(len(node_ids))]
    bulk_attrs_dict['active'] = {
        'nodes': node_ids,
        'values': active_values,
        'value_type': 'bool'
    }
    
    # Time the bulk attribute setting
    start = time.time()
    graph.set_node_attributes(bulk_attrs_dict)
    attr_time = time.time() - start
    
    print(f"Set attributes for {len(node_ids)} nodes in {attr_time:.4f}s")
    print(f"Total attributes set: {len(bulk_attrs_dict) * len(node_ids):,}")
    print(f"Rate: {len(bulk_attrs_dict) * len(node_ids) / attr_time:,.0f} attributes/sec")
    
    total_time = node_creation_time + attr_time
    print(f"\n=== Total Creation Time ===")
    print(f"Nodes: {node_creation_time:.4f}s")
    print(f"Attributes: {attr_time:.4f}s")
    print(f"Total: {total_time:.4f}s")
    print(f"Rate: {len(node_ids) / total_time:,.0f} nodes/sec (including attributes)")
    
    # Compare to individual attribute setting
    print("\n=== Comparison: Individual attribute setting ===")
    graph2 = gr.Graph()
    individual_nodes = graph2.add_nodes(1000)  # Smaller test
    
    start = time.time()
    for i, node_id in enumerate(individual_nodes):
        graph2.set_node_attribute(node_id, "department", gr.AttrValue(random.choice(departments)))
        graph2.set_node_attribute(node_id, "salary", gr.AttrValue(random.randint(40000, 200000)))
        graph2.set_node_attribute(node_id, "performance", gr.AttrValue(random.uniform(1.0, 5.0)))
        graph2.set_node_attribute(node_id, "active", gr.AttrValue(random.choice([True, False])))
    individual_time = time.time() - start
    
    print(f"Individual setting for {len(individual_nodes)} nodes: {individual_time:.4f}s")
    individual_rate = len(individual_nodes) * 4 / individual_time  # 4 attributes per node
    bulk_rate = len(bulk_attrs_dict) * len(node_ids) / attr_time
    
    print(f"Individual rate: {individual_rate:,.0f} attributes/sec")
    print(f"Bulk rate: {bulk_rate:,.0f} attributes/sec") 
    print(f"Bulk speedup: {bulk_rate / individual_rate:.1f}x faster")

if __name__ == "__main__":
    test_bulk_attribute_performance()