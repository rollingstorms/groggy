#!/usr/bin/env python3
"""
Test the optimized node filtering performance
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-groggy/python'))

import groggy as gr
import time
import random

def create_test_graph(n_nodes=5000, n_edges=10000):
    """Create a test graph with the same structure as benchmark"""
    g = gr.Graph()
    
    # Create nodes with attributes similar to benchmark
    print(f"Creating {n_nodes} nodes...")
    for i in range(n_nodes):
        dept = random.choice(['Engineering', 'Sales', 'Marketing', 'Design', 'HR'])
        salary = random.randint(60000, 200000)
        performance = round(random.uniform(1.0, 5.0), 1)
        active = random.choice([True, False])
        
        g.add_node(
            department=dept,
            salary=salary, 
            performance=performance,
            active=active
        )
    
    # Create edges
    print(f"Creating {n_edges} edges...")
    for _ in range(n_edges):
        source = random.randint(0, n_nodes - 1)
        target = random.randint(0, n_nodes - 1)
        if source != target:
            relationship = random.choice(['reports_to', 'collaborates', 'mentored_by'])
            weight = round(random.uniform(0.1, 1.0), 1)
            
            g.add_edge(source, target, relationship=relationship, weight=weight)
    
    print(f"Graph created: {len(g.nodes)} nodes, {len(g.edges)} edges")
    return g

def test_node_filtering_performance(g):
    """Test different types of node filtering"""
    print("\n=== Testing Node Filtering Performance ===")
    
    # Test 1: Simple attribute equals
    print("\n1. Filter by department (Engineering):")
    start = time.time()
    engineers = g.filter_nodes('department == "Engineering"')
    duration = time.time() - start
    print(f"   Time: {duration:.4f}s, Results: {len(engineers.nodes)}")
    
    # Test 2: Numeric range filter
    print("\n2. Filter by salary (> 120000):")
    start = time.time() 
    high_earners = g.filter_nodes('salary > 120000')
    duration = time.time() - start
    print(f"   Time: {duration:.4f}s, Results: {len(high_earners.nodes)}")
    
    # Test 3: Boolean filter
    print("\n3. Filter by active status:")
    start = time.time()
    active_employees = g.filter_nodes('active == True')  
    duration = time.time() - start
    print(f"   Time: {duration:.4f}s, Results: {len(active_employees.nodes)}")
    
    # Test 4: Performance comparison filter
    print("\n4. Filter by performance (> 4.0):")
    start = time.time()
    high_performers = g.filter_nodes('performance > 4.0')
    duration = time.time() - start
    print(f"   Time: {duration:.4f}s, Results: {len(high_performers.nodes)}")

def test_edge_filtering_performance(g):
    """Test edge filtering for comparison"""
    print("\n=== Testing Edge Filtering Performance ===")
    
    # Test 1: Filter by relationship
    print("\n1. Filter by relationship (reports_to):")
    start = time.time()
    reports = g.filter_edges('relationship == "reports_to"')
    duration = time.time() - start
    print(f"   Time: {duration:.4f}s, Results: {len(reports)}")
    
    # Test 2: Filter by weight
    print("\n2. Filter by weight (> 0.8):")
    start = time.time()
    strong_connections = g.filter_edges('weight > 0.8')
    duration = time.time() - start
    print(f"   Time: {duration:.4f}s, Results: {len(strong_connections)}")

if __name__ == "__main__":
    print("=== Node Filtering Optimization Test ===")
    
    # Create test graph
    g = create_test_graph(n_nodes=5000, n_edges=10000)
    
    # Test node filtering performance
    test_node_filtering_performance(g)
    
    # Test edge filtering for comparison
    test_edge_filtering_performance(g)
    
    print("\n✅ Node filtering optimization test completed!")
