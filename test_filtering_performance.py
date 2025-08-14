#!/usr/bin/env python3
"""
Focused performance test for node vs edge filtering

This investigates the performance issue where node filtering is much slower than edge filtering.
"""

import sys
sys.path.insert(0, 'python-groggy/python')

import time
import groggy as gr

def create_test_graph(num_nodes=1000, num_edges=2000):
    """Create a test graph with attributes for performance testing."""
    print(f"Creating test graph: {num_nodes} nodes, {num_edges} edges...")
    
    g = gr.Graph()
    
    # Add nodes with numeric attributes
    nodes = []
    for i in range(num_nodes):
        node_id = g.add_node(
            index=i,
            salary=50000 + (i * 100),  # Spread salaries from 50k to 150k
            age=25 + (i % 40),         # Ages from 25 to 65
            dept_id=i % 10             # 10 different departments
        )
        nodes.append(node_id)
    
    # Add edges with numeric attributes
    import random
    random.seed(42)
    for _ in range(num_edges):
        source = random.choice(nodes)
        target = random.choice(nodes)
        if source != target:
            g.add_edge(source, target, 
                      weight=random.uniform(0.1, 1.0),
                      strength=random.randint(1, 10))
    
    print(f"✅ Created graph: {g.node_count()} nodes, {g.edge_count()} edges")
    return g, nodes

def benchmark_node_filtering(g, iterations=10):
    """Benchmark node filtering performance."""
    print(f"\n🔍 Benchmarking Node Filtering ({iterations} iterations)")
    
    # Test 1: Simple numeric filter
    print("Test 1: Simple numeric filter (salary > 75000)")
    times = []
    for i in range(iterations):
        start_time = time.perf_counter()
        
        # Create filter and apply
        filter_obj = gr.parse_node_query("salary > 75000")
        result = g.filter_nodes(filter_obj)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    per_node_time = (avg_time * 1_000_000) / g.node_count()  # microseconds per node
    print(f"   Average time: {avg_time:.6f}s")
    print(f"   Per-node time: {per_node_time:.2f}μs")
    print(f"   Results found: {len(result.nodes)}")
    
    # Test 2: Range filter
    print("\nTest 2: Range filter (age >= 30 AND age <= 50)")
    times = []
    for i in range(iterations):
        start_time = time.perf_counter()
        
        # Create filter and apply
        filter_obj = gr.parse_node_query("age >= 30")
        result = g.filter_nodes(filter_obj)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    per_node_time = (avg_time * 1_000_000) / g.node_count()  # microseconds per node
    print(f"   Average time: {avg_time:.6f}s")
    print(f"   Per-node time: {per_node_time:.2f}μs")
    print(f"   Results found: {len(result.nodes)}")
    
    return avg_time

def benchmark_edge_filtering(g, iterations=10):
    """Benchmark edge filtering performance."""
    print(f"\n🔗 Benchmarking Edge Filtering ({iterations} iterations)")
    
    # Test 1: Simple numeric filter
    print("Test 1: Simple numeric filter (weight > 0.5)")
    times = []
    for i in range(iterations):
        start_time = time.perf_counter()
        
        # Create filter and apply
        filter_obj = gr.parse_edge_query("weight > 0.5")
        result = g.filter_edges(filter_obj)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    per_edge_time = (avg_time * 1_000_000) / g.edge_count()  # microseconds per edge
    print(f"   Average time: {avg_time:.6f}s")
    print(f"   Per-edge time: {per_edge_time:.2f}μs")
    print(f"   Results found: {len(result)}")
    
    # Test 2: Range filter  
    print("\nTest 2: Range filter (strength >= 5)")
    times = []
    for i in range(iterations):
        start_time = time.perf_counter()
        
        # Create filter and apply
        filter_obj = gr.parse_edge_query("strength >= 5")
        result = g.filter_edges(filter_obj)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    per_edge_time = (avg_time * 1_000_000) / g.edge_count()  # microseconds per edge
    print(f"   Average time: {avg_time:.6f}s")
    print(f"   Per-edge time: {per_edge_time:.2f}μs")
    print(f"   Results found: {len(result)}")
    
    return avg_time

def benchmark_direct_access(g, iterations=10):
    """Benchmark direct node access for comparison."""
    print(f"\n⚡ Benchmarking Direct Node Access ({iterations} iterations)")
    
    times = []
    for i in range(iterations):
        start_time = time.perf_counter()
        
        # Access all node attributes directly
        count = 0
        for node_id in g.node_ids:
            node_view = g.nodes[node_id]
            salary = node_view['salary']
            # Handle AttrValue - extract actual value
            if hasattr(salary, 'value'):
                salary_value = salary.value
            else:
                salary_value = salary
            if salary_value > 75000:  # Same condition as filtering
                count += 1
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    per_node_time = (avg_time * 1_000_000) / g.node_count()  # microseconds per node
    print(f"   Average time: {avg_time:.6f}s")
    print(f"   Per-node time: {per_node_time:.2f}μs")
    print(f"   Results found: {count}")
    
    return avg_time

def test_scaling_behavior():
    """Test how performance scales with graph size."""
    print(f"\n📊 Testing Scaling Behavior")
    
    sizes = [(500, 1000), (1000, 2000), (2000, 4000)]
    
    for num_nodes, num_edges in sizes:
        print(f"\n--- Scale: {num_nodes} nodes, {num_edges} edges ---")
        g, _ = create_test_graph(num_nodes, num_edges)
        
        # Quick single test
        node_time = benchmark_node_filtering(g, iterations=3)
        edge_time = benchmark_edge_filtering(g, iterations=3)
        direct_time = benchmark_direct_access(g, iterations=3)
        
        print(f"   Summary:")
        print(f"   • Node filtering: {node_time:.6f}s")
        print(f"   • Edge filtering: {edge_time:.6f}s") 
        print(f"   • Direct access: {direct_time:.6f}s")
        print(f"   • Node/Edge ratio: {node_time/edge_time:.2f}x")
        print(f"   • Filtering/Direct ratio: {node_time/direct_time:.2f}x")

if __name__ == "__main__":
    print("🚀 Groggy Filtering Performance Investigation")
    print("=" * 60)
    
    # Create test graph
    g, nodes = create_test_graph(1000, 2000)
    
    # Run detailed benchmarks
    node_time = benchmark_node_filtering(g)
    edge_time = benchmark_edge_filtering(g)
    direct_time = benchmark_direct_access(g)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📈 Performance Summary")
    print(f"{'=' * 60}")
    print(f"Node filtering time: {node_time:.6f}s")
    print(f"Edge filtering time: {edge_time:.6f}s")
    print(f"Direct access time: {direct_time:.6f}s")
    print(f"")
    print(f"Node vs Edge filtering: {node_time/edge_time:.2f}x slower")
    print(f"Filtering vs Direct access: {node_time/direct_time:.2f}x slower")
    
    if node_time/edge_time > 2.0:
        print(f"")
        print(f"⚠️  PERFORMANCE ISSUE DETECTED:")
        print(f"   Node filtering is {node_time/edge_time:.2f}x slower than edge filtering")
        print(f"   This suggests an algorithmic issue in node filtering implementation")
    
    # Test scaling
    test_scaling_behavior()