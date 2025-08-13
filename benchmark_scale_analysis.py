#!/usr/bin/env python3
"""
Benchmark Scale Analysis - Compare Simple vs Complex Filtering at Different Scales
==============================================================================

This script tests whether the filtering performance degrades with scale
and compares simple vs complex filtering patterns used in the benchmark.
"""

import time
import random
import statistics
import groggy as gr

def create_benchmark_style_graph(num_nodes=10000):
    """Create graph with same attributes as the benchmark"""
    print(f"📊 Creating benchmark-style graph: {num_nodes} nodes")
    
    start = time.time()
    graph = gr.Graph()
    
    # Add nodes
    nodes = graph.add_nodes(num_nodes)
    
    # Add attributes with same distribution as benchmark
    departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations']
    roles = ['junior', 'mid', 'senior', 'principal', 'manager', 'director']
    locations = ['NYC', 'SF', 'LA', 'Chicago', 'Austin', 'Remote']
    
    print("   Setting attributes individually...")
    attr_start = time.time()
    
    # Set attributes individually to avoid bulk API issues
    for i, node_id in enumerate(nodes):
        graph.set_node_attribute(node_id, "department", gr.AttrValue(departments[i % len(departments)]))
        graph.set_node_attribute(node_id, "role", gr.AttrValue(roles[i % len(roles)]))
        graph.set_node_attribute(node_id, "location", gr.AttrValue(locations[i % len(locations)]))
        graph.set_node_attribute(node_id, "salary", gr.AttrValue(random.randint(40000, 200000)))
        graph.set_node_attribute(node_id, "age", gr.AttrValue(random.randint(22, 65)))
        graph.set_node_attribute(node_id, "performance", gr.AttrValue(random.uniform(1.0, 5.0)))
        graph.set_node_attribute(node_id, "active", gr.AttrValue(random.choice([True, False])))
    attr_time = time.time() - attr_start
    print(f"   Attributes set in: {attr_time:.3f}s")
    
    total_time = time.time() - start
    print(f"   Graph created in: {total_time:.3f}s")
    return graph, nodes

def test_simple_vs_complex_filtering(graph, scale_name, runs=3):
    """Test simple vs complex filtering patterns"""
    print(f"\n🔍 FILTERING SCALE TEST: {scale_name}")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Simple single attribute (like granular test)
    print("📋 Test 1: Simple Single Attribute (department='Engineering')")
    times = []
    for run in range(runs):
        start = time.time()
        simple_filter = gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering"))
        result = graph.filter_nodes(simple_filter)
        times.append(time.time() - start)
        if run == 0:
            print(f"   First run: {times[0]:.4f}s ({len(result)} results)")
    
    results['simple'] = statistics.mean(times)
    print(f"   Average: {results['simple']:.4f}s")
    
    # Test 2: Complex AND (exactly like benchmark)
    print("\\n📋 Test 2: Complex AND (department='Engineering' AND performance>4.0 AND active=True)")
    times = []
    for run in range(runs):
        start = time.time()
        filters = [
            gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering")),
            gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(gr.AttrValue(4.0))),
            gr.NodeFilter.attribute_equals("active", gr.AttrValue(True))
        ]
        complex_filter = gr.NodeFilter.and_filters(filters)
        result = graph.filter_nodes(complex_filter)
        times.append(time.time() - start)
        if run == 0:
            print(f"   First run: {times[0]:.4f}s ({len(result)} results)")
    
    results['complex_and'] = statistics.mean(times)
    print(f"   Average: {results['complex_and']:.4f}s")
    
    # Test 3: Complex OR (like benchmark)
    print("\\n📋 Test 3: Complex OR (salary>150000 OR performance>4.5)")
    times = []
    for run in range(runs):
        start = time.time()
        filters = [
            gr.NodeFilter.attribute_filter("salary", gr.AttributeFilter.greater_than(gr.AttrValue(150000))),
            gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(gr.AttrValue(4.5)))
        ]
        complex_filter = gr.NodeFilter.or_filters(filters)
        result = graph.filter_nodes(complex_filter)
        times.append(time.time() - start)
        if run == 0:
            print(f"   First run: {times[0]:.4f}s ({len(result)} results)")
    
    results['complex_or'] = statistics.mean(times)
    print(f"   Average: {results['complex_or']:.4f}s")
    
    # Test 4: Numeric range (benchmark style)
    print("\\n📋 Test 4: Numeric Range (salary > 120000)")
    times = []
    for run in range(runs):
        start = time.time()
        numeric_filter = gr.NodeFilter.attribute_filter("salary", gr.AttributeFilter.greater_than(gr.AttrValue(120000)))
        result = graph.filter_nodes(numeric_filter)
        times.append(time.time() - start)
        if run == 0:
            print(f"   First run: {times[0]:.4f}s ({len(result)} results)")
    
    results['numeric'] = statistics.mean(times)
    print(f"   Average: {results['numeric']:.4f}s")
    
    return results

def scaling_analysis():
    """Test filtering performance at different scales"""
    print("🚀 FILTERING PERFORMANCE SCALING ANALYSIS")
    print("=" * 80)
    print("Testing simple vs complex filters at different scales to identify bottlenecks\\n")
    
    # Test different scales - check if Complex AND scaling gets worse
    scales = [
        (1000, "1K nodes"),
        (10000, "10K nodes"),
        (25000, "25K nodes")
    ]
    
    all_results = []
    
    for num_nodes, scale_name in scales:
        print(f"\\n{'='*80}")
        print(f"SCALE: {scale_name}")
        print('='*80)
        
        # Create graph
        graph, nodes = create_benchmark_style_graph(num_nodes)
        
        # Test filtering
        results = test_simple_vs_complex_filtering(graph, scale_name)
        
        # Calculate throughput
        throughputs = {}
        for test_name, time_val in results.items():
            throughputs[test_name] = num_nodes / time_val if time_val > 0 else 0
        
        # Store results
        all_results.append({
            'scale': num_nodes,
            'scale_name': scale_name,
            'times': results,
            'throughputs': throughputs
        })
        
        # Print summary for this scale
        print(f"\\n📊 SCALE SUMMARY: {scale_name}")
        print(f"{'Test':<20} {'Time (s)':<10} {'Throughput (nodes/s)':<20}")
        print("-" * 50)
        for test_name, time_val in results.items():
            throughput = throughputs[test_name]
            print(f"{test_name:<20} {time_val:<10.4f} {throughput:<20.0f}")
    
    # Final analysis
    print(f"\\n🎯 SCALING ANALYSIS SUMMARY")
    print("=" * 80)
    print("Performance degradation with scale:\\n")
    
    print(f"{'Scale':<15} {'Simple':<10} {'Complex AND':<15} {'Complex OR':<15} {'Numeric':<10}")
    print("-" * 70)
    
    for result in all_results:
        scale_name = result['scale_name']
        times = result['times']
        print(f"{scale_name:<15} {times['simple']:<10.4f} {times['complex_and']:<15.4f} {times['complex_or']:<15.4f} {times['numeric']:<10.4f}")
    
    # Compare complexity overhead at different scales
    print(f"\\n📈 COMPLEXITY OVERHEAD ANALYSIS")
    print("=" * 50)
    print("How much slower are complex filters vs simple filters?\\n")
    
    print(f"{'Scale':<15} {'AND Overhead':<15} {'OR Overhead':<15}")
    print("-" * 45)
    
    for result in all_results:
        scale_name = result['scale_name']
        times = result['times']
        and_overhead = times['complex_and'] / times['simple'] if times['simple'] > 0 else 0
        or_overhead = times['complex_or'] / times['simple'] if times['simple'] > 0 else 0
        print(f"{scale_name:<15} {and_overhead:<15.1f}x {or_overhead:<15.1f}x")
    
    # Identify the bottleneck
    print(f"\\n🔍 BOTTLENECK IDENTIFICATION")
    print("=" * 40)
    
    # Check if performance degrades linearly or worse with scale
    small_scale = all_results[0]  # 1K nodes
    large_scale = all_results[-1]  # 100K nodes
    
    scale_factor = large_scale['scale'] / small_scale['scale']  # 100x
    
    for test_name in ['simple', 'complex_and', 'complex_or', 'numeric']:
        small_time = small_scale['times'][test_name]
        large_time = large_scale['times'][test_name]
        
        time_factor = large_time / small_time if small_time > 0 else 0
        efficiency = scale_factor / time_factor if time_factor > 0 else 0
        
        print(f"{test_name}:")
        print(f"  Scale factor: {scale_factor:.0f}x nodes")
        print(f"  Time factor: {time_factor:.1f}x slower")
        print(f"  Efficiency: {efficiency:.2f} (1.0 = linear scaling)")
        
        if efficiency < 0.5:
            print(f"  🚨 POOR SCALING - Much worse than linear")
        elif efficiency < 0.8:
            print(f"  ⚠️  SUBLINEAR - Some inefficiency")
        else:
            print(f"  ✅ GOOD SCALING - Near linear")
        print()

if __name__ == "__main__":
    scaling_analysis()