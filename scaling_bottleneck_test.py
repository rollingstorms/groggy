#!/usr/bin/env python3
"""
Test scaling behavior to find where performance degrades
"""

import time
import random
import groggy as gr

def create_test_graph(num_nodes):
    """Create test graph of specified size with bulk operations"""
    graph = gr.Graph()
    
    # Bulk node creation
    start = time.perf_counter()
    bulk_node_ids = graph.add_nodes(num_nodes)
    node_creation_time = time.perf_counter() - start
    
    # Bulk attribute setting
    departments = ['Engineering', 'Marketing', 'Sales']
    
    start = time.perf_counter()
    # Set all attributes in one operation
    bulk_attrs_dict = {
        'department': {
            'nodes': bulk_node_ids,
            'values': [departments[i % len(departments)] for i in range(num_nodes)],
            'value_type': 'text'
        },
        'performance': {
            'nodes': bulk_node_ids,
            'values': [random.uniform(1.0, 5.0) for _ in range(num_nodes)],
            'value_type': 'float'
        },
        'active': {
            'nodes': bulk_node_ids,
            'values': [random.choice([True, False]) for _ in range(num_nodes)],
            'value_type': 'bool'
        }
    }
    graph.set_node_attributes(bulk_attrs_dict)
    attr_creation_time = time.perf_counter() - start
    
    print(f"   {num_nodes:6d} nodes: creation={node_creation_time:.6f}s, attrs={attr_creation_time:.6f}s")
    return graph

def test_filter_scaling():
    """Test filter performance at different scales"""
    print("📈 Testing filter performance scaling...")
    
    sizes = [1000, 5000, 10000, 25000, 50000]
    results = []
    
    for size in sizes:
        print(f"\n🔬 Testing {size} nodes:")
        graph = create_test_graph(size)
        
        # Create the filter
        filters = [
            gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering")),
            gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(gr.AttrValue(4.0))),
            gr.NodeFilter.attribute_equals("active", gr.AttrValue(True))
        ]
        and_filter = gr.NodeFilter.and_filters(filters)
        
        # Time the filter operation multiple times for accuracy
        times = []
        for i in range(3):
            start = time.perf_counter()
            result = graph.filter_nodes(and_filter)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            if i == 0:
                result_count = len(result)
        
        avg_time = sum(times) / len(times)
        throughput = size / avg_time
        
        print(f"   Filter time: {avg_time:.6f}s ({result_count} results)")
        print(f"   Throughput: {throughput:.0f} nodes/sec")
        
        results.append({
            'size': size,
            'time': avg_time,
            'throughput': throughput,
            'results': result_count
        })
    
    print("\n📊 SCALING ANALYSIS")
    print("="*60)
    print("Size      Time(s)    Throughput    Results    Efficiency")
    print("-"*60)
    
    baseline_efficiency = None
    for r in results:
        efficiency = r['throughput'] / r['size']  # throughput per node
        if baseline_efficiency is None:
            baseline_efficiency = efficiency
        
        relative_efficiency = efficiency / baseline_efficiency
        
        print(f"{r['size']:6d}    {r['time']:.6f}    {r['throughput']:9.0f}    {r['results']:7d}    {relative_efficiency:.2f}x")
    
    # Check for quadratic scaling
    print(f"\n🔍 Scaling analysis:")
    if len(results) >= 2:
        small = results[0]
        large = results[-1]
        
        size_ratio = large['size'] / small['size']
        time_ratio = large['time'] / small['time']
        
        print(f"   Size increased {size_ratio:.1f}x: {small['size']} → {large['size']}")
        print(f"   Time increased {time_ratio:.1f}x: {small['time']:.6f}s → {large['time']:.6f}s")
        
        if time_ratio > size_ratio * 1.5:
            print(f"   🚨 WORSE than linear scaling! (should be ~{size_ratio:.1f}x)")
            if time_ratio > size_ratio * size_ratio * 0.5:
                print(f"   🚨 Approaching QUADRATIC scaling!")
        elif time_ratio > size_ratio * 1.1:
            print(f"   ⚠️  Slightly worse than linear scaling")
        else:
            print(f"   ✅ Good linear scaling")

def test_component_scaling():
    """Test scaling of individual filter components"""
    print("\n🔍 Testing individual component scaling at 50K nodes...")
    
    graph = create_test_graph(50000)
    
    # Test each filter component individually
    components = [
        ("Department filter", gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering"))),
        ("Performance filter", gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(gr.AttrValue(4.0)))),
        ("Active filter", gr.NodeFilter.attribute_equals("active", gr.AttrValue(True)))
    ]
    
    print(f"   Component scaling at 50K nodes:")
    for name, filter_obj in components:
        start = time.perf_counter()
        result = graph.filter_nodes(filter_obj)
        elapsed = time.perf_counter() - start
        throughput = 50000 / elapsed
        
        print(f"   {name:20s}: {elapsed:.6f}s ({len(result):5d} results, {throughput:.0f} nodes/sec)")

if __name__ == "__main__":
    print("🔬 SCALING BOTTLENECK ANALYSIS")
    print("="*50)
    
    test_filter_scaling()
    test_component_scaling()