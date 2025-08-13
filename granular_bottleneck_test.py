#!/usr/bin/env python3
"""
Extremely granular bottleneck analysis - isolate every single operation
"""

import time
import random
import groggy as gr

def time_operation(name, func):
    """Time a single operation with high precision"""
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    print(f"   {name}: {elapsed:.6f}s")
    return result, elapsed

def create_minimal_test_graph():
    """Create minimal graph for granular testing"""
    print("🔬 Creating minimal test graph...")
    
    # Small but meaningful size
    num_nodes = 10000
    
    graph = gr.Graph()
    
    # Time bulk node creation
    def create_nodes():
        return graph.add_nodes(num_nodes)
    
    bulk_node_ids, _ = time_operation("Bulk node creation", create_nodes)
    
    # Time bulk attribute setting for each attribute separately
    departments = ['Engineering', 'Marketing', 'Sales']
    
    def set_department_attr():
        values_list = [departments[i % len(departments)] for i in range(num_nodes)]
        bulk_attrs_dict = {
            'department': {
                'nodes': bulk_node_ids,
                'values': values_list,
                'value_type': 'text'
            }
        }
        graph.set_node_attributes(bulk_attrs_dict)
    
    def set_performance_attr():
        values_list = [random.uniform(1.0, 5.0) for _ in range(num_nodes)]
        bulk_attrs_dict = {
            'performance': {
                'nodes': bulk_node_ids,
                'values': values_list,
                'value_type': 'float'
            }
        }
        graph.set_node_attributes(bulk_attrs_dict)
    
    def set_active_attr():
        values_list = [random.choice([True, False]) for _ in range(num_nodes)]
        bulk_attrs_dict = {
            'active': {
                'nodes': bulk_node_ids,
                'values': values_list,
                'value_type': 'bool'
            }
        }
        graph.set_node_attributes(bulk_attrs_dict)
    
    time_operation("Department attribute setting", set_department_attr)
    time_operation("Performance attribute setting", set_performance_attr)
    time_operation("Active attribute setting", set_active_attr)
    
    print(f"   Graph ready: {num_nodes} nodes")
    return graph

def test_individual_filter_components(graph):
    """Test each filter component separately to find the bottleneck"""
    print("\n🔍 Testing individual filter components...")
    
    # Test 1: Single AttributeEquals filter
    def single_dept_filter():
        filter_obj = gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering"))
        return graph.filter_nodes(filter_obj)
    
    result1, time1 = time_operation("Single department filter", single_dept_filter)
    
    # Test 2: Single AttributeFilter (numeric)
    def single_perf_filter():
        filter_obj = gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(gr.AttrValue(4.0)))
        return graph.filter_nodes(filter_obj)
    
    result2, time2 = time_operation("Single performance filter", single_perf_filter)
    
    # Test 3: Single boolean filter
    def single_active_filter():
        filter_obj = gr.NodeFilter.attribute_equals("active", gr.AttrValue(True))
        return graph.filter_nodes(filter_obj)
    
    result3, time3 = time_operation("Single active filter", single_active_filter)
    
    print(f"   Results: dept={len(result1)}, perf={len(result2)}, active={len(result3)}")
    
    # Test 4: Two-way AND
    def two_way_and():
        filters = [
            gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering")),
            gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(gr.AttrValue(4.0)))
        ]
        and_filter = gr.NodeFilter.and_filters(filters)
        return graph.filter_nodes(and_filter)
    
    result4, time4 = time_operation("Two-way AND filter", two_way_and)
    
    # Test 5: Three-way AND (full benchmark)
    def three_way_and():
        filters = [
            gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering")),
            gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(gr.AttrValue(4.0))),
            gr.NodeFilter.attribute_equals("active", gr.AttrValue(True))
        ]
        and_filter = gr.NodeFilter.and_filters(filters)
        return graph.filter_nodes(and_filter)
    
    result5, time5 = time_operation("Three-way AND filter", three_way_and)
    
    print(f"   Results: 2-way={len(result4)}, 3-way={len(result5)}")
    
    # Test 6: Filter object creation overhead
    def just_create_filter():
        filters = [
            gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering")),
            gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(gr.AttrValue(4.0))),
            gr.NodeFilter.attribute_equals("active", gr.AttrValue(True))
        ]
        return gr.NodeFilter.and_filters(filters)
    
    filter_obj, time6 = time_operation("Filter object creation only", just_create_filter)
    
    # Test 7: Apply the pre-created filter
    def apply_precreated_filter():
        return graph.filter_nodes(filter_obj)
    
    result7, time7 = time_operation("Apply pre-created filter", apply_precreated_filter)
    
    print(f"   Pre-created filter result: {len(result7)}")
    
    return {
        'single_filters': [time1, time2, time3],
        'and_filters': [time4, time5],
        'filter_creation': time6,
        'filter_application': time7
    }

def test_filter_scaling(graph):
    """Test how filter performance scales with repetition"""
    print("\n⚡ Testing filter scaling and caching...")
    
    # Create the filter once
    filters = [
        gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering")),
        gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(gr.AttrValue(4.0))),
        gr.NodeFilter.attribute_equals("active", gr.AttrValue(True))
    ]
    and_filter = gr.NodeFilter.and_filters(filters)
    
    # Run the same filter multiple times
    times = []
    for i in range(5):
        def run_filter():
            return graph.filter_nodes(and_filter)
        
        result, elapsed = time_operation(f"Run {i+1}", run_filter)
        times.append(elapsed)
        if i == 0:
            print(f"       First run result: {len(result)} nodes")
    
    print(f"   Average time: {sum(times)/len(times):.6f}s")
    print(f"   Time variation: {max(times) - min(times):.6f}s")
    
    return times

def analyze_python_overhead():
    """Test pure Python vs Rust operations"""
    print("\n🐍 Analyzing Python/PyO3 overhead...")
    
    # Test AttrValue creation overhead
    def create_attr_values():
        return [gr.AttrValue("Engineering"), gr.AttrValue(4.0), gr.AttrValue(True)]
    
    attrs, time1 = time_operation("AttrValue creation", create_attr_values)
    
    # Test filter object creation
    def create_filters():
        return [
            gr.NodeFilter.attribute_equals("department", attrs[0]),
            gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(attrs[1])),
            gr.NodeFilter.attribute_equals("active", attrs[2])
        ]
    
    filters, time2 = time_operation("Filter object creation", create_filters)
    
    # Test AND filter creation
    def create_and_filter():
        return gr.NodeFilter.and_filters(filters)
    
    and_filter, time3 = time_operation("AND filter creation", create_and_filter)
    
    return [time1, time2, time3]

def main():
    print("🔬 GRANULAR BOTTLENECK ANALYSIS")
    print("="*50)
    
    # Create test graph
    graph = create_minimal_test_graph()
    
    # Test filter components
    filter_times = test_individual_filter_components(graph)
    
    # Test scaling
    scaling_times = test_filter_scaling(graph)
    
    # Test Python overhead
    python_times = analyze_python_overhead()
    
    print("\n📊 BOTTLENECK ANALYSIS")
    print("="*50)
    
    total_single = sum(filter_times['single_filters'])
    print(f"Total single filter time: {total_single:.6f}s")
    print(f"Three-way AND time: {filter_times['and_filters'][1]:.6f}s")
    print(f"Filter creation overhead: {filter_times['filter_creation']:.6f}s")
    print(f"Python/PyO3 overhead: {sum(python_times):.6f}s")
    
    # Calculate where the time is going
    if filter_times['and_filters'][1] > total_single * 1.5:
        print("🚨 BOTTLENECK: AND logic is slower than sum of individual filters!")
    elif filter_times['filter_creation'] > filter_times['filter_application'] * 0.5:
        print("🚨 BOTTLENECK: Filter object creation overhead!")
    elif sum(python_times) > filter_times['filter_application'] * 0.3:
        print("🚨 BOTTLENECK: Python/PyO3 conversion overhead!")
    else:
        print("✅ Performance profile looks normal - bottleneck is likely in Rust filtering logic")

if __name__ == "__main__":
    main()