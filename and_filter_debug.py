#!/usr/bin/env python3
"""
Debug the AND filter logic specifically
"""

import time
import random
import groggy as gr

def create_debug_graph():
    """Create graph for AND filter debugging"""
    print("🔬 Creating debug graph (25K nodes)...")
    
    num_nodes = 25000
    graph = gr.Graph()
    bulk_node_ids = graph.add_nodes(num_nodes)
    
    departments = ['Engineering', 'Marketing', 'Sales']
    
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
    
    print(f"   Graph ready: {num_nodes} nodes")
    return graph

def test_manual_and_vs_builtin_and():
    """Compare manual AND logic vs built-in AND filter"""
    print("\n🔍 Comparing manual AND vs built-in AND...")
    
    graph = create_debug_graph()
    
    # Test 1: Manual AND - apply filters sequentially ourselves
    print("\n   Manual AND (applying filters sequentially in Python):")
    
    start = time.perf_counter()
    
    # Step 1: Get all nodes
    all_nodes = list(range(25000))  # We know node IDs are 0..24999
    print(f"      Starting with {len(all_nodes)} nodes")
    
    # Step 2: Apply first filter
    dept_filter = gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering"))
    result1 = graph.filter_nodes(dept_filter)
    step1_time = time.perf_counter()
    print(f"      After dept filter: {len(result1)} nodes ({step1_time - start:.6f}s)")
    
    # Step 3: Apply second filter to full graph (not intersecting yet)
    perf_filter = gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(gr.AttrValue(4.0)))
    result2 = graph.filter_nodes(perf_filter)
    step2_time = time.perf_counter()
    print(f"      After perf filter: {len(result2)} nodes ({step2_time - step1_time:.6f}s)")
    
    # Step 4: Apply third filter to full graph
    active_filter = gr.NodeFilter.attribute_equals("active", gr.AttrValue(True))
    result3 = graph.filter_nodes(active_filter)
    step3_time = time.perf_counter()
    print(f"      After active filter: {len(result3)} nodes ({step3_time - step2_time:.6f}s)")
    
    # Step 5: Check result types and try to convert to Python
    print(f"      Result types: {type(result1)}, {type(result2)}, {type(result3)}")
    
    # Try to get length and convert to list if possible
    try:
        list1 = list(result1) if hasattr(result1, '__iter__') else [f"Length: {len(result1)}"]
        list2 = list(result2) if hasattr(result2, '__iter__') else [f"Length: {len(result2)}"] 
        list3 = list(result3) if hasattr(result3, '__iter__') else [f"Length: {len(result3)}"]
        
        if all(isinstance(l, list) and len(l) > 0 and isinstance(l[0], int) for l in [list1, list2, list3]):
            set1 = set(list1)
            set2 = set(list2)
            set3 = set(list3)
            intersection = set1 & set2 & set3
            intersection_time = time.perf_counter()
            print(f"      Python intersection: {len(intersection)} nodes ({intersection_time - step3_time:.6f}s)")
        else:
            intersection_time = time.perf_counter()
            print(f"      Cannot intersect - results are handles: {list1[0]}, {list2[0]}, {list3[0]} ({intersection_time - step3_time:.6f}s)")
    except Exception as e:
        intersection_time = time.perf_counter()
        print(f"      Error processing results: {e} ({intersection_time - step3_time:.6f}s)")
    
    manual_total = intersection_time - start
    print(f"      Manual AND total: {manual_total:.6f}s")
    
    # Test 2: Built-in AND filter
    print(f"\n   Built-in AND filter:")
    
    start = time.perf_counter()
    filters = [dept_filter, perf_filter, active_filter]
    and_filter = gr.NodeFilter.and_filters(filters)
    builtin_result = graph.filter_nodes(and_filter)
    builtin_total = time.perf_counter() - start
    
    print(f"      Built-in AND result: {len(builtin_result)} nodes ({builtin_total:.6f}s)")
    
    # Compare
    print(f"\n   📊 Comparison:")
    print(f"      Manual AND: {manual_total:.6f}s")
    print(f"      Built-in AND: {builtin_total:.6f}s")
    print(f"      Speedup: {manual_total/builtin_total:.1f}x {'faster' if manual_total < builtin_total else 'slower'}")
    
    # Results comparison (approximated since we can't intersect handles)
    print(f"      🔍 Results comparison: builtin={len(builtin_result)} nodes")
    print(f"      Note: Manual intersection not computed (results are Rust handles)")

def test_and_filter_internals():
    """Test what happens inside the AND filter"""
    print("\n🔍 Testing AND filter with different orders...")
    
    graph = create_debug_graph()
    
    dept_filter = gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering"))
    perf_filter = gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(gr.AttrValue(4.0)))
    active_filter = gr.NodeFilter.attribute_equals("active", gr.AttrValue(True))
    
    # Test different filter orders
    orders = [
        ("Original", [dept_filter, perf_filter, active_filter]),
        ("Most selective first", [perf_filter, dept_filter, active_filter]),
        ("Least selective first", [active_filter, dept_filter, perf_filter])
    ]
    
    for name, filter_list in orders:
        start = time.perf_counter()
        and_filter = gr.NodeFilter.and_filters(filter_list)
        result = graph.filter_nodes(and_filter)
        elapsed = time.perf_counter() - start
        
        print(f"   {name:20s}: {elapsed:.6f}s ({len(result)} results)")

def test_two_vs_three_filters():
    """Test if the problem is with the number of filters"""
    print("\n🔍 Testing 2-filter vs 3-filter AND...")
    
    graph = create_debug_graph()
    
    dept_filter = gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering"))
    perf_filter = gr.NodeFilter.attribute_filter("performance", gr.AttributeFilter.greater_than(gr.AttrValue(4.0)))
    active_filter = gr.NodeFilter.attribute_equals("active", gr.AttrValue(True))
    
    # 2-filter AND
    start = time.perf_counter()
    two_filter = gr.NodeFilter.and_filters([dept_filter, perf_filter])
    result2 = graph.filter_nodes(two_filter)
    time2 = time.perf_counter() - start
    
    # 3-filter AND  
    start = time.perf_counter()
    three_filter = gr.NodeFilter.and_filters([dept_filter, perf_filter, active_filter])
    result3 = graph.filter_nodes(three_filter)
    time3 = time.perf_counter() - start
    
    print(f"   2-filter AND: {time2:.6f}s ({len(result2)} results)")
    print(f"   3-filter AND: {time3:.6f}s ({len(result3)} results)")
    print(f"   Time ratio: {time3/time2:.1f}x")

if __name__ == "__main__":
    print("🔬 AND FILTER DEBUG ANALYSIS")
    print("="*50)
    
    test_manual_and_vs_builtin_and()
    test_and_filter_internals()
    test_two_vs_three_filters()