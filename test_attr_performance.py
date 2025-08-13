#!/usr/bin/env python3
"""
Test attribute setting performance specifically
"""

import time
import groggy as gr

def test_attribute_performance():
    """Test individual attribute operations for bottlenecks"""
    
    print("Testing attribute performance...")
    
    # Test 1: Single attribute setting (what filtering depends on)
    print("\n=== Test 1: Single attribute setting ===")
    graph = gr.Graph()
    nodes = graph.add_nodes(1000)
    
    start = time.time()
    for i, node_id in enumerate(nodes):
        graph.set_node_attribute(node_id, "value", gr.AttrValue(i))
    single_attr_time = time.time() - start
    
    print(f"Set {len(nodes)} single attributes in {single_attr_time:.4f}s")
    print(f"Rate: {len(nodes)/single_attr_time:,.0f} attributes/sec")
    
    # Test 2: Attribute getting (what filtering uses internally)
    print("\n=== Test 2: Single attribute getting ===")
    start = time.time()
    values = []
    for node_id in nodes:
        attr = graph.get_node_attribute(node_id, "value")
        if attr:
            values.append(attr.value)
    get_attr_time = time.time() - start
    
    print(f"Got {len(values)} attributes in {get_attr_time:.4f}s")
    print(f"Rate: {len(values)/get_attr_time:,.0f} attributes/sec")
    
    # Test 3: AttrValue creation (used everywhere)
    print("\n=== Test 3: AttrValue creation ===")
    start = time.time()
    attr_values = []
    for i in range(10000):
        attr_values.append(gr.AttrValue(i))
    attr_creation_time = time.time() - start
    
    print(f"Created {len(attr_values)} AttrValues in {attr_creation_time:.4f}s")
    print(f"Rate: {len(attr_values)/attr_creation_time:,.0f} AttrValues/sec")
    
    # Test 4: Simple node creation WITHOUT attributes
    print("\n=== Test 4: Node creation without attributes ===")
    graph2 = gr.Graph()
    start = time.time()
    nodes2 = graph2.add_nodes(10000)
    clean_node_time = time.time() - start
    
    print(f"Created {len(nodes2)} nodes (no attrs) in {clean_node_time:.4f}s")
    print(f"Rate: {len(nodes2)/clean_node_time:,.0f} nodes/sec")
    
    # Test 5: Node creation WITH attributes using add_node(**kwargs)
    print("\n=== Test 5: Node creation with kwargs ===")
    graph3 = gr.Graph()
    start = time.time()
    nodes3 = []
    for i in range(1000):
        nodes3.append(graph3.add_node(value=i))
    kwargs_node_time = time.time() - start
    
    print(f"Created {len(nodes3)} nodes (with kwargs) in {kwargs_node_time:.4f}s")
    print(f"Rate: {len(nodes3)/kwargs_node_time:,.0f} nodes/sec")
    
    print(f"\n=== Performance Comparison ===")
    print(f"Clean nodes: {len(nodes2)/clean_node_time:,.0f} nodes/sec")
    print(f"Kwargs nodes: {len(nodes3)/kwargs_node_time:,.0f} nodes/sec")
    print(f"Single attr set: {len(nodes)/single_attr_time:,.0f} attrs/sec")
    print(f"Single attr get: {len(values)/get_attr_time:,.0f} attrs/sec")
    print(f"AttrValue creation: {len(attr_values)/attr_creation_time:,.0f} AttrValues/sec")
    
    # Ratios
    overhead_ratio = (len(nodes2)/clean_node_time) / (len(nodes3)/kwargs_node_time)
    print(f"\nKwargs overhead: {overhead_ratio:.1f}x slower than clean nodes")

if __name__ == "__main__":
    test_attribute_performance()