#!/usr/bin/env python3
"""
Compare optimized performance
"""

import time
import groggy as gr

def time_operation(func, name):
    """Time a function and return the result"""
    start = time.time()
    result = func()
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.4f}s")
    return result, elapsed

def test_optimized_performance():
    """Test the key performance improvements"""
    
    print("=== Testing optimized performance ===")
    
    def create_nodes():
        g = gr.Graph()
        return g.add_nodes(50000)
    
    def create_edges():
        g = gr.Graph()
        nodes = g.add_nodes(10000)
        edges = [(i, (i+1) % len(nodes)) for i in range(5000)]
        return g.add_edges(edges)
    
    def create_node_with_attrs():
        g = gr.Graph()
        results = []
        for i in range(1000):
            results.append(g.add_node(id=i, name=f"node_{i}", value=i*2.5))
        return results
    
    def filter_nodes():
        g = gr.Graph()
        nodes = g.add_nodes(10000)
        # Set some attributes
        for i, node_id in enumerate(nodes[:1000]):
            g.set_node_attribute(node_id, "value", gr.AttrValue(i))
        
        # Create filter
        filter_obj = gr.NodeFilter.attribute_filter("value", 
                                                   gr.AttributeFilter.greater_than(gr.AttrValue(500)))
        return g.filter_nodes(filter_obj)
    
    # Run tests
    nodes, nodes_time = time_operation(create_nodes, "Bulk node creation (50K)")
    edges, edges_time = time_operation(create_edges, "Bulk edge creation (5K)")
    attrs, attrs_time = time_operation(create_node_with_attrs, "Node creation with attrs (1K)")
    filtered, filter_time = time_operation(filter_nodes, "Node filtering")
    
    # Calculate rates
    print(f"\n=== Performance Summary ===")
    print(f"Node creation rate: {50000/nodes_time:,.0f} nodes/sec")
    print(f"Edge creation rate: {5000/edges_time:,.0f} edges/sec") 
    print(f"Attr node rate: {1000/attrs_time:,.0f} nodes/sec")
    print(f"Filter performance: {filter_time:.4f}s for 1K nodes")

if __name__ == "__main__":
    test_optimized_performance()