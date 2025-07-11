#!/usr/bin/env python3
"""
Profile attribute processing specifically.
"""

import time
from python.groggy import Graph

def test_attribute_overhead():
    """Test the cost of different numbers of attributes."""
    
    batch_size = 1000
    
    for n_attrs in [0, 1, 3, 5, 10]:
        g = Graph()
        
        # Create nodes with varying numbers of attributes
        if n_attrs == 0:
            nodes = [{"id": f"node_{i}"} for i in range(batch_size)]
        else:
            nodes = []
            for i in range(batch_size):
                node = {"id": f"node_{i}"}
                for j in range(n_attrs):
                    node[f"attr_{j}"] = f"value_{i}_{j}"
                nodes.append(node)
        
        # Time the batch addition
        start = time.time()
        g.add_nodes(nodes)
        elapsed = time.time() - start
        
        print(f"{n_attrs:2d} attrs: {elapsed*1e6/batch_size:.2f}μs per node")

def test_attribute_types():
    """Test the cost of different attribute types."""
    
    batch_size = 1000
    
    test_cases = [
        ("strings", lambda i: f"value_{i}"),
        ("integers", lambda i: i),
        ("floats", lambda i: float(i) + 0.5),
        ("booleans", lambda i: i % 2 == 0),
        ("lists", lambda i: [i, i+1, i+2]),
        ("dicts", lambda i: {"nested": i}),
    ]
    
    for name, value_func in test_cases:
        g = Graph()
        nodes = [{"id": f"node_{i}", "attr": value_func(i)} for i in range(batch_size)]
        
        start = time.time()
        g.add_nodes(nodes)
        elapsed = time.time() - start
        
        print(f"{name:8s}: {elapsed*1e6/batch_size:.2f}μs per node")

if __name__ == "__main__":
    print("=== Attribute Processing Performance ===\n")
    
    print("Attribute count overhead:")
    test_attribute_overhead()
    
    print("\nAttribute type overhead:")
    test_attribute_types()
