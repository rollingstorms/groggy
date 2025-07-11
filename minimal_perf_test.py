#!/usr/bin/env python3
"""
Minimal performance test to isolate the issue.
"""

import time
from python.groggy import Graph

def test_single_vs_batch():
    """Compare single node addition vs batch."""
    
    # Test 1: Single node additions
    g1 = Graph()
    
    start = time.time()
    for i in range(1000):
        g1.add_node(f"node_{i}", attr=i)
    single_time = time.time() - start
    
    # Test 2: Batch node additions
    g2 = Graph()
    nodes = [{"id": f"node_{i}", "attr": i} for i in range(1000)]
    
    start = time.time()
    g2.add_nodes(nodes)
    batch_time = time.time() - start
    
    print(f"Single additions: {single_time*1e6/1000:.2f}μs per node")
    print(f"Batch additions:  {batch_time*1e6/1000:.2f}μs per node")
    print(f"Batch vs Single ratio: {batch_time/single_time:.2f}x")

def test_no_attributes():
    """Test without attributes to isolate attribute processing overhead."""
    
    g = Graph()
    nodes = [{"id": f"node_{i}"} for i in range(1000)]
    
    start = time.time()
    g.add_nodes(nodes)
    no_attrs_time = time.time() - start
    
    print(f"No attributes: {no_attrs_time*1e6/1000:.2f}μs per node")

def test_empty_attributes():
    """Test with empty attribute dictionaries."""
    
    g = Graph()
    nodes = [{"id": f"node_{i}"} for i in range(1000)]
    
    start = time.time()
    g.add_nodes(nodes)
    empty_attrs_time = time.time() - start
    
    print(f"Empty attributes: {empty_attrs_time*1e6/1000:.2f}μs per node")

if __name__ == "__main__":
    print("=== Minimal Performance Test ===\n")
    
    test_single_vs_batch()
    print()
    test_no_attributes()
    print()
    test_empty_attributes()
