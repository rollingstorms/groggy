#!/usr/bin/env python3
"""
Quick scaling test to find the performance crossover point
"""
import time
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import groggy
import networkx as nx

def quick_test(num_nodes, num_attrs):
    # Generate node data
    nodes_groggy = []
    nodes_nx = []
    
    for i in range(num_nodes):
        attrs = {f'attr_{j}': random.random() for j in range(num_attrs)}
        
        # Groggy format
        node_dict = {'id': f'node_{i}'}
        node_dict.update(attrs)
        nodes_groggy.append(node_dict)
        
        # NetworkX format
        nodes_nx.append((f'node_{i}', attrs))
    
    # Test Groggy
    start = time.time()
    g = groggy.Graph(backend='rust')
    g.add_nodes(nodes_groggy)
    groggy_time = time.time() - start
    
    # Test NetworkX
    start = time.time()
    nx_g = nx.Graph()
    nx_g.add_nodes_from(nodes_nx)
    nx_time = time.time() - start
    
    ratio = nx_time / groggy_time if groggy_time > 0 else float('inf')
    
    print(f"Nodes: {num_nodes:5d}, Attrs: {num_attrs:2d} | "
          f"Groggy: {groggy_time:.4f}s, NetworkX: {nx_time:.4f}s | "
          f"Ratio: {ratio:.2f}x")
    
    return groggy_time, nx_time, ratio

if __name__ == "__main__":
    print("SCALING ANALYSIS - Finding Performance Crossover")
    print("="*60)
    print("Testing node creation with varying attributes...")
    print()
    
    # Test with different attribute counts
    for num_attrs in [0, 1, 5, 10, 20, 50]:
        print(f"\n--- {num_attrs} attributes per node ---")
        for num_nodes in [100, 500, 1000, 2000, 5000]:
            try:
                quick_test(num_nodes, num_attrs)
            except Exception as e:
                print(f"Error at {num_nodes} nodes: {e}")
                break
