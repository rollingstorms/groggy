#!/usr/bin/env python3

"""
Simple test to compare connected components performance and identify bottlenecks.
"""

import time
import groggy

def test_simple_graph():
    """Test with a simple connected graph"""
    print("Creating simple connected graph...")
    g = groggy.Graph()
    
    # Create a chain of connected nodes: 0-1-2-3-4-...-99
    nodes = []
    for i in range(100):
        node_id = g.add_node(index=i)
        nodes.append(node_id)
    
    # Connect them in a chain
    for i in range(99):
        g.add_edge(nodes[i], nodes[i+1])
    
    print(f"Graph created: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Test connected components
    start_time = time.time()
    components = g.connected_components()
    end_time = time.time()
    
    print(f"Connected components: {len(components)} components found")
    print(f"Time taken: {end_time - start_time:.6f} seconds")
    print(f"Component sizes: {[comp.node_count() for comp in components]}")
    
    return end_time - start_time

def test_disconnected_graph():
    """Test with multiple disconnected components"""
    print("\nCreating disconnected graph...")
    g = groggy.Graph()
    
    # Create 10 separate components of 10 nodes each
    for comp in range(10):
        component_nodes = []
        for i in range(10):
            node_id = g.add_node(component=comp, index=i)
            component_nodes.append(node_id)
        
        # Connect nodes within component in a chain
        for i in range(9):
            g.add_edge(component_nodes[i], component_nodes[i+1])
    
    print(f"Graph created: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Test connected components
    start_time = time.time()
    components = g.connected_components()
    end_time = time.time()
    
    print(f"Connected components: {len(components)} components found")
    print(f"Time taken: {end_time - start_time:.6f} seconds")
    
    component_sizes = [comp.node_count() for comp in components]
    component_sizes.sort(reverse=True)
    print(f"Component sizes: {component_sizes}")
    
    return end_time - start_time

def test_large_graph():
    """Test with a larger graph to see scaling behavior"""
    print("\nCreating larger graph...")
    g = groggy.Graph()
    
    # Create a larger connected component
    nodes = []
    for i in range(1000):
        node_id = g.add_node(index=i)
        nodes.append(node_id)
    
    # Create a more connected graph - each node connects to next 3 nodes
    for i in range(1000):
        for j in range(1, 4):  # Connect to next 3 nodes
            if i + j < 1000:
                g.add_edge(nodes[i], nodes[i + j])
    
    print(f"Graph created: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Test connected components
    start_time = time.time()
    components = g.connected_components()
    end_time = time.time()
    
    print(f"Connected components: {len(components)} components found")
    print(f"Time taken: {end_time - start_time:.6f} seconds")
    print(f"Component sizes: {[comp.node_count() for comp in components]}")
    
    return end_time - start_time

def main():
    print("Connected Components Performance Analysis")
    print("=" * 50)
    
    # Run tests
    time1 = test_simple_graph()
    time2 = test_disconnected_graph()
    time3 = test_large_graph()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"Simple connected (100 nodes): {time1:.6f}s")
    print(f"Disconnected (10x10 nodes): {time2:.6f}s")
    print(f"Large connected (1000 nodes): {time3:.6f}s")

if __name__ == "__main__":
    main()
