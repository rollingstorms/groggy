#!/usr/bin/env python3
"""
Test both backends to ensure attribute parity
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/gli')

import gli
import time

def test_backend_parity():
    print("Testing backend parity for attributes...")
    
    results = {}
    
    for backend_name in ['python', 'rust']:
        print(f"\n=== Testing {backend_name.upper()} backend ===")
        
        gli.set_backend(backend_name)
        print(f"Current backend: {gli.get_current_backend()}")
        
        # Create a graph
        g = gli.Graph.empty()
        
        # Add nodes with attributes
        node1_id = g.add_node(name="Alice", age=30, active=True)
        node2_id = g.add_node(name="Bob", age=25, active=False)
        
        # Add edge with attributes
        edge_id = g.add_edge(node1_id, node2_id, relationship="friend", strength=0.8)
        
        # Test attribute retrieval
        node1 = g.get_node(node1_id)
        node2 = g.get_node(node2_id)
        edge = g.get_edge(edge_id)
        
        # Test modification
        g.set_node_attribute(node1_id, "age", 31)
        g.set_edge_attribute(edge_id, "strength", 0.9)
        
        # Test complex data
        g.set_node_attribute(node1_id, "hobbies", ["reading", "coding", "hiking"])
        g.set_node_attribute(node1_id, "profile", {"city": "SF", "job": "Engineer"})
        
        # Collect results
        updated_node1 = g.get_node(node1_id)
        updated_edge = g.get_edge(edge_id)
        
        results[backend_name] = {
            'node1_attrs': dict(updated_node1.attributes),
            'node2_attrs': dict(node2.attributes),
            'edge_attrs': dict(updated_edge.attributes),
            'node_count': g.node_count(),
            'edge_count': g.edge_count()
        }
        
        print(f"Node 1: {results[backend_name]['node1_attrs']}")
        print(f"Node 2: {results[backend_name]['node2_attrs']}")
        print(f"Edge: {results[backend_name]['edge_attrs']}")
        print(f"Graph stats: {results[backend_name]['node_count']} nodes, {results[backend_name]['edge_count']} edges")
    
    # Compare results
    print(f"\n=== BACKEND COMPARISON ===")
    
    python_results = results['python']
    rust_results = results['rust']
    
    # Check if results are equivalent
    differences = []
    
    # Compare attributes (order doesn't matter)
    if set(python_results['node1_attrs'].keys()) != set(rust_results['node1_attrs'].keys()):
        differences.append("Node 1 attribute keys differ")
    else:
        for key in python_results['node1_attrs']:
            if python_results['node1_attrs'][key] != rust_results['node1_attrs'][key]:
                differences.append(f"Node 1 attribute '{key}' differs")
                
    if set(python_results['edge_attrs'].keys()) != set(rust_results['edge_attrs'].keys()):
        differences.append("Edge attribute keys differ")
    else:
        for key in python_results['edge_attrs']:
            if python_results['edge_attrs'][key] != rust_results['edge_attrs'][key]:
                differences.append(f"Edge attribute '{key}' differs")
    
    if python_results['node_count'] != rust_results['node_count']:
        differences.append("Node counts differ")
        
    if python_results['edge_count'] != rust_results['edge_count']:
        differences.append("Edge counts differ")
    
    if not differences:
        print("✅ Both backends produce identical results!")
    else:
        print("❌ Backend differences found:")
        for diff in differences:
            print(f"  - {diff}")
    
    return len(differences) == 0

def test_performance_comparison():
    print(f"\n=== PERFORMANCE COMPARISON ===")
    
    for backend_name in ['python', 'rust']:
        gli.set_backend(backend_name)
        
        start_time = time.time()
        
        g = gli.Graph.empty()
        
        # Create nodes and edges with attributes
        node_ids = []
        for i in range(1000):
            node_id = g.add_node(
                id=i,
                name=f"Node_{i}",
                value=i * 0.1,
                active=i % 2 == 0
            )
            node_ids.append(node_id)
        
        # Create edges
        for i in range(0, len(node_ids)-1, 2):
            g.add_edge(node_ids[i], node_ids[i+1], weight=0.5, type="connection")
        
        # Test retrieval and modification
        for i in range(min(100, len(node_ids))):
            node = g.get_node(node_ids[i])
            g.set_node_attribute(node_ids[i], "updated", True)
        
        end_time = time.time()
        
        print(f"{backend_name.upper()} backend: {end_time - start_time:.3f} seconds")
        print(f"  Nodes: {g.node_count()}, Edges: {g.edge_count()}")

if __name__ == "__main__":
    success = test_backend_parity()
    test_performance_comparison()
    
    if success:
        print(f"\n🎉 All tests passed! Both backends support attributes correctly.")
    else:
        print(f"\n⚠️  Some differences found between backends.")
