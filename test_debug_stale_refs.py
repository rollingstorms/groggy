#!/usr/bin/env python3
"""
Debug test for stale reference issue with meta-nodes.
Reproducing the error: RuntimeError: Node 103 not found while attempting to add edge
"""

import sys
import traceback

try:
    import groggy as gr
    print("✓ Groggy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import groggy: {e}")
    sys.exit(1)

def create_test_graph():
    """Create a test graph that should trigger the stale reference issue"""
    print("Creating test graph...")
    g = gr.Graph()
    
    # Create nodes with high degree
    for i in range(10):
        g.add_node(name=f"Node_{i}", value=i * 10)
    
    # Create dense connections to get high degree nodes
    for i in range(10):
        for j in range(i+1, 10):
            g.add_edge(i, j, weight=0.5 + (i+j) * 0.1)
    
    print(f"Created graph with {g.node_count} nodes and {g.edge_count} edges")
    
    # Show degree distribution
    degrees = g.degree()
    print("Degree distribution:")
    for i, degree in enumerate(degrees):
        if degree > 4:
            print(f"  Node {i}: degree {degree}")
    
    return g

def test_reproduce_stale_reference_error():
    """Try to reproduce the exact error the user encountered"""
    print("\n=== Test: Reproduce Stale Reference Error ===")
    
    g = create_test_graph()
    
    try:
        print("Getting high-degree nodes...")
        high_degree_nodes = g.nodes[g.degree() > 4]
        print(f"Found {len(high_degree_nodes)} high-degree nodes")
        
        print("Getting connected components...")
        components = high_degree_nodes.connected_components()
        print(f"Found {len(components)} components")
        
        print("Iterating through components and attempting collapse...")
        for i, component in enumerate(components):
            print(f"\nProcessing component {i}:")
            print(f"  Component has {component.node_count()} nodes, {component.edge_count()} edges")
            
            try:
                print("  Attempting collapse with edge_strategy='keep_external'...")
                meta_node = component.collapse(
                    node_aggs={"size": "count"},
                    edge_strategy='keep_external'
                )
                print(f"  ✓ Successfully created meta-node: {meta_node}")
                
            except Exception as e:
                print(f"  ✗ Error during collapse: {e}")
                print("  Full traceback:")
                traceback.print_exc()
                return False
                
    except Exception as e:
        print(f"✗ Error during operation: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False
    
    return True

def test_sequential_meta_node_operations():
    """Test if multiple meta-node operations cause stale references"""
    print("\n=== Test: Sequential Meta-Node Operations ===")
    
    g = gr.Graph()
    
    # Create a simpler graph to isolate the issue
    for i in range(6):
        g.add_node(name=f"Node_{i}", group=i//2)
    
    # Create edges
    for i in range(5):
        g.add_edge(i, i+1, weight=1.0)
    
    print(f"Created simple graph: {g.node_count} nodes, {g.edge_count} edges")
    
    try:
        # First meta-node operation
        print("Creating first meta-node...")
        subgraph1 = g.nodes[[0, 1]]
        meta1 = subgraph1.collapse(node_aggs={"size": "count"})
        print(f"✓ First meta-node created: {meta1}")
        print(f"Graph now has {g.node_count} nodes, {g.edge_count} edges")
        
        # Second meta-node operation
        print("Creating second meta-node...")
        remaining_nodes = [nid for nid in g.node_ids if nid != meta1.id]
        print(f"Remaining node IDs: {remaining_nodes}")
        
        if len(remaining_nodes) >= 2:
            subgraph2 = g.nodes[remaining_nodes[:2]]
            meta2 = subgraph2.collapse(node_aggs={"size": "count"})
            print(f"✓ Second meta-node created: {meta2}")
            print(f"Graph now has {g.node_count} nodes, {g.edge_count} edges")
        
    except Exception as e:
        print(f"✗ Error during sequential operations: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False
    
    return True

def test_node_removal_edge_consistency():
    """Test if node removal properly handles edge cleanup"""
    print("\n=== Test: Node Removal Edge Consistency ===")
    
    g = gr.Graph()
    
    # Create nodes
    for i in range(5):
        g.add_node(name=f"Node_{i}")
    
    # Create edges
    edges_created = []
    for i in range(4):
        g.add_edge(i, i+1, weight=1.0)
        edges_created.append((i, i+1))
    
    print(f"Created graph: {g.node_count} nodes, {g.edge_count} edges")
    print(f"Edges created: {edges_created}")
    
    # Show all edge IDs
    print("Edge IDs and endpoints:")
    for edge in g.edges:
        print(f"  Edge {edge.id}: {edge.source} -> {edge.target}")
    
    try:
        # Create meta-node and see what happens to edges
        print("\nCreating meta-node from nodes [0, 1]...")
        subgraph = g.nodes[[0, 1]]
        meta_node = subgraph.collapse(node_aggs={"size": "count"})
        
        print(f"✓ Meta-node created: {meta_node}")
        print(f"Graph now has {g.node_count} nodes, {g.edge_count} edges")
        
        print("Remaining edge IDs and endpoints:")
        for edge in g.edges:
            print(f"  Edge {edge.id}: {edge.source} -> {edge.target}")
            
        print("Remaining node IDs:")
        for nid in g.node_ids:
            print(f"  Node {nid}")
            
    except Exception as e:
        print(f"✗ Error during node removal test: {e}")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all debugging tests"""
    print("Meta-Node Stale Reference Debug Tests")
    print("=" * 50)
    
    tests = [
        test_reproduce_stale_reference_error,
        test_sequential_meta_node_operations,
        test_node_removal_edge_consistency,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            if result:
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"💥 {test.__name__} CRASHED: {e}")
            traceback.print_exc()
            results.append(False)
    
    # Summary
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"DEBUG TEST RESULTS")
    print(f"{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed < total:
        print(f"⚠️ {total - passed} test(s) revealed issues")
    else:
        print("🎉 All tests passed - issue may be elsewhere")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())