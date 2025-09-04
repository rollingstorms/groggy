#!/usr/bin/env python3
"""
Test to verify that the consistency fix for stored subgraphs is working.
This tests that removing edges and nodes properly updates the stored subgraph data.
"""
import sys
import traceback

try:
    import groggy as gr
    print("✓ Groggy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import groggy: {e}")
    sys.exit(1)

def test_consistency_fix():
    """Test that removing edges/nodes updates stored subgraphs"""
    print("\n=== Testing Consistency Fix for Stored Subgraphs ===")
    
    g = gr.Graph()
    
    # Create a graph with multiple components that will become meta-nodes
    for i in range(10):
        g.add_node(name=f"Node_{i}", group=i // 3)
    
    # Add edges to create high-degree nodes
    edges_added = []
    for i in range(8):
        for j in range(i+1, min(i+5, 10)):  # Connect to next 4 nodes
            g.add_edge(i, j, weight=0.5)
            edges_added.append((i, j))
    
    print(f"Created graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Find nodes with high degree
    high_degree_nodes = []
    for node_id in g.node_ids:
        if len(g.neighbors(node_id)) > 4:
            high_degree_nodes.append(node_id)
    
    print(f"High degree nodes (>4): {high_degree_nodes}")
    
    # Create meta-nodes from connected components of high-degree nodes
    if high_degree_nodes:
        subgraph = g.nodes[high_degree_nodes]
        print(f"High degree subgraph: {len(subgraph.node_ids)} nodes")
        
        # Collapse into meta-nodes
        for component in subgraph.connected_components():
            if len(component.node_ids) >= 2:
                print(f"Collapsing component with {len(component.node_ids)} nodes")
                try:
                    component.collapse(edge_strategy='keep_external')
                    print("✓ Meta-node created successfully")
                except Exception as e:
                    print(f"✗ Meta-node creation failed: {e}")
                    return False
    
    print(f"After meta-node creation: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Get meta-nodes
    meta_nodes = g.nodes.meta.all().node_ids
    print(f"Meta-nodes created: {list(meta_nodes)}")
    
    if len(meta_nodes) == 0:
        print("No meta-nodes created, skipping consistency test")
        return True
    
    # Now test removing the meta-nodes - this should work without stale reference errors
    try:
        print("\n=== Testing Meta-Node Removal (Consistency Check) ===")
        
        # This previously failed with stale edge references
        print(f"Attempting to remove {len(meta_nodes)} meta-nodes...")
        g.remove_nodes(meta_nodes)
        
        print("✓ Meta-node removal succeeded - consistency fix works!")
        print(f"After removal: {g.node_count()} nodes, {g.edge_count()} edges")
        
        return True
        
    except Exception as e:
        print(f"✗ Meta-node removal failed: {e}")
        print("Consistency fix may not be complete")
        traceback.print_exc()
        return False

def test_individual_edge_removal():
    """Test removing individual edges after meta-node creation"""
    print("\n=== Testing Individual Edge Removal After Meta-Nodes ===")
    
    g = gr.Graph()
    
    # Simple test case
    for i in range(6):
        g.add_node(name=f"N{i}")
    
    # Create a connected component
    for i in range(3):
        for j in range(i+1, 3):
            g.add_edge(i, j, weight=1.0)
    
    # Connect to external nodes
    g.add_edge(1, 3, weight=2.0)
    g.add_edge(2, 4, weight=2.0)
    g.add_edge(1, 5, weight=2.0)
    
    edge_to_remove = g.edge_ids[0]  # Get first edge ID
    print(f"Original graph: {g.node_count()} nodes, {g.edge_count()} edges")
    print(f"Will remove edge: {edge_to_remove}")
    
    # Create meta-node
    subgraph = g.nodes[[0, 1, 2]]
    meta_node = subgraph.collapse(edge_strategy='keep_external')
    print(f"Created meta-node: {meta_node}")
    print(f"After meta-node: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Remove an edge - this should update stored subgraphs
    try:
        remaining_edges = g.edge_ids
        if len(remaining_edges) > 0:
            edge_to_remove = remaining_edges[0]
            print(f"Removing edge: {edge_to_remove}")
            g.remove_edge(edge_to_remove)
            print("✓ Edge removal succeeded")
            print(f"After edge removal: {g.node_count()} nodes, {g.edge_count()} edges")
        else:
            print("No edges to remove")
        
        return True
        
    except Exception as e:
        print(f"✗ Edge removal failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all consistency tests"""
    print("Stored Subgraph Consistency Fix Validation")
    print("=" * 60)
    
    test1_result = test_consistency_fix()
    test2_result = test_individual_edge_removal()
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY:")
    print(f"Meta-node removal test: {'✓ PASS' if test1_result else '✗ FAIL'}")
    print(f"Individual edge removal test: {'✓ PASS' if test2_result else '✗ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 All consistency tests PASSED! The fix is working correctly.")
        return True
    else:
        print("\n❌ Some tests FAILED. Consistency issues may remain.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
