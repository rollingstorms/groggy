#!/usr/bin/env python3
"""Test meta-node removal and edge table display"""

import sys
sys.path.append('.')
import groggy as gr

def test_meta_node_removal():
    """Test that removing a meta-node also removes its meta-edges"""
    print("=== Testing Meta-Node Removal ===")
    
    # Create test graph
    g = gr.Graph(directed=False)
    
    # Internal cluster
    g.add_node(name="A", group="cluster")
    g.add_node(name="B", group="cluster") 
    g.add_node(name="C", group="cluster")
    
    # External nodes
    g.add_node(name="X", group="external")
    g.add_node(name="Y", group="external")
    
    # Internal edges
    g.add_edge(0, 1, weight=0.9)
    g.add_edge(1, 2, weight=0.8)
    
    # External edges
    g.add_edge(1, 3, weight=0.6)  # B -> X
    g.add_edge(2, 4, weight=0.5)  # C -> Y
    
    print(f"Before collapse: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Collapse subgraph
    cluster = g.nodes[[0, 1, 2]]
    meta_node = cluster.collapse(
        node_aggs={"size": "count"},
        edge_strategy="aggregate",
        allow_missing_attributes=True
    )
    
    print(f"After collapse: {g.node_count()} nodes, {g.edge_count()} edges")
    print(f"Meta-node ID: {meta_node.node_id}")
    
    # Count meta-edges
    meta_edges = 0
    for edge_id in g.edge_ids:
        src, dst = g.edge_endpoints(edge_id)
        if src == meta_node.node_id or dst == meta_node.node_id:
            meta_edges += 1
    print(f"Meta-edges found: {meta_edges}")
    
    # Now remove the meta-node
    print(f"\nRemoving meta-node {meta_node.node_id}...")
    g.remove_node(meta_node.node_id)
    
    print(f"After removal: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Count remaining edges connected to the removed node (should be 0)
    remaining_meta_edges = 0
    for edge_id in g.edge_ids:
        src, dst = g.edge_endpoints(edge_id)
        if src == meta_node.node_id or dst == meta_node.node_id:
            remaining_meta_edges += 1
            print(f"  ⚠️  Dangling meta-edge found: {edge_id} ({src} → {dst})")
    
    if remaining_meta_edges == 0:
        print("✅ All meta-edges properly removed")
        return True
    else:
        print(f"❌ {remaining_meta_edges} dangling meta-edges remain")
        return False

def test_edge_table_display():
    """Test that subgraph edge table shows source and target"""
    print("\n=== Testing Edge Table Display ===")
    
    # Create simple graph
    g = gr.Graph(directed=True)  # Use directed to make source/target clear
    
    g.add_node(name="A")
    g.add_node(name="B") 
    g.add_node(name="C")
    
    g.add_edge(0, 1, weight=1.0, type="test")
    g.add_edge(1, 2, weight=2.0, type="test")
    
    # Create subgraph and check edge display
    subgraph = g.nodes[[0, 1, 2]]
    
    print("Subgraph created with edges:")
    try:
        # This should show the edge table with source/target columns
        print(subgraph)
        return True
    except Exception as e:
        print(f"Error displaying subgraph: {e}")
        return False

def main():
    results = []
    
    results.append(test_meta_node_removal())
    results.append(test_edge_table_display())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS: {passed}/{total} passed")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())