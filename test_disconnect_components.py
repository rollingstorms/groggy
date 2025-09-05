#!/usr/bin/env python3
"""
Test with actual disconnected components to test hierarchical meta-node issues.
"""

import groggy as gr

def test_disconnected_components():
    """Test with truly disconnected components."""
    print("=== DISCONNECTED COMPONENTS TEST ===")
    
    # Create two completely separate clusters
    g = gr.Graph()
    
    # Cluster 1: triangle (nodes 0,1,2)
    g.add_node(name="A1", cluster=1)  # id=0
    g.add_node(name="A2", cluster=1)  # id=1
    g.add_node(name="A3", cluster=1)  # id=2
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=1.0) 
    g.add_edge(2, 0, weight=1.0)
    
    # Cluster 2: triangle (nodes 3,4,5) - completely disconnected
    g.add_node(name="B1", cluster=2)  # id=3
    g.add_node(name="B2", cluster=2)  # id=4 
    g.add_node(name="B3", cluster=2)  # id=5
    g.add_edge(3, 4, weight=1.0)
    g.add_edge(4, 5, weight=1.0)
    g.add_edge(5, 3, weight=1.0)
    
    # Single low-degree node
    g.add_node(name="isolated")  # id=6
    
    print(f"Initial: {len(g.nodes)} nodes, {len(g.edges)} edges")
    for node in g.nodes:
        print(f"  Node {node.id}: degree {node.degree}")
    
    # Get high degree nodes (degree > 1) 
    high_degree_nodes = [node.id for node in g.nodes if node.degree > 1]
    print(f"\nHigh degree nodes: {high_degree_nodes}")
    
    subgraph = g.nodes[high_degree_nodes]
    components = subgraph.connected_components()
    print(f"Components: {len(components)}")
    
    # Collapse each component separately
    meta_node_ids = []
    for i, component in enumerate(components):
        print(f"\nComponent {i}: {component.node_count()} nodes")
        
        meta_node = component.collapse(
            node_aggs={"size": "count", "cluster_id": ("first", "cluster")},
            edge_strategy='keep_external',
            node_strategy='extract'  # SHOULD keep originals
        )
        
        print(f"  Meta-node {meta_node.id} created")
        print(f"  has_subgraph: {meta_node.has_subgraph}")
        meta_node_ids.append(meta_node.id)
    
    print(f"\nAfter component collapse: {len(g.nodes)} nodes")
    print(f"Meta-node IDs: {meta_node_ids}")
    
    # Check meta-nodes in the graph
    print(f"\n--- Meta-nodes in graph ---")
    for node in g.nodes:
        if hasattr(node, 'has_subgraph') and node.has_subgraph:
            print(f"Meta-node {node.id}:")
            print(f"  has_subgraph: {node.has_subgraph}")
            if hasattr(node, 'subgraph'):
                print(f"  subgraph: {node.subgraph.node_count()} nodes")
    
    # Test the hierarchical collapse - this is where the issue might occur
    if len(meta_node_ids) >= 2:
        print(f"\n--- HIERARCHICAL COLLAPSE ---")
        print(f"Collapsing {len(meta_node_ids)} meta-nodes into super meta-node...")
        
        meta_subgraph = g.nodes[meta_node_ids]
        print(f"Meta-subgraph: {meta_subgraph.node_count()} nodes")
        
        # Check meta-nodes before super collapse
        print("Before super collapse:")
        for node_id in meta_node_ids:
            for node in g.nodes:
                if node.id == node_id:
                    print(f"  Meta-node {node_id}: has_subgraph={node.has_subgraph}")
                    break
        
        super_meta = meta_subgraph.collapse(
            node_aggs={"component_count": "count"},
            edge_strategy='keep_external', 
            node_strategy='extract'  # SHOULD keep the meta-nodes
        )
        
        print(f"Super meta-node created: {super_meta.id}")
        print(f"Super has_subgraph: {super_meta.has_subgraph}")
        print(f"Final graph: {len(g.nodes)} nodes")
        
        # Check if component meta-nodes still exist and have subgraphs
        print("\nAfter super collapse:")
        for node_id in meta_node_ids:
            found = False
            for node in g.nodes:
                if node.id == node_id:
                    found = True
                    print(f"  Meta-node {node_id}: has_subgraph={node.has_subgraph}")
                    if hasattr(node, 'subgraph') and node.has_subgraph:
                        try:
                            count = node.subgraph.node_count()
                            print(f"    subgraph: {count} nodes (GOOD)")
                        except Exception as e:
                            print(f"    subgraph ERROR: {e}")
                    else:
                        print(f"    NO SUBGRAPH (BAD)")
                    break
            if not found:
                print(f"  Meta-node {node_id}: NOT FOUND (strategy should be extract!)")
        
        # Show all meta-nodes in final graph
        print("\nAll meta-nodes in final graph:")
        for node in g.nodes:
            if hasattr(node, 'has_subgraph') and node.has_subgraph:
                print(f"  {node.id}: has_subgraph={node.has_subgraph}")

if __name__ == "__main__":
    test_disconnected_components()
