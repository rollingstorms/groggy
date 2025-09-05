#!/usr/bin/env python3
"""
Test the hierarchical meta-node collapse issue.
"""

import groggy as gr

def test_hierarchical_collapse_issue():
    """Test the hierarchical collapse where meta-nodes lose subgraph data."""
    print("=== TESTING HIERARCHICAL META-NODE ISSUE ===")
    
    # Create a graph with 2 clusters of high-degree nodes
    g = gr.Graph()
    
    # Create cluster 1: nodes 0-3 (fully connected)
    cluster1_nodes = []
    for i in range(4):
        node_id = g.add_node(name=f"C1_Node_{i}", cluster=1)
        cluster1_nodes.append(node_id)
    
    # Make cluster 1 fully connected
    for i in range(len(cluster1_nodes)):
        for j in range(i+1, len(cluster1_nodes)):
            g.add_edge(cluster1_nodes[i], cluster1_nodes[j], weight=1.0)
    
    # Create cluster 2: nodes 4-6 (fully connected)
    cluster2_nodes = []
    for i in range(3):
        node_id = g.add_node(name=f"C2_Node_{i}", cluster=2)
        cluster2_nodes.append(node_id)
    
    # Make cluster 2 fully connected
    for i in range(len(cluster2_nodes)):
        for j in range(i+1, len(cluster2_nodes)):
            g.add_edge(cluster2_nodes[i], cluster2_nodes[j], weight=1.0)
    
    print(f"Initial graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    
    # Step 1: Find high-degree nodes and create components
    high_degree_node_ids = [node.id for node in g.nodes if node.degree > 2]
    print(f"High degree nodes: {high_degree_node_ids}")
    print(f"Node degrees: {[(node.id, node.degree) for node in g.nodes]}")
    
    high_degree_subgraph = g.nodes[high_degree_node_ids]
    components = high_degree_subgraph.connected_components()
    print(f"Components found: {len(components)}")
    
    # Step 2: Collapse each component using EXTRACT strategy
    meta_node_ids = []
    for i, component in enumerate(components):
        print(f"\nComponent {i+1}: {component.node_count()} nodes")
        
        meta_node = component.collapse(
            node_aggs={"size": "count"},
            edge_strategy='keep_external',
            node_strategy='extract',  # Keep originals
        )
        
        print(f"  Created meta-node {meta_node.id}")
        print(f"  Meta-node has_subgraph: {meta_node.has_subgraph}")
        if hasattr(meta_node, 'subgraph'):
            print(f"  Meta-node subgraph node count: {meta_node.subgraph.node_count()}")
        
        meta_node_ids.append(meta_node.id)
    
    print(f"\nAfter component collapse: {len(g.nodes)} nodes")
    print(f"Meta-node IDs: {meta_node_ids}")
    
    # Step 3: Check if we can access meta-nodes
    for meta_id in meta_node_ids:
        try:
            # Get the node from the graph using indexing
            meta_node_obj = list(g.nodes.filter(f"id == {meta_id}"))[0]
            print(f"Meta-node {meta_id}:")
            print(f"  has_subgraph: {meta_node_obj.has_subgraph}")
            if hasattr(meta_node_obj, 'subgraph'):
                print(f"  subgraph node count: {meta_node_obj.subgraph.node_count()}")
        except Exception as e:
            print(f"Meta-node {meta_id}: error accessing: {e}")
    
    # Step 4: Try to collapse the meta-nodes (this is where they might lose subgraph data)
    if len(meta_node_ids) > 1:
        print(f"\n--- Collapsing meta-nodes themselves ---")
        meta_subgraph = g.nodes[meta_node_ids]
        print(f"Meta-subgraph: {meta_subgraph.node_count()} nodes")
        
        super_meta = meta_subgraph.collapse(
            node_aggs={"total_components": "count"},
            edge_strategy='keep_external',
            node_strategy='extract',  # Keep meta-nodes
        )
        
        print(f"Super meta-node created: {super_meta.id}")
        print(f"Super meta-node has_subgraph: {super_meta.has_subgraph}")
        
        print(f"Final graph: {len(g.nodes)} nodes")
        
        # Check if the component meta-nodes still exist and have their subgraphs
        print(f"\n--- Checking component meta-nodes after super collapse ---")
        for meta_id in meta_node_ids:
            try:
                meta_node_obj = list(g.nodes.filter(f"id == {meta_id}"))[0]
                print(f"Meta-node {meta_id}:")
                print(f"  exists: True")
                print(f"  has_subgraph: {meta_node_obj.has_subgraph}")
                if hasattr(meta_node_obj, 'subgraph'):
                    try:
                        subgraph_count = meta_node_obj.subgraph.node_count()
                        print(f"  subgraph node count: {subgraph_count}")
                    except Exception as e:
                        print(f"  subgraph access error: {e}")
                else:
                    print("  no subgraph attribute")
            except Exception as e:
                print(f"Meta-node {meta_id}: does not exist or error: {e}")

if __name__ == "__main__":
    test_hierarchical_collapse_issue()
