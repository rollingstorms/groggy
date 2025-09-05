#!/usr/bin/env python3
"""
Test the exact hierarchical scenario from the user.
"""

import groggy as gr

def test_user_scenario():
    """Test the exact scenario the user described."""
    print("=== Testing User's Hierarchical Scenario ===")
    
    # Create a graph that will have multiple disconnected components of high-degree nodes
    g = gr.Graph()
    
    # Create 3 separate clusters, each with high-degree nodes
    nodes = []
    
    # Cluster 1: nodes 0-6 (high degree within cluster)
    cluster1_nodes = []
    for i in range(7):
        node = g.add_node(name=f"Cluster1_Node_{i}", cluster=1)
        nodes.append(node)
        cluster1_nodes.append(node)
    
    # Make cluster 1 densely connected (each node will have degree 6)
    for i in range(len(cluster1_nodes)):
        for j in range(i+1, len(cluster1_nodes)):
            g.add_edge(cluster1_nodes[i], cluster1_nodes[j], weight=1.0)
    
    # Cluster 2: nodes 7-12 (high degree within cluster)  
    cluster2_nodes = []
    for i in range(6):
        node = g.add_node(name=f"Cluster2_Node_{i}", cluster=2)
        nodes.append(node)
        cluster2_nodes.append(node)
    
    # Make cluster 2 densely connected (each node will have degree 5)
    for i in range(len(cluster2_nodes)):
        for j in range(i+1, len(cluster2_nodes)):
            g.add_edge(cluster2_nodes[i], cluster2_nodes[j], weight=1.0)
    
    # Add a few low-degree nodes that won't be in components
    for i in range(3):
        low_degree_node = g.add_node(name=f"Low_Degree_{i}", cluster=0)
        nodes.append(low_degree_node)
        # Connect to just one other node (low degree)
        if len(nodes) > 1:
            g.add_edge(low_degree_node, nodes[0], weight=0.1)
    
    print(f"Initial graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    
    # Check degrees
    print("Node degrees:")
    for node in g.nodes:
        print(f"  Node {node.id}: degree {node.degree}")
    
    # Step 1: components = g.nodes[g.degree() > 4].connected_components()
    print(f"\n=== Step 1: Finding connected components of high-degree nodes ===")
    
    # Find high degree nodes
    high_degree_node_ids = []
    for node in g.nodes:
        if node.degree > 4:
            high_degree_node_ids.append(node.id)
    
    print(f"High degree nodes (degree > 4): {high_degree_node_ids}")
    
    if len(high_degree_node_ids) > 0:
        # Get subgraph of high-degree nodes
        high_degree_subgraph = g.nodes[high_degree_node_ids]
        print(f"High degree subgraph: {high_degree_subgraph.node_count()} nodes")
        
        # Get connected components
        components = high_degree_subgraph.connected_components()
        print(f"Connected components: {len(components)}")
        
        # Collapse each component
        for i, component in enumerate(components):
            print(f"\nCollapsing component {i+1}: {component.node_count()} nodes")
            meta_node = component.collapse(
                node_aggs={"size": "count", "avg_cluster": ("mean", "cluster")},
                edge_strategy='keep_external',
                node_strategy='extract',  # Keep original nodes
            )
            print(f"  Created meta-node {meta_node.id}")
        
        print(f"\nAfter component collapse: {len(g.nodes)} nodes, {len(g.edges)} edges")
        
        # Count meta-nodes
        meta_node_count = 0
        meta_node_ids = []
        for node in g.nodes:
            if hasattr(node, 'has_subgraph') and node.has_subgraph:
                meta_node_count += 1
                meta_node_ids.append(node.id)
        
        print(f"Meta-nodes created: {meta_node_count} (IDs: {meta_node_ids})")
        
        # Step 2: g.nodes.meta.all().collapse(...)
        print(f"\n=== Step 2: Collapsing all meta-nodes ===")
        
        if meta_node_count > 1:
            # Create subgraph of all meta-nodes
            meta_subgraph = g.nodes[meta_node_ids]
            print(f"Meta-node subgraph: {meta_subgraph.node_count()} nodes")
            
            # Collapse all meta-nodes into one super meta-node
            super_meta = meta_subgraph.collapse(
                node_aggs={"total_components": "count"},
                edge_strategy='keep_external',
                node_strategy='extract',  # Keep the component meta-nodes
            )
            print(f"Created super meta-node: {super_meta.id}")
            
            # Final count
            final_meta_count = 0
            for node in g.nodes:
                if hasattr(node, 'has_subgraph') and node.has_subgraph:
                    final_meta_count += 1
            
            print(f"\nFinal result:")
            print(f"  Total nodes: {len(g.nodes)}")
            print(f"  Meta-nodes: {final_meta_count}")
            print(f"  Expected output: {final_meta_count}")
            
        else:
            print(f"Only {meta_node_count} meta-node found, no super meta-node needed")
            print(f"Final meta-node count: {meta_node_count}")

if __name__ == "__main__":
    test_user_scenario()
