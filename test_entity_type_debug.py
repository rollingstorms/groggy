#!/usr/bin/env python3
"""
Debug entity_type attribute preservation during hierarchical collapse.
"""

import groggy as gr

def debug_entity_type_preservation():
    """Debug what happens to entity_type during collapse."""
    print("=== DEBUGGING ENTITY_TYPE PRESERVATION ===")
    
    # Simple case with 2 triangles
    g = gr.Graph()
    
    # Triangle 1
    g.add_node(name="A1")  # 0
    g.add_node(name="A2")  # 1 
    g.add_node(name="A3")  # 2
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    g.add_edge(2, 0, weight=1.0)
    
    # Triangle 2  
    g.add_node(name="B1")  # 3
    g.add_node(name="B2")  # 4
    g.add_node(name="B3")  # 5
    g.add_edge(3, 4, weight=1.0)
    g.add_edge(4, 5, weight=1.0) 
    g.add_edge(5, 3, weight=1.0)
    
    print(f"Initial: {len(g.nodes)} nodes")
    
    # Create components and collapse them
    all_nodes = [node.id for node in g.nodes]
    subgraph = g.nodes[all_nodes]
    components = subgraph.connected_components()
    
    meta_node_ids = []
    for i, comp in enumerate(components):
        meta_node = comp.collapse(
            node_aggs={"size": "count"},
            edge_strategy='keep_external',
            node_strategy='extract'
        )
        meta_node_ids.append(meta_node.id)
        print(f"Component {i} -> meta-node {meta_node.id}")
    
    # Check entity_type attributes of all nodes BEFORE super collapse
    print(f"\n--- BEFORE super collapse ---")
    for node in g.nodes:
        entity_type = None
        try:
            if hasattr(node, 'attrs') and 'entity_type' in node.attrs:
                entity_type = node.attrs['entity_type']
            elif hasattr(node, 'get_attribute'):
                entity_type = node.get_attribute('entity_type')
        except:
            entity_type = "unknown"
        
        print(f"Node {node.id}: type={type(node).__name__}, entity_type={entity_type}")
    
    # Now collapse the meta-nodes
    print(f"\nCollapsing meta-nodes {meta_node_ids}...")
    meta_subgraph = g.nodes[meta_node_ids]
    super_meta = meta_subgraph.collapse(
        node_aggs={"total": "count"},
        edge_strategy='keep_external',
        node_strategy='extract'  # Extract should preserve them
    )
    
    print(f"Super meta-node: {super_meta.id}")
    
    # Check entity_type attributes of all nodes AFTER super collapse
    print(f"\n--- AFTER super collapse ---")
    for node in g.nodes:
        entity_type = None
        try:
            if hasattr(node, 'attrs') and 'entity_type' in node.attrs:
                entity_type = node.attrs['entity_type']
            elif hasattr(node, 'get_attribute'):
                entity_type = node.get_attribute('entity_type')
        except:
            entity_type = "unknown"
        
        print(f"Node {node.id}: type={type(node).__name__}, entity_type={entity_type}")
        
        # Check if this was originally a meta-node
        if node.id in meta_node_ids:
            print(f"  ^^^ This WAS a meta-node (ID {node.id}) - should still be MetaNode!")

if __name__ == "__main__":
    debug_entity_type_preservation()
