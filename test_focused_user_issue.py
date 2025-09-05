#!/usr/bin/env python3
"""
Focus test on the exact user issue: extract acting like collapse, meta-nodes losing subgraphs.
"""

import groggy as gr

def test_extract_vs_collapse():
    """Test extract vs collapse behavior."""
    print("=== EXTRACT VS COLLAPSE TEST ===")
    
    # Test 1: Extract should keep original nodes
    print("Test 1: EXTRACT strategy")
    g1 = gr.Graph()
    g1.add_node(name="A")  # 0
    g1.add_node(name="B")  # 1 
    g1.add_edge(0, 1, weight=1.0)
    
    print(f"Before: {len(g1.nodes)} nodes")
    subgraph1 = g1.nodes[[0, 1]]
    
    meta_node1 = subgraph1.collapse(
        node_aggs={"size": "count"},
        node_strategy='extract'  # Should keep nodes 0,1
    )
    
    print(f"After EXTRACT: {len(g1.nodes)} nodes (expected: 3)")
    if len(g1.nodes) == 3:
        print("✅ EXTRACT working correctly")
    else:
        print("❌ EXTRACT not working - behaving like COLLAPSE")
    
    # Test 2: Collapse should remove original nodes
    print("\nTest 2: COLLAPSE strategy")
    g2 = gr.Graph()
    g2.add_node(name="A")  # 0
    g2.add_node(name="B")  # 1 
    g2.add_edge(0, 1, weight=1.0)
    
    print(f"Before: {len(g2.nodes)} nodes")
    subgraph2 = g2.nodes[[0, 1]]
    
    meta_node2 = subgraph2.collapse(
        node_aggs={"size": "count"},
        node_strategy='collapse'  # Should remove nodes 0,1
    )
    
    print(f"After COLLAPSE: {len(g2.nodes)} nodes (expected: 1)")
    if len(g2.nodes) == 1:
        print("✅ COLLAPSE working correctly")
    else:
        print("❌ COLLAPSE not working")

def test_hierarchical_meta_node_issue():
    """Test the hierarchical meta-node subgraph preservation issue."""
    print("\n=== HIERARCHICAL META-NODE ISSUE ===")
    
    g = gr.Graph()
    
    # Create 2 separate triangles
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
    
    # Step 1: Create components and collapse each
    all_nodes = [node.id for node in g.nodes]
    subgraph = g.nodes[all_nodes]
    components = subgraph.connected_components()
    
    print(f"Components found: {len(components)}")
    
    meta_nodes = []
    for i, comp in enumerate(components):
        meta_node = comp.collapse(
            node_aggs={"size": "count"},
            edge_strategy='keep_external',
            node_strategy='extract'  # Keep originals
        )
        meta_nodes.append(meta_node)
        print(f"Component {i}: meta-node {meta_node.id}, has_subgraph: {meta_node.has_subgraph}")
    
    print(f"After component collapse: {len(g.nodes)} nodes")
    
    # Step 2: Check meta-node types in graph
    print("\nMeta-nodes in graph:")
    meta_node_ids = [mn.id for mn in meta_nodes]
    for node in g.nodes:
        if node.id in meta_node_ids:
            print(f"  Node {node.id}: {type(node).__name__}, has_subgraph: {node.has_subgraph}")
    
    # Step 3: Collapse the meta-nodes hierarchically
    if len(meta_nodes) > 1:
        print(f"\nHierarchical collapse of {len(meta_nodes)} meta-nodes...")
        
        meta_subgraph = g.nodes[meta_node_ids]
        super_meta = meta_subgraph.collapse(
            node_aggs={"component_count": "count"},
            edge_strategy='keep_external',
            node_strategy='extract'  # Should keep meta-nodes
        )
        
        print(f"Super meta-node: {super_meta.id}")
        print(f"Final graph: {len(g.nodes)} nodes")
        
        # Step 4: Check if original meta-nodes preserved their status
        print("\nAfter hierarchical collapse:")
        for node in g.nodes:
            if node.id in meta_node_ids:
                node_type = type(node).__name__
                if node_type == "MetaNode":
                    print(f"  ✅ Node {node.id}: Still MetaNode, has_subgraph: {node.has_subgraph}")
                else:
                    print(f"  ❌ Node {node.id}: LOST MetaNode status, now: {node_type}")
    
    return len([n for n in g.nodes if hasattr(n, 'has_subgraph') and n.has_subgraph])

if __name__ == "__main__":
    test_extract_vs_collapse()
    final_meta_count = test_hierarchical_meta_node_issue()
    
    print(f"\n=== SUMMARY ===")
    print(f"Final meta-node count: {final_meta_count}")
    
    if final_meta_count >= 1:  # Should have at least the super meta-node
        print("✅ Some meta-nodes preserved")
    else:
        print("❌ All meta-nodes lost their status")
