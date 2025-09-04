#!/usr/bin/env python3
"""Debug meta-node creation step by step"""

import sys
sys.path.append('.')
import groggy as gr

def test_debug_meta_node():
    print("=== Debug Meta-Node Creation ===")
    
    # Create simple graph
    g = gr.Graph(directed=False)
    g.add_node(name="A")
    g.add_node(name="B") 
    
    g.add_edge(0, 1, weight=1.0)
    
    # Create subgraph
    subgraph = g.nodes[[0, 1]]
    print(f"Subgraph: {subgraph}")
    
    # Test 1: Create meta-node without missing attributes (should use strict path)
    print("\n1. Testing strict path (allow_missing_attributes=False):")
    try:
        meta_node1 = subgraph.collapse(
            node_aggs={"size": "count"},
            allow_missing_attributes=False
        )
        print(f"   Meta-node created: ID {meta_node1.node_id}")
        print(f"   has_subgraph: {meta_node1.has_subgraph}")
        print(f"   subgraph_id: {meta_node1.subgraph_id}")
        
        # Check attribute directly
        attr_value = g.get_node_attr(meta_node1.node_id, 'contains_subgraph')
        print(f"   contains_subgraph attr: {attr_value} (type: {type(attr_value)})")
        
        subgraph_result = meta_node1.subgraph
        print(f"   subgraph property: {subgraph_result}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Create meta-node with defaults (should use defaults path)
    print("\n2. Testing defaults path (allow_missing_attributes=True):")
    try:
        g2 = gr.Graph(directed=False)
        g2.add_node(name="X")
        g2.add_node(name="Y") 
        g2.add_edge(0, 1, weight=2.0)
        
        subgraph2 = g2.nodes[[0, 1]]
        
        meta_node2 = subgraph2.collapse(
            node_aggs={"size": "count"},
            allow_missing_attributes=True
        )
        print(f"   Meta-node created: ID {meta_node2.node_id}")
        print(f"   has_subgraph: {meta_node2.has_subgraph}")
        print(f"   subgraph_id: {meta_node2.subgraph_id}")
        
        # Check attribute directly
        attr_value = g2.get_node_attr(meta_node2.node_id, 'contains_subgraph')
        print(f"   contains_subgraph attr: {attr_value} (type: {type(attr_value)})")
        
        # Check debug attribute
        try:
            debug_attr = g2.get_node_attr(meta_node2.node_id, 'debug_defaults_path')
            print(f"   debug_defaults_path attr: {debug_attr}")
        except:
            print(f"   debug_defaults_path attr: <not found>")
        
        subgraph_result = meta_node2.subgraph
        print(f"   subgraph property: {subgraph_result}")
        
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_debug_meta_node()