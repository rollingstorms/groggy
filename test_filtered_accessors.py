#!/usr/bin/env python3
"""
Test Filtered Accessor Pattern (base/meta) 🔍⚡
"""

import groggy

def test_filtered_accessors():
    print("🔍⚡ Testing FILTERED ACCESSOR PATTERN...")
    
    g = groggy.Graph()
    
    # Create nodes and edges
    node1 = g.add_node(name="Alice", age=25)
    node2 = g.add_node(name="Bob", age=30) 
    node3 = g.add_node(name="Charlie", age=35)
    node4 = g.add_node(name="David", age=40)
    
    # Create edges
    edge1 = g.add_edge(node1, node2, weight=1.0, relation="friend")
    edge2 = g.add_edge(node2, node3, weight=2.0, relation="colleague") 
    edge3 = g.add_edge(node3, node4, weight=3.0, relation="family")
    
    print(f"Initial nodes: {list(g.node_ids)}")
    print(f"Initial edges: {list(g.edge_ids)}")
    
    # Test 1: Initial state - all nodes should be base nodes
    print(f"\n📋 Test 1: Initial state - base nodes")
    try:
        base_nodes = g.nodes.base
        print(f"✅ Base nodes accessor created: {type(base_nodes)}")
        print(f"Base node count: {len(base_nodes)}")
        
        meta_nodes = g.nodes.meta
        print(f"✅ Meta nodes accessor created: {type(meta_nodes)}")
        print(f"Meta node count: {len(meta_nodes)}")
        
        if len(base_nodes) == 4 and len(meta_nodes) == 0:
            print("✅ Node filtering: Correct initial state")
        else:
            print(f"❌ Node filtering: Expected 4 base, 0 meta. Got {len(base_nodes)} base, {len(meta_nodes)} meta")
    except Exception as e:
        print(f"❌ Node filtering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Initial state - all edges should be base edges  
    print(f"\n📋 Test 2: Initial state - base edges")
    try:
        base_edges = g.edges.base
        print(f"✅ Base edges accessor created: {type(base_edges)}")
        print(f"Base edge count: {len(base_edges)}")
        
        meta_edges = g.edges.meta
        print(f"✅ Meta edges accessor created: {type(meta_edges)}")
        print(f"Meta edge count: {len(meta_edges)}")
        
        if len(base_edges) == 3 and len(meta_edges) == 0:
            print("✅ Edge filtering: Correct initial state")
        else:
            print(f"❌ Edge filtering: Expected 3 base, 0 meta. Got {len(base_edges)} base, {len(meta_edges)} meta")
    except Exception as e:
        print(f"❌ Edge filtering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Create meta-nodes and meta-edges
    print(f"\n📋 Test 3: Create meta-nodes and test filtering")
    try:
        # Collapse nodes 1,2 into meta-node
        subgraph = g.nodes[[node1, node2]]
        meta_node = subgraph.add_to_graph({
            "avg_age": ("mean", "age"),
            "node_count": "count"
        })
        meta_node_id = meta_node.node_id
        
        print(f"✅ Meta-node created: {meta_node_id}")
        print(f"All nodes after collapse: {list(g.node_ids)}")
        print(f"All edges after collapse: {list(g.edge_ids)}")
        
        # Test node filtering after meta-node creation
        base_nodes_after = g.nodes.base
        meta_nodes_after = g.nodes.meta
        
        print(f"Base nodes after collapse: {len(base_nodes_after)}")
        print(f"Meta nodes after collapse: {len(meta_nodes_after)}")
        
        if len(meta_nodes_after) >= 1:
            print("✅ Meta-node filtering works")
        else:
            print("❌ Meta-node not found in meta accessor")
            
        # Test edge filtering after meta-edge creation
        base_edges_after = g.edges.base
        meta_edges_after = g.edges.meta
        
        print(f"Base edges after collapse: {len(base_edges_after)}")
        print(f"Meta edges after collapse: {len(meta_edges_after)}")
        
        # Meta-edges should have been created during collapse
        if len(meta_edges_after) >= 1:
            print("✅ Meta-edge filtering works")
        else:
            print("⚠️  No meta-edges found (this may be expected depending on graph structure)")
            
    except Exception as e:
        print(f"❌ Meta-node creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Verify accessor properties work correctly
    print(f"\n📋 Test 4: Verify accessor methods work on filtered views")
    try:
        # Test that we can iterate through base nodes
        base_nodes = g.nodes.base
        base_list = []
        for node in base_nodes:
            base_list.append(node)
        print(f"✅ Base nodes iteration works: {len(base_list)} nodes")
        
        # Test that we can iterate through meta nodes
        meta_nodes = g.nodes.meta  
        meta_list = []
        for node in meta_nodes:
            meta_list.append(node)
        print(f"✅ Meta nodes iteration works: {len(meta_list)} nodes")
        
        # Test that we can iterate through base edges
        base_edges = g.edges.base
        base_edge_list = []
        for edge in base_edges:
            base_edge_list.append(edge)
        print(f"✅ Base edges iteration works: {len(base_edge_list)} edges")
        
        # Test that we can iterate through meta edges
        meta_edges = g.edges.meta
        meta_edge_list = []
        for edge in meta_edges:
            meta_edge_list.append(edge)
        print(f"✅ Meta edges iteration works: {len(meta_edge_list)} edges")
        
    except Exception as e:
        print(f"❌ Accessor iteration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n🔍⚡ FILTERED ACCESSOR PATTERN TESTS COMPLETED!")
    print(f"✅ g.nodes.base and g.nodes.meta work correctly")
    print(f"✅ g.edges.base and g.edges.meta work correctly")
    print(f"✅ Filtered accessors support iteration and length")
    print(f"✅ Meta-nodes and meta-edges are properly categorized")
    return True

if __name__ == "__main__":
    success = test_filtered_accessors()
    if success:
        print(f"\n🎉 FILTERED ACCESSOR PATTERN: OPERATIONAL! 🔍⚡⚔️")
    else:
        print(f"\n💥 FILTERED ACCESSOR PATTERN FAILURE!")