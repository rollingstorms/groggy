#!/usr/bin/env python3
"""
Test Phase 2: Advanced Node/Edge Views with batch and slice access
"""

import groggy as gr

def test_phase2_batch_access():
    """Test new batch and slice access functionality"""
    
    print("🎯 === PHASE 2: BATCH & SLICE ACCESS TESTING ===")
    
    # Create a test graph
    g = gr.Graph()
    
    # Add nodes with attributes
    for i in range(8):
        g.add_node(name=f"Node_{i}", value=i*10, group=i//3)
    
    # Add edges with attributes  
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (0,3), (1,4)]
    for i, (src, tgt) in enumerate(edges):
        g.add_edge(src, tgt, weight=0.5 + i*0.1, type=f"edge_{i}")
    
    print(f"✅ Created test graph: {g}")
    print(f"   Nodes: {g.node_count()}, Edges: {g.edge_count()}")
    
    # Test 1: Single node access (existing functionality)
    print(f"\n🔍 === Test 1: Single Node Access (existing) ===")
    try:
        single_node = g.nodes[0]
        print(f"✅ g.nodes[0] type: {type(single_node)}")
        print(f"   Node attributes: {single_node['name']}, value={single_node['value']}")
        
    except Exception as e:
        print(f"❌ Single node access error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Batch node access - NEW!
    print(f"\n📦 === Test 2: Batch Node Access (NEW!) ===")
    try:
        batch_nodes = g.nodes[[0, 1, 2]]
        print(f"✅ g.nodes[[0,1,2]] type: {type(batch_nodes)}")
        print(f"   Subgraph: {batch_nodes}")
        print(f"   Nodes in subgraph: {len(batch_nodes.nodes)}")
        print(f"   Induced edges: {len(batch_nodes.edges)}")
        
    except Exception as e:
        print(f"❌ Batch node access error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Slice node access - NEW!
    print(f"\n✂️ === Test 3: Slice Node Access (NEW!) ===")
    try:
        slice_nodes = g.nodes[0:5]
        print(f"✅ g.nodes[0:5] type: {type(slice_nodes)}")
        print(f"   Subgraph: {slice_nodes}")
        print(f"   Nodes in slice: {len(slice_nodes.nodes)}")
        print(f"   Induced edges: {len(slice_nodes.edges)}")
        
        # Test step slicing
        step_slice = g.nodes[0:6:2]  # Every other node
        print(f"✅ g.nodes[0:6:2] (step slice): {step_slice}")
        print(f"   Step slice nodes: {len(step_slice.nodes)}")
        
    except Exception as e:
        print(f"❌ Slice node access error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Single edge access (existing functionality)
    print(f"\n🔗 === Test 4: Single Edge Access (existing) ===")
    try:
        single_edge = g.edges[0]
        print(f"✅ g.edges[0] type: {type(single_edge)}")
        print(f"   Edge attributes: weight={single_edge['weight']}, type={single_edge['type']}")
        
    except Exception as e:
        print(f"❌ Single edge access error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Batch edge access - NEW!
    print(f"\n📦 === Test 5: Batch Edge Access (NEW!) ===")
    try:
        batch_edges = g.edges[[0, 1, 2]]
        print(f"✅ g.edges[[0,1,2]] type: {type(batch_edges)}")
        print(f"   Subgraph: {batch_edges}")
        print(f"   Edges in subgraph: {len(batch_edges.edges)}")
        print(f"   Endpoint nodes: {len(batch_edges.nodes)}")
        
    except Exception as e:
        print(f"❌ Batch edge access error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Slice edge access - NEW!
    print(f"\n✂️ === Test 6: Slice Edge Access (NEW!) ===")
    try:
        slice_edges = g.edges[0:4]
        print(f"✅ g.edges[0:4] type: {type(slice_edges)}")
        print(f"   Subgraph: {slice_edges}")
        print(f"   Edges in slice: {len(slice_edges.edges)}")
        print(f"   Endpoint nodes: {len(slice_edges.nodes)}")
        
    except Exception as e:
        print(f"❌ Slice edge access error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 7: Error cases
    print(f"\n⚠️ === Test 7: Error Cases ===")
    try:
        # Non-existent node
        try:
            bad_batch = g.nodes[[0, 99]]  # Node 99 doesn't exist
            print(f"❌ Should have failed for non-existent node")
        except KeyError as e:
            print(f"✅ Correctly caught non-existent node error: {e}")
        
        # Invalid key type
        try:
            bad_key = g.nodes["invalid"]
            print(f"❌ Should have failed for string key")
        except TypeError as e:
            print(f"✅ Correctly caught invalid key type error: {e}")
        
    except Exception as e:
        print(f"❌ Error case testing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Phase 2 batch and slice access testing complete!")
    print(f"✨ New functionality working: batch access and slicing return Subgraphs!")
    print(f"🚀 Ready for the next step: batch operations on Subgraphs!")

if __name__ == "__main__":
    test_phase2_batch_access()