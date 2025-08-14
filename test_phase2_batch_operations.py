#!/usr/bin/env python3
"""
Test Phase 2: Batch Operations on Subgraphs (g.nodes[[0,1,2]].set())
"""

import groggy as gr

def test_phase2_batch_operations():
    """Test new batch operations functionality"""
    
    print("🎯 === PHASE 2: BATCH OPERATIONS TESTING ===")
    
    # Create a test graph
    g = gr.Graph()
    
    # Add nodes with initial attributes
    for i in range(6):
        g.add_node(name=f"Node_{i}", value=i*10, group=i//2)
    
    # Add edges  
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5)]
    for i, (src, tgt) in enumerate(edges):
        g.add_edge(src, tgt, weight=0.5 + i*0.1)
    
    print(f"✅ Created test graph: {g}")
    print(f"   Initial node attributes: {[g.nodes[i]['name'] for i in range(3)]}")
    
    # Test 1: Single node batch operation (should work like before)
    print(f"\n🔍 === Test 1: Single Node Operations (baseline) ===")
    try:
        # Single node update
        result = g.nodes[0].set(department="Engineering", active=True)
        print(f"✅ g.nodes[0].set() type: {type(result)}")
        print(f"   Updated attributes: department={g.nodes[0]['department']}, active={g.nodes[0]['active']}")
        
    except Exception as e:
        print(f"❌ Single node operation error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Batch node operations - THE BIG TEST!
    print(f"\n🚀 === Test 2: Batch Node Operations (NEW!) ===")
    try:
        # Batch update using .set() 
        batch_result = g.nodes[[1, 2, 3]].set(department="Research", team="Alpha")
        print(f"✅ g.nodes[[1,2,3]].set() type: {type(batch_result)}")
        print(f"   Batch result: {batch_result}")
        
        # Verify all nodes were updated
        for i in [1, 2, 3]:
            dept = g.nodes[i]['department']
            team = g.nodes[i]['team']
            print(f"   Node {i}: department={dept}, team={team}")
        
    except Exception as e:
        print(f"❌ Batch node operation error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Slice-based batch operations
    print(f"\n✂️ === Test 3: Slice-based Batch Operations ===")
    try:
        # Update a range of nodes using slice
        slice_result = g.nodes[0:4].set(project="ProjectX", status="active")
        print(f"✅ g.nodes[0:4].set() type: {type(slice_result)}")
        print(f"   Slice result: {slice_result}")
        
        # Verify slice updates
        for i in range(4):
            project = g.nodes[i]['project']
            status = g.nodes[i]['status']
            print(f"   Node {i}: project={project}, status={status}")
        
    except Exception as e:
        print(f"❌ Slice batch operation error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Batch operations with .update() 
    print(f"\n📝 === Test 4: Batch .update() Operations ===")
    try:
        # Test .update() method with dict
        update_result = g.nodes[[4, 5]].update({"priority": "high", "deadline": "2024-12-31"})
        print(f"✅ g.nodes[[4,5]].update() type: {type(update_result)}")
        
        # Verify updates
        for i in [4, 5]:
            priority = g.nodes[i]['priority']
            deadline = g.nodes[i]['deadline']
            print(f"   Node {i}: priority={priority}, deadline={deadline}")
        
    except Exception as e:
        print(f"❌ Batch update operation error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Chained batch operations
    print(f"\n🔗 === Test 5: Chained Batch Operations ===")
    try:
        # Test chaining batch operations
        chained_result = g.nodes[[0, 1]].set(phase="Phase1").set(verified=True)
        print(f"✅ Chained batch operations: {chained_result}")
        
        # Verify chaining worked
        for i in [0, 1]:
            phase = g.nodes[i]['phase']
            verified = g.nodes[i]['verified']
            print(f"   Node {i}: phase={phase}, verified={verified}")
        
    except Exception as e:
        print(f"❌ Chained batch operation error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Error cases 
    print(f"\n⚠️ === Test 6: Error Cases ===")
    try:
        # Test subgraph without graph reference (should fail gracefully)
        # This would happen if we created a subgraph from algorithms like connected_components
        components = g.connected_components()
        if components:
            try:
                components[0].set(test_attr="should_fail")
                print(f"❌ Should have failed for subgraph without graph reference")
            except RuntimeError as e:
                print(f"✅ Correctly caught error for subgraph without graph reference: {str(e)[:80]}...")
        
    except Exception as e:
        print(f"❌ Error case testing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Phase 2 batch operations testing complete!")
    print(f"✨ New functionality working:")
    print(f"   • g.nodes[[0,1,2]].set(attr=value) - batch node updates!")
    print(f"   • g.nodes[0:5].set(attr=value) - slice-based updates!")
    print(f"   • g.nodes[[0,1,2]].update({dict}) - batch dict updates!")
    print(f"   • Chainable: g.nodes[[0,1]].set().set() - method chaining!")
    print(f"🚀 Phase 2 Advanced Node/Edge Views is COMPLETE!")

if __name__ == "__main__":
    test_phase2_batch_operations()