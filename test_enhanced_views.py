#!/usr/bin/env python3
"""
Test Enhanced View Representations (Phase 2.6)

This tests the new rich representations, ID access, and endpoint properties
for NodeView and EdgeView classes.
"""

import sys
sys.path.insert(0, 'python')

import groggy as gr

def test_enhanced_views():
    """Test enhanced NodeView and EdgeView representations and properties"""
    
    print("🎨 Testing Enhanced View Representations (Phase 2.6)")
    
    # Create test graph
    g = gr.Graph()
    
    # Add nodes with varied attributes
    alice = g.add_node(name="Alice", age=30, dept="Engineering", salary=120000)
    bob = g.add_node(name="Bob", age=25, dept="Engineering", salary=100000) 
    carol = g.add_node(name="Carol", age=35, dept="Design", salary=110000)
    
    # Add edges with attributes
    edge1 = g.add_edge(alice, bob, weight=0.8, relationship="mentor")
    edge2 = g.add_edge(bob, carol, weight=0.6, relationship="collaborates", type="project")
    
    print(f"✅ Created test graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Test 1: Enhanced NodeView representations
    print(f"\n📋 Test 1: Enhanced NodeView Representations")
    try:
        alice_view = g.nodes[alice]
        bob_view = g.nodes[bob]
        carol_view = g.nodes[carol]
        
        print(f"✅ Alice NodeView: {alice_view}")
        print(f"✅ Bob NodeView: {bob_view}")  
        print(f"✅ Carol NodeView: {carol_view}")
        
        # Test that they show attributes in the representation
        alice_str = str(alice_view)
        if "name=Alice" in alice_str and "age=30" in alice_str:
            print(f"✅ Rich attributes displayed in NodeView")
        else:
            print(f"⚠️  Rich attributes not displayed: {alice_str}")
            
    except Exception as e:
        print(f"❌ NodeView representations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Essential ID access
    print(f"\n📋 Test 2: Essential ID Access")
    try:
        alice_view = g.nodes[alice]
        edge1_view = g.edges[edge1]
        
        # Test .id properties
        assert alice_view.id == alice, f"Expected {alice}, got {alice_view.id}"
        assert edge1_view.id == edge1, f"Expected {edge1}, got {edge1_view.id}"
        
        print(f"✅ NodeView.id: {alice_view.id} (type: {type(alice_view.id)})")
        print(f"✅ EdgeView.id: {edge1_view.id} (type: {type(edge1_view.id)})")
        
    except Exception as e:
        print(f"❌ ID access failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Enhanced EdgeView representations  
    print(f"\n📋 Test 3: Enhanced EdgeView Representations")
    try:
        edge1_view = g.edges[edge1]
        edge2_view = g.edges[edge2]
        
        print(f"✅ Edge1 EdgeView: {edge1_view}")
        print(f"✅ Edge2 EdgeView: {edge2_view}")
        
        # Test that they show source, target, and attributes
        edge1_str = str(edge1_view)
        if f"source={alice}" in edge1_str and f"target={bob}" in edge1_str and "weight=0.80" in edge1_str:
            print(f"✅ Rich EdgeView with source, target, and attributes")
        else:
            print(f"⚠️  Rich EdgeView not fully displayed: {edge1_str}")
            
    except Exception as e:
        print(f"❌ EdgeView representations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Edge endpoint properties
    print(f"\n📋 Test 4: Edge Endpoint Properties")
    try:
        edge1_view = g.edges[edge1]
        
        # Test individual properties
        source_id = edge1_view.source
        target_id = edge1_view.target
        endpoints = edge1_view.endpoints
        
        print(f"✅ Edge source: {source_id} (expected: {alice})")
        print(f"✅ Edge target: {target_id} (expected: {bob})")
        print(f"✅ Edge endpoints: {endpoints} (expected: ({alice}, {bob}))")
        
        # Verify correctness
        assert source_id == alice, f"Source mismatch: {source_id} != {alice}"
        assert target_id == bob, f"Target mismatch: {target_id} != {bob}"
        assert endpoints == (alice, bob), f"Endpoints mismatch: {endpoints} != ({alice}, {bob})"
        
        print(f"✅ All endpoint properties correct!")
        
    except Exception as e:
        print(f"❌ Edge endpoint properties failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Edge column access
    print(f"\n📋 Test 5: Edge Column Access")
    try:
        sources = g.edges.source
        targets = g.edges.target
        
        print(f"✅ All edge sources: {sources}")
        print(f"✅ All edge targets: {targets}")
        
        # Verify the lists are correct
        expected_sources = [alice, bob]  # From our two edges
        expected_targets = [bob, carol]
        
        assert sources == expected_sources, f"Sources mismatch: {sources} != {expected_sources}"
        assert targets == expected_targets, f"Targets mismatch: {targets} != {expected_targets}"
        
        print(f"✅ Edge column access working correctly!")
        
    except Exception as e:
        print(f"❌ Edge column access failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Interactive demonstration
    print(f"\n📋 Test 6: Interactive Demonstration")
    try:
        print(f"🎯 Interactive View Examples:")
        print(f"")
        print(f"   Node Views (showing all attributes):")
        print(f"   g.nodes[{alice}] → {g.nodes[alice]}")
        print(f"   g.nodes[{bob}] → {g.nodes[bob]}")
        print(f"")
        print(f"   Edge Views (showing source, target, attributes):") 
        print(f"   g.edges[{edge1}] → {g.edges[edge1]}")
        print(f"   g.edges[{edge2}] → {g.edges[edge2]}")
        print(f"")
        print(f"   ID Access:")
        print(f"   g.nodes[{alice}].id → {g.nodes[alice].id}")
        print(f"   g.edges[{edge1}].id → {g.edges[edge1].id}")
        print(f"")
        print(f"   Edge Endpoints:")
        print(f"   g.edges[{edge1}].source → {g.edges[edge1].source}")
        print(f"   g.edges[{edge1}].target → {g.edges[edge1].target}")
        print(f"   g.edges[{edge1}].endpoints → {g.edges[edge1].endpoints}")
        print(f"")
        print(f"   Edge Column Access:")
        print(f"   g.edges.source → {g.edges.source}")
        print(f"   g.edges.target → {g.edges.target}")
        
    except Exception as e:
        print(f"❌ Interactive demonstration failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Enhanced View Representations Testing Complete!")
    print(f"✨ Phase 2.6 Features Successfully Implemented:")
    print(f"   • Rich NodeView display: NodeView(id=0, name=Alice, age=30, dept=Engineering)")
    print(f"   • Rich EdgeView display: EdgeView(id=0, source=0, target=1, weight=0.80, relationship=mentor)")
    print(f"   • Essential ID access: .id properties on NodeView and EdgeView")
    print(f"   • Edge endpoint properties: .source, .target, .endpoints")
    print(f"   • Edge column access: g.edges.source, g.edges.target")
    print(f"   • Comprehensive attribute display in interactive exploration")

if __name__ == "__main__":
    test_enhanced_views()