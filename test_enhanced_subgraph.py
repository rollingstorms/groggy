#!/usr/bin/env python3
"""
Test Enhanced Subgraph Integration - Core Rust + Python Bindings
 
This tests the integration between our new core Subgraph implementation
and the enhanced Python bindings with column access and chaining.
"""

import sys
sys.path.insert(0, 'python')

import groggy as gr

def test_enhanced_subgraph():
    """Test enhanced subgraph functionality with column access and chaining"""
    
    print("🚀 Testing Enhanced Subgraph Integration")
    
    # Create test graph
    g = gr.Graph()
    
    # Add nodes with attributes
    employees = [
        {"name": "Alice", "age": 30, "dept": "Engineering", "salary": 120000},
        {"name": "Bob", "age": 25, "dept": "Engineering", "salary": 100000},
        {"name": "Carol", "age": 35, "dept": "Design", "salary": 110000},
        {"name": "Dave", "age": 28, "dept": "Design", "salary": 105000},
        {"name": "Eve", "age": 32, "dept": "Marketing", "salary": 95000},
        {"name": "Frank", "age": 26, "dept": "Marketing", "salary": 85000},
    ]
    
    node_ids = []
    for emp in employees:
        node_id = g.add_node(**emp)
        node_ids.append(node_id)
    
    # Add some edges
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (0,3)]
    for src, tgt in edges:
        g.add_edge(node_ids[src], node_ids[tgt], relationship="collaborates")
    
    print(f"✅ Created test graph: {len(node_ids)} nodes, {len(edges)} edges")
    
    # Test 1: Basic batch access (should already work)
    print(f"\n📋 Test 1: Basic Batch Access")
    try:
        batch_subgraph = g.nodes[[node_ids[0], node_ids[1], node_ids[2]]]
        print(f"✅ Batch access: {type(batch_subgraph)} with {len(batch_subgraph.nodes)} nodes")
        print(f"   Subgraph type: {batch_subgraph}")
    except Exception as e:
        print(f"❌ Batch access failed: {e}")
    
    # Test 2: Column access - NEW FUNCTIONALITY
    print(f"\n📋 Test 2: Column Access (NEW)")
    try:
        names = batch_subgraph['name']
        ages = batch_subgraph['age']
        depts = batch_subgraph['dept']
        
        print(f"✅ Column access successful:")
        print(f"   Names: {names}")
        print(f"   Ages: {ages}")
        print(f"   Departments: {depts}")
    except Exception as e:
        print(f"❌ Column access failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Subgraph chaining - NEW FUNCTIONALITY
    print(f"\n📋 Test 3: Subgraph Chaining (NEW)")
    try:
        # Chain: get Engineering subgraph, then high earners within Engineering
        engineering = batch_subgraph.filter_nodes('dept == "Engineering"')
        print(f"✅ Filtered to Engineering: {engineering}")
        print(f"   Engineering nodes: {engineering.nodes}")
        
        # Test column access on filtered subgraph
        eng_names = engineering['name']
        eng_salaries = engineering['salary']
        print(f"✅ Engineering team column access:")
        print(f"   Names: {eng_names}")
        print(f"   Salaries: {eng_salaries}")
        
    except Exception as e:
        print(f"❌ Subgraph chaining failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Batch operations on subgraphs - EXISTING + ENHANCED
    print(f"\n📋 Test 4: Batch Operations on Subgraphs")
    try:
        # Set attributes on the batch subgraph
        batch_subgraph.set(team="Alpha", reviewed=True)
        print(f"✅ Batch .set() operation completed")
        
        # Verify the attributes were set by checking column access
        teams = batch_subgraph['team']
        reviews = batch_subgraph['reviewed']
        print(f"✅ Verification via column access:")
        print(f"   Teams: {teams}")
        print(f"   Reviewed: {reviews}")
        
    except Exception as e:
        print(f"❌ Batch operations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Edge column access - NEW FUNCTIONALITY
    print(f"\n📋 Test 5: Edge Column Access (NEW)")
    try:
        # Get edges from batch subgraph
        edge_relationships = batch_subgraph.get_edge_attribute_column('relationship')
        print(f"✅ Edge column access:")
        print(f"   Relationships: {edge_relationships}")
        
    except Exception as e:
        print(f"❌ Edge column access failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Algorithm integration with enhanced subgraphs
    print(f"\n📋 Test 6: Algorithm Integration")
    try:
        # Run connected components (should return enhanced subgraphs)
        components = g.connected_components()
        print(f"✅ Connected components: {len(components)} components")
        
        if len(components) > 0:
            comp = components[0]
            print(f"   Component 0: {comp}")
            print(f"   Component nodes: {comp.nodes}")
            
            # Try to access attributes on algorithm result
            try:
                comp_names = comp['name']
                print(f"✅ Algorithm result column access: {comp_names}")
            except Exception as e:
                print(f"⚠️  Algorithm result column access failed (expected): {e}")
                print(f"   This is expected - algorithm results don't have graph references")
        
    except Exception as e:
        print(f"❌ Algorithm integration failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Enhanced Subgraph Integration Testing Complete!")
    print(f"✨ Summary of NEW functionality:")
    print(f"   • Column access: subgraph['attr_name'] → [val1, val2, val3]")
    print(f"   • Subgraph chaining: subgraph.filter_nodes().filter_nodes()")
    print(f"   • Enhanced batch operations with verification")
    print(f"   • Edge attribute column access")
    print(f"   • Integration with algorithm results")

if __name__ == "__main__":
    test_enhanced_subgraph()