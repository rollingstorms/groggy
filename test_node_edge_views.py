#!/usr/bin/env python3
"""
Test the new NodeView and EdgeView fluent API
"""

import groggy as gr

def test_fluent_node_edge_views():
    """Test the new fluent NodeView and EdgeView API"""
    
    print("🎯 === TESTING FLUENT NODE/EDGE VIEWS ===")
    
    # Create a test graph
    g = gr.Graph()
    
    # Add some nodes
    g.add_node(name="Alice", age=30, role="Engineer")
    g.add_node(name="Bob", age=25, role="Designer") 
    g.add_node(name="Carol", age=35, role="Manager")
    
    # Add some edges
    g.add_edge(0, 1, weight=0.8, type="collaborates")
    g.add_edge(1, 2, weight=0.9, type="reports_to")
    
    print(f"✅ Created test graph: {g}")
    
    # Test 1: Basic node access
    print(f"\n🔍 === Test 1: Basic Node Access ===")
    try:
        nodes_accessor = g.nodes
        print(f"✅ g.nodes returns: {nodes_accessor}")
        print(f"   Type: {type(nodes_accessor)}")
        
        # Test single node access
        node_view = g.nodes[0]  
        print(f"✅ g.nodes[0] returns: {node_view}")
        print(f"   Type: {type(node_view)}")
        
        # Test attribute access
        name = node_view["name"]
        print(f"✅ node_view['name'] = {name}")
        
    except Exception as e:
        print(f"❌ Basic node access error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Basic edge access
    print(f"\n🔗 === Test 2: Basic Edge Access ===")
    try:
        edges_accessor = g.edges
        print(f"✅ g.edges returns: {edges_accessor}")
        print(f"   Type: {type(edges_accessor)}")
        
        # Test single edge access
        edge_view = g.edges[0]
        print(f"✅ g.edges[0] returns: {edge_view}")
        print(f"   Type: {type(edge_view)}")
        
        # Test attribute access
        weight = edge_view["weight"]
        print(f"✅ edge_view['weight'] = {weight}")
        
    except Exception as e:
        print(f"❌ Basic edge access error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Fluent attribute updates
    print(f"\n✨ === Test 3: Fluent Attribute Updates ===")
    try:
        # Test node fluent updates
        node = g.nodes[0]
        print(f"Before update: {node['name']}, age={node['age']}")
        
        # Test .set() with kwargs
        updated_node = node.set(age=31, promoted=True)
        print(f"✅ node.set(age=31, promoted=True) succeeded")
        print(f"   Type returned: {type(updated_node)}")
        
        # Test chaining
        chain_result = g.nodes[0].set(salary=120000).set(department="Engineering")
        print(f"✅ Chained updates succeeded: {type(chain_result)}")
        
        # Verify the updates worked
        final_age = g.nodes[0]["age"]
        promoted = g.nodes[0]["promoted"]
        print(f"✅ Verification - age: {final_age}, promoted: {promoted}")
        
    except Exception as e:
        print(f"❌ Fluent updates error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Dict-based updates
    print(f"\n📝 === Test 4: Dict-based Updates ===")
    try:
        # Test .update() with dict
        edge = g.edges[0]
        print(f"Before update: weight={edge['weight']}")
        
        updated_edge = edge.update({"weight": 0.95, "last_interaction": "2024-01-15"})
        print(f"✅ edge.update() succeeded: {type(updated_edge)}")
        
        # Verify updates
        new_weight = g.edges[0]["weight"]
        last_interaction = g.edges[0]["last_interaction"]
        print(f"✅ Verification - weight: {new_weight}, last_interaction: {last_interaction}")
        
    except Exception as e:
        print(f"❌ Dict updates error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Item assignment syntax
    print(f"\n🔧 === Test 5: Item Assignment Syntax ===")
    try:
        # Test node["attr"] = value
        g.nodes[1]["experience"] = "5 years"
        g.nodes[1]["skills"] = ["Python", "Design"]
        
        experience = g.nodes[1]["experience"]
        skills = g.nodes[1]["skills"]
        print(f"✅ Item assignment - experience: {experience}, skills: {skills}")
        
        # Test edge["attr"] = value  
        g.edges[1]["priority"] = "high"
        g.edges[1]["frequency"] = 3.5
        
        priority = g.edges[1]["priority"]
        frequency = g.edges[1]["frequency"]
        print(f"✅ Edge assignment - priority: {priority}, frequency: {frequency}")
        
    except Exception as e:
        print(f"❌ Item assignment error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Iteration and length
    print(f"\n🔄 === Test 6: Iteration and Length ===")
    try:
        # Test len() and iteration
        print(f"✅ len(g.nodes): {len(g.nodes)}")
        print(f"✅ len(g.edges): {len(g.edges)}")
        
        print(f"Node iteration:")
        for i, node_id in enumerate(g.nodes):
            if i < 3:  # Limit output
                print(f"  Node {node_id}: {g.nodes[node_id]['name']}")
        
        print(f"Edge iteration:")
        for i, edge_id in enumerate(g.edges):
            if i < 3:  # Limit output
                print(f"  Edge {edge_id}: weight={g.edges[edge_id]['weight']}")
        
    except Exception as e:
        print(f"❌ Iteration error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Fluent Node/Edge Views testing complete!")

if __name__ == "__main__":
    test_fluent_node_edge_views()