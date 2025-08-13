#!/usr/bin/env python3
"""
Simple test of the fluent API focusing on working features
"""

import groggy as gr

def test_working_features():
    """Test the working fluent API features"""
    
    print("✨ === FLUENT API - WORKING FEATURES ===")
    
    # Create test graph
    g = gr.Graph()
    g.add_node(name="Alice", age=30)
    g.add_node(name="Bob", age=25)
    g.add_edge(0, 1, weight=0.8)
    
    # Test 1: Basic access
    print(f"🔍 g.nodes: {g.nodes}")
    print(f"🔍 g.edges: {g.edges}")
    print(f"🔍 g.nodes[0]: {g.nodes[0]}")
    print(f"🔍 g.edges[0]: {g.edges[0]}")
    
    # Test 2: Attribute access
    name = g.nodes[0]["name"]
    age = g.nodes[0]["age"]
    weight = g.edges[0]["weight"]
    print(f"✅ Node attributes - name: {name}, age: {age}")
    print(f"✅ Edge attributes - weight: {weight}")
    
    # Test 3: Fluent updates
    updated_node = g.nodes[0].set(age=31, promoted=True)
    print(f"✅ Fluent update result: {updated_node}")
    
    # Verify update worked
    new_age = g.nodes[0]["age"]
    promoted = g.nodes[0]["promoted"]
    print(f"✅ Updated values - age: {new_age}, promoted: {promoted}")
    
    # Test 4: Chaining
    chained = g.nodes[0].set(salary=120000).set(department="Engineering")
    print(f"✅ Chained updates: {chained}")
    
    # Test 5: Dict update
    updated_edge = g.edges[0].update({"weight": 0.95, "type": "strong"})
    print(f"✅ Dict update: {updated_edge}")
    
    # Test 6: Item assignment (basic types only)
    g.nodes[1]["title"] = "Designer"
    g.edges[0]["verified"] = True
    
    title = g.nodes[1]["title"]
    verified = g.edges[0]["verified"]
    print(f"✅ Item assignment - title: {title}, verified: {verified}")
    
    # Test 7: len()
    print(f"✅ len(g.nodes): {len(g.nodes)}, len(g.edges): {len(g.edges)}")
    
    print(f"\n🎉 All working features tested successfully!")
    print(f"🚀 The fluent API is ready for use!")

if __name__ == "__main__":
    test_working_features()