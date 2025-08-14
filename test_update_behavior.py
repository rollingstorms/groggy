#!/usr/bin/env python3
"""
Test .update() method behavior - should work for existing attributes like Python dict.update()
"""

import groggy as gr

def test_update_behavior():
    """Test .update() method behavior with existing and new attributes"""
    
    print("🔧 === TESTING .update() METHOD BEHAVIOR ===")
    
    # Create a test graph
    g = gr.Graph()
    g.add_node(name="Alice", age=30, role="Engineer")
    
    print(f"✅ Created node with initial attributes:")
    print(f"   name: {g.nodes[0]['name']}")
    print(f"   age: {g.nodes[0]['age']}")
    print(f"   role: {g.nodes[0]['role']}")
    
    # Test 1: .set() with existing attributes (should always work)
    print(f"\n✅ === Test 1: .set() with existing attributes ===")
    try:
        g.nodes[0].set(age=31)  # Update existing attribute
        print(f"✅ .set(age=31) worked: {g.nodes[0]['age']}")
        
        g.nodes[0].set(name="Alice Updated")  # Update existing attribute
        print(f"✅ .set(name='Alice Updated') worked: {g.nodes[0]['name']}")
        
    except Exception as e:
        print(f"❌ .set() error: {e}")
    
    # Test 2: .update() with existing attributes (the problematic case)
    print(f"\n🔧 === Test 2: .update() with existing attributes ===")
    try:
        g.nodes[0].update({"age": 32})  # Update existing attribute
        print(f"✅ .update({{'age': 32}}) worked: {g.nodes[0]['age']}")
        
        g.nodes[0].update({"role": "Senior Engineer"})  # Update existing attribute
        print(f"✅ .update({{'role': 'Senior Engineer'}}) worked: {g.nodes[0]['role']}")
        
    except Exception as e:
        print(f"❌ .update() with existing attributes error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: .update() with new attributes
    print(f"\n🆕 === Test 3: .update() with new attributes ===")
    try:
        g.nodes[0].update({"salary": 120000})  # New attribute
        print(f"✅ .update({{'salary': 120000}}) worked: {g.nodes[0]['salary']}")
        
        g.nodes[0].update({"department": "Engineering"})  # New attribute
        print(f"✅ .update({{'department': 'Engineering'}}) worked: {g.nodes[0]['department']}")
        
    except Exception as e:
        print(f"❌ .update() with new attributes error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: .update() with mixed existing and new attributes
    print(f"\n🔀 === Test 4: .update() with mixed existing and new attributes ===")
    try:
        g.nodes[0].update({"age": 33, "bonus": 5000})  # Mix of existing and new
        print(f"✅ .update() mixed worked:")
        print(f"   age (existing): {g.nodes[0]['age']}")
        print(f"   bonus (new): {g.nodes[0]['bonus']}")
        
    except Exception as e:
        print(f"❌ .update() with mixed attributes error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Compare with Python dict behavior
    print(f"\n🐍 === Test 5: Python dict behavior (reference) ===")
    test_dict = {"name": "Alice", "age": 30, "role": "Engineer"}
    print(f"Initial dict: {test_dict}")
    
    test_dict.update({"age": 31})  # Update existing
    print(f"After update existing: {test_dict}")
    
    test_dict.update({"salary": 120000})  # Add new
    print(f"After update new: {test_dict}")
    
    test_dict.update({"age": 32, "bonus": 5000})  # Mix
    print(f"After update mixed: {test_dict}")
    
    print(f"\n🎯 Summary:")
    print(f"Python dict.update() works for existing, new, and mixed attributes.")
    print(f"Groggy .update() should have the same behavior!")

if __name__ == "__main__":
    test_update_behavior()