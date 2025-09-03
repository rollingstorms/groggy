#!/usr/bin/env python3
"""
Test enhanced missing attribute handling with defaults.
"""

import groggy

def test_enhanced_missing_attributes():
    print("🧪 Testing Enhanced Missing Attribute Handling...")
    
    g = groggy.Graph()
    
    # Create nodes with some attributes
    node1 = g.add_node(name="Alice", salary=75000, department="engineering")
    node2 = g.add_node(name="Bob", salary=85000, department="engineering")
    
    print(f"Created nodes: {[node1, node2]}")
    
    # Test 1: Strict validation (should error on missing attributes)
    print(f"\n📋 Test 1: Strict validation (error on missing attributes)")
    subgraph = g.nodes[[node1, node2]]
    
    try:
        meta_node = subgraph.collapse_to_node({
            "salary": "sum",     # This should work
            "bonus": "sum"       # This should ERROR (doesn't exist)
        })
        print(f"❌ UNEXPECTED: No error for missing 'bonus' attribute")
    except Exception as e:
        print(f"✅ EXPECTED ERROR: {e}")
        assert "bonus" in str(e), f"Error message should mention 'bonus', got: {e}"
        assert "not found" in str(e), f"Error should mention 'not found', got: {e}"
    
    # Test 2: Using existing attributes only (should work)
    print(f"\n📋 Test 2: Using existing attributes only")
    try:
        meta_node = subgraph.collapse_to_node({
            "person_count": "count",  # Always works
            "salary": "sum",          # Should work (exists)
            "department": "first"     # Should work (exists)
        })
        
        # Verify results
        all_nodes = g.nodes[list(g.node_ids)]
        person_count = all_nodes.get_node_attribute(meta_node, "person_count")
        total_salary = all_nodes.get_node_attribute(meta_node, "salary")
        department = all_nodes.get_node_attribute(meta_node, "department")
        
        print(f"✅ Meta-node created: {meta_node}")
        print(f"  person_count: {person_count} (expected: 2)")
        print(f"  total_salary: {total_salary} (expected: 160000)")
        print(f"  department: {department} (expected: engineering)")
        
        assert person_count == 2, f"Expected person_count=2, got {person_count}"
        assert total_salary == 160000, f"Expected total_salary=160000, got {total_salary}"
        assert department == "engineering", f"Expected department=engineering, got {department}"
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False
    
    print(f"\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_enhanced_missing_attributes()
    if success:
        print("\n🎉 ENHANCED MISSING ATTRIBUTE HANDLING TEST PASSED!")
    else:
        print("\n❌ TEST FAILED!")
