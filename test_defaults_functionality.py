#!/usr/bin/env python3
"""
Test defaults functionality for enhanced missing attribute handling.
"""

import groggy

def test_defaults_functionality():
    print("🧪 Testing Enhanced Missing Attribute Handling with Defaults...")
    
    g = groggy.Graph()
    
    # Create nodes with some attributes but missing others
    node1 = g.add_node(name="Alice", salary=75000)  # Missing: bonus, rating
    node2 = g.add_node(name="Bob", salary=85000)    # Missing: bonus, rating
    
    print(f"Created nodes: {[node1, node2]}")
    
    # Test 1: Collapse with defaults (advanced usage)
    print(f"\n📋 Test 1: Using defaults for missing attributes")
    subgraph = g.nodes[[node1, node2]]
    
    try:
        meta_node = subgraph.collapse_to_node_with_defaults(
            agg_functions={
                "person_count": "count",    # Always works
                "salary": "sum",            # Should work (exists) 
                "bonus": "sum",             # Should use default (missing)
                "rating": "mean",           # Should use default (missing)
            },
            defaults={
                "bonus": 5000,              # Default bonus value
                "rating": 3.5               # Default rating value
            }
        )
        
        # Verify results
        all_nodes = g.nodes[list(g.node_ids)]
        person_count = all_nodes.get_node_attribute(meta_node, "person_count")
        total_salary = all_nodes.get_node_attribute(meta_node, "salary")
        bonus_sum = all_nodes.get_node_attribute(meta_node, "bonus") 
        avg_rating = all_nodes.get_node_attribute(meta_node, "rating")
        
        print(f"✅ Meta-node created: {meta_node}")
        print(f"  person_count: {person_count} (expected: 2)")
        print(f"  total_salary: {total_salary} (expected: 160000)")
        print(f"  bonus_sum: {bonus_sum} (expected: 5000 - default value)")
        print(f"  avg_rating: {avg_rating} (expected: 3.5 - default value)")
        
        # Verify results
        assert person_count == 2, f"Expected person_count=2, got {person_count}"
        assert total_salary == 160000, f"Expected total_salary=160000, got {total_salary}"
        assert bonus_sum == 5000, f"Expected bonus_sum=5000, got {bonus_sum}"
        assert avg_rating == 3.5, f"Expected avg_rating=3.5, got {avg_rating}"
        
        print("✅ All defaults functionality working correctly!")
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Error when missing attribute has no default
    print(f"\n📋 Test 2: Error when no default provided for missing attribute")
    
    try:
        # This should error because 'department' doesn't exist and no default provided
        meta_node2 = subgraph.collapse_to_node_with_defaults(
            agg_functions={
                "salary": "sum",
                "department": "first"    # Missing with no default - should ERROR
            },
            defaults={}  # Empty defaults
        )
        print(f"❌ UNEXPECTED: No error for missing 'department' attribute")
        return False
    except Exception as e:
        print(f"✅ EXPECTED ERROR: {e}")
        assert "department" in str(e), f"Error should mention 'department', got: {e}"
    
    print(f"\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_defaults_functionality()
    if success:
        print("\n🎉 DEFAULTS FUNCTIONALITY TEST PASSED!")
    else:
        print("\n❌ TEST FAILED!")
