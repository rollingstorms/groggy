#!/usr/bin/env python3
"""
Test Enhanced Aggregation Syntax with Three Forms 🚀
"""

import groggy

def test_enhanced_aggregation():
    print("🚀 Testing ENHANCED AGGREGATION SYNTAX with Three Forms...")
    
    g = groggy.Graph()
    
    # Create test data
    node1 = g.add_node(name="Alice", age=25, salary=50000)
    node2 = g.add_node(name="Bob", age=30, salary=60000)
    node3 = g.add_node(name="Charlie", age=35, salary=70000)
    
    subgraph = g.nodes[[node1, node2, node3]]
    
    print(f"Created subgraph with nodes: {[node1, node2, node3]}")
    
    # Test Form 1: Simple string mapping
    print(f"\n📋 Test 1: Simple string mapping - {{'age': 'mean', 'salary': 'sum'}}")
    try:
        meta_node1 = subgraph.add_to_graph({"age": "mean", "salary": "sum"})
        print(f"✅ Form 1 meta-node created: {meta_node1}")
        
        # Check attributes
        all_nodes = g.nodes[list(g.node_ids)]
        age_mean = all_nodes.get_node_attribute(meta_node1, "age")
        salary_sum = all_nodes.get_node_attribute(meta_node1, "salary")
        entity_type = all_nodes.get_node_attribute(meta_node1, "entity_type")
        
        print(f"  age (mean): {age_mean}")
        print(f"  salary (sum): {salary_sum}")
        print(f"  entity_type: {entity_type}")
        
        assert abs(age_mean - 30.0) < 0.01, f"Expected age mean ~30, got {age_mean}"
        assert salary_sum == 180000, f"Expected salary sum 180000, got {salary_sum}"
        assert entity_type == "meta", f"Expected entity_type='meta', got {entity_type}"
        
    except Exception as e:
        print(f"❌ Form 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Form 2: Tuple mapping with custom names
    print(f"\n📋 Test 2: Tuple mapping - {{'avg_age': ('mean', 'age'), 'total_salary': ('sum', 'salary')}}")
    try:
        # Create new subgraph for second test
        subgraph2 = g.nodes[[node1, node2]]
        meta_node2 = subgraph2.add_to_graph({
            "avg_age": ("mean", "age"),
            "total_salary": ("sum", "salary"),
            "person_count": ("count", None)
        })
        print(f"✅ Form 2 meta-node created: {meta_node2}")
        
        # Check attributes
        avg_age = all_nodes.get_node_attribute(meta_node2, "avg_age")
        total_salary = all_nodes.get_node_attribute(meta_node2, "total_salary")
        person_count = all_nodes.get_node_attribute(meta_node2, "person_count")
        entity_type = all_nodes.get_node_attribute(meta_node2, "entity_type")
        
        print(f"  avg_age: {avg_age}")
        print(f"  total_salary: {total_salary}")
        print(f"  person_count: {person_count}")
        print(f"  entity_type: {entity_type}")
        
        assert abs(avg_age - 27.5) < 0.01, f"Expected avg_age ~27.5, got {avg_age}"
        assert total_salary == 110000, f"Expected total_salary 110000, got {total_salary}"
        assert person_count == 2, f"Expected person_count 2, got {person_count}"
        assert entity_type == "meta", f"Expected entity_type='meta', got {entity_type}"
        
    except Exception as e:
        print(f"❌ Form 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Form 3: Dict-of-dicts with defaults
    print(f"\n📋 Test 3: Dict-of-dicts with defaults")
    try:
        # Create new subgraph for third test
        subgraph3 = g.nodes[[node2, node3]]
        meta_node3 = subgraph3.add_to_graph({
            "avg_age": {"func": "mean", "source": "age"},
            "total_salary": {"func": "sum", "source": "salary"},
            "bonus": {"func": "sum", "source": "bonus", "default": 5000}  # bonus doesn't exist, uses default
        })
        print(f"✅ Form 3 meta-node created: {meta_node3}")
        
        # Check attributes
        avg_age = all_nodes.get_node_attribute(meta_node3, "avg_age")
        total_salary = all_nodes.get_node_attribute(meta_node3, "total_salary")
        bonus = all_nodes.get_node_attribute(meta_node3, "bonus")
        entity_type = all_nodes.get_node_attribute(meta_node3, "entity_type")
        
        print(f"  avg_age: {avg_age}")
        print(f"  total_salary: {total_salary}")
        print(f"  bonus (with default): {bonus}")
        print(f"  entity_type: {entity_type}")
        
        assert abs(avg_age - 32.5) < 0.01, f"Expected avg_age ~32.5, got {avg_age}"
        assert total_salary == 130000, f"Expected total_salary 130000, got {total_salary}"
        assert bonus == 10000, f"Expected bonus 10000 (2*5000), got {bonus}"  # 2 nodes * 5000 default
        assert entity_type == "meta", f"Expected entity_type='meta', got {entity_type}"
        
    except Exception as e:
        print(f"❌ Form 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n🚀 ALL ENHANCED AGGREGATION SYNTAX TESTS PASSED!")
    print(f"✅ Form 1: Simple string mapping")
    print(f"✅ Form 2: Tuple mapping with custom names")
    print(f"✅ Form 3: Dict-of-dicts with defaults")
    return True

if __name__ == "__main__":
    success = test_enhanced_aggregation()
    if success:
        print(f"\n🎉 ENHANCED AGGREGATION SYNTAX: OPERATIONAL! 🚀⚡")
    else:
        print(f"\n💥 AGGREGATION SYNTAX FAILURE!")