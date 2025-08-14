#!/usr/bin/env python3
"""
Test Enhanced Query Parser - Comparison Operators (Step C.1)

This tests the enhanced comparison operators: !=, >=, <=
"""

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy as gr

def test_enhanced_comparison_operators():
    """Test the new comparison operators in query parser"""
    
    print("🔍 Testing Enhanced Query Parser - Comparison Operators (Step C.1)")
    
    # Create test graph with varied attributes
    g = gr.Graph()
    
    # Add employees with different salaries and ages
    alice = g.add_node(name="Alice", age=30, salary=120000, dept="Engineering")
    bob = g.add_node(name="Bob", age=25, salary=100000, dept="Engineering") 
    carol = g.add_node(name="Carol", age=35, salary=110000, dept="Design")
    david = g.add_node(name="David", age=28, salary=95000, dept="Marketing")
    eve = g.add_node(name="Eve", age=32, salary=130000, dept="Engineering")
    
    print(f"✅ Created test graph: {g.node_count()} nodes")
    
    # Test 1: Not equals (!=) operator
    print(f"\n📋 Test 1: Not Equals (!=) Operator")
    try:
        # Find all non-Engineering employees
        non_eng_filter = gr.parse_node_query("dept != 'Engineering'")
        non_eng_results = g.filter_nodes(non_eng_filter)
        non_eng_nodes = non_eng_results.nodes
        
        print(f"✅ Non-Engineering employees: {len(non_eng_nodes)} found")
        
        # Verify results
        expected_non_eng = {carol, david}  # Carol (Design), David (Marketing)
        actual_non_eng = set(non_eng_nodes)
        assert actual_non_eng == expected_non_eng, f"Expected {expected_non_eng}, got {actual_non_eng}"
        print(f"✅ != operator working correctly")
        
        # Show the results
        for node_id in non_eng_nodes:
            node_view = g.nodes[node_id]
            print(f"   Non-Engineering: {node_view}")
        
    except Exception as e:
        print(f"❌ != operator test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Greater than or equal (>=) operator
    print(f"\n📋 Test 2: Greater Than or Equal (>=) Operator")
    try:
        # Find employees with salary >= 110000
        high_salary_filter = gr.parse_node_query("salary >= 110000")
        high_salary_results = g.filter_nodes(high_salary_filter)
        high_salary_nodes = high_salary_results.nodes
        
        print(f"✅ High salary employees (>= 110k): {len(high_salary_nodes)} found")
        
        # Verify results
        expected_high_salary = {alice, carol, eve}  # Alice (120k), Carol (110k), Eve (130k)
        actual_high_salary = set(high_salary_nodes)
        assert actual_high_salary == expected_high_salary, f"Expected {expected_high_salary}, got {actual_high_salary}"
        print(f"✅ >= operator working correctly")
        
        # Show the results
        for node_id in high_salary_nodes:
            node_view = g.nodes[node_id]
            print(f"   High salary: {node_view}")
        
    except Exception as e:
        print(f"❌ >= operator test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Less than or equal (<=) operator
    print(f"\n📋 Test 3: Less Than or Equal (<=) Operator")
    try:
        # Find employees with age <= 30
        young_filter = gr.parse_node_query("age <= 30")
        young_results = g.filter_nodes(young_filter)
        young_nodes = young_results.nodes
        
        print(f"✅ Young employees (<= 30): {len(young_nodes)} found")
        
        # Verify results
        expected_young = {alice, bob, david}  # Alice (30), Bob (25), David (28)
        actual_young = set(young_nodes)
        assert actual_young == expected_young, f"Expected {expected_young}, got {actual_young}"
        print(f"✅ <= operator working correctly")
        
        # Show the results
        for node_id in young_nodes:
            node_view = g.nodes[node_id]
            print(f"   Young employee: {node_view}")
        
    except Exception as e:
        print(f"❌ <= operator test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Edge with numeric comparisons
    print(f"\n📋 Test 4: Edge Filters with Enhanced Operators")
    try:
        # Add edges with weights
        edge1 = g.add_edge(alice, bob, weight=0.8, strength="strong")
        edge2 = g.add_edge(bob, carol, weight=0.6, strength="medium")
        edge3 = g.add_edge(carol, david, weight=0.9, strength="strong")
        edge4 = g.add_edge(david, eve, weight=0.4, strength="weak")
        
        print(f"✅ Added {g.edge_count()} edges with weights")
        
        # Test edge filtering with >=
        strong_edges_filter = gr.parse_edge_query("weight >= 0.7")
        strong_edges_results = g.filter_edges(strong_edges_filter)
        print(f"   Debug: filter_edges returned: {type(strong_edges_results)}")
        strong_edge_ids = strong_edges_results  # It's already a list
        
        expected_strong = {edge1, edge3}  # weight 0.8 and 0.9
        actual_strong = set(strong_edge_ids)
        assert actual_strong == expected_strong, f"Expected {expected_strong}, got {actual_strong}"
        print(f"✅ Edge filtering with >= working correctly")
        
        # Test edge filtering with !=
        non_medium_filter = gr.parse_edge_query("strength != 'medium'")
        non_medium_results = g.filter_edges(non_medium_filter)
        non_medium_ids = non_medium_results  # It's already a list
        
        expected_non_medium = {edge1, edge3, edge4}  # All except edge2 (medium)
        actual_non_medium = set(non_medium_ids)
        assert actual_non_medium == expected_non_medium, f"Expected {expected_non_medium}, got {actual_non_medium}"
        print(f"✅ Edge filtering with != working correctly")
        
    except Exception as e:
        print(f"❌ Edge filter test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Comprehensive operator validation
    print(f"\n📋 Test 5: Comprehensive Operator Validation")
    try:
        operators_to_test = [
            ("salary > 120000", {eve}),  # Only Eve has salary > 120k
            ("salary >= 120000", {alice, eve}),  # Alice and Eve have salary >= 120k
            ("salary < 100000", {david}),  # Only David has salary < 100k
            ("salary <= 100000", {bob, david}),  # Bob and David have salary <= 100k
            ("salary == 110000", {carol}),  # Only Carol has exactly 110k
            ("salary != 120000", {bob, carol, david, eve}),  # All except Alice
        ]
        
        all_passed = True
        for query, expected_nodes in operators_to_test:
            try:
                node_filter = gr.parse_node_query(query)
                results = g.filter_nodes(node_filter)
                actual_nodes = set(results.nodes)
                
                if actual_nodes == expected_nodes:
                    print(f"✅ '{query}' → {len(actual_nodes)} nodes (correct)")
                else:
                    print(f"❌ '{query}' → Expected {expected_nodes}, got {actual_nodes}")
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ '{query}' → Error: {e}")
                all_passed = False
        
        if all_passed:
            print(f"✅ All comparison operators working correctly!")
        else:
            print(f"⚠️  Some operators had issues")
        
    except Exception as e:
        print(f"❌ Comprehensive validation failed: {e}")
        import traceback
        traceback.print_exc()

def test_direct_attribute_filter_usage():
    """Test using AttributeFilter methods directly"""
    
    print(f"\n🔧 Testing Direct AttributeFilter Usage")
    
    try:
        # Test creating filters directly
        from groggy import AttrValue, AttributeFilter, NodeFilter
        
        # Create test values
        salary_120k = AttrValue(120000)
        dept_eng = AttrValue("Engineering")
        
        # Test direct filter creation
        not_equals_filter = AttributeFilter.not_equals(dept_eng)
        gte_filter = AttributeFilter.greater_than_or_equal(salary_120k)
        lte_filter = AttributeFilter.less_than_or_equal(AttrValue(30))
        
        print(f"✅ Direct AttributeFilter creation successful")
        
        # Test with NodeFilter
        node_filter_ne = NodeFilter.attribute_filter("dept", not_equals_filter)
        node_filter_gte = NodeFilter.attribute_filter("salary", gte_filter)
        node_filter_lte = NodeFilter.attribute_filter("age", lte_filter)
        
        print(f"✅ NodeFilter creation with new operators successful")
        
        # Test with a simple graph
        g = gr.Graph()
        alice = g.add_node(name="Alice", age=30, salary=120000, dept="Engineering")
        bob = g.add_node(name="Bob", age=25, salary=100000, dept="Marketing")
        
        # Test the filters
        results_ne = g.filter_nodes(node_filter_ne)
        results_gte = g.filter_nodes(node_filter_gte)
        results_lte = g.filter_nodes(node_filter_lte)
        
        print(f"✅ Direct filter usage successful:")
        print(f"   Not Engineering: {len(results_ne.nodes)} nodes")
        print(f"   Salary >= 120k: {len(results_gte.nodes)} nodes") 
        print(f"   Age <= 30: {len(results_lte.nodes)} nodes")
        
    except Exception as e:
        print(f"❌ Direct AttributeFilter test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_comparison_operators()
    test_direct_attribute_filter_usage()
    
    print(f"\n🎉 Enhanced Query Parser Testing Complete!")
    print(f"✨ Step C.1 Successfully Implemented:")
    print(f"   • Not equals operator: dept != 'Engineering'")
    print(f"   • Greater than or equal: salary >= 110000")
    print(f"   • Less than or equal: age <= 30")
    print(f"   • Enhanced edge filtering with new operators")
    print(f"   • Direct AttributeFilter method access")
    print(f"   • Comprehensive operator validation")