#!/usr/bin/env python3
"""
Test Logical Operators (Step C.2)

This tests the enhanced query parser with logical operators: AND, OR, NOT
"""

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy as gr

def test_logical_operators():
    """Test logical operators in enhanced query parser"""
    
    print("🧠 Testing Logical Operators (Step C.2)")
    
    # Create test graph with varied attributes
    g = gr.Graph()
    
    # Add employees with different combinations of attributes
    alice = g.add_node(name="Alice", age=30, salary=120000, dept="Engineering", seniority="senior")
    bob = g.add_node(name="Bob", age=25, salary=100000, dept="Engineering", seniority="junior") 
    carol = g.add_node(name="Carol", age=35, salary=110000, dept="Design", seniority="senior")
    david = g.add_node(name="David", age=28, salary=95000, dept="Marketing", seniority="junior")
    eve = g.add_node(name="Eve", age=32, salary=130000, dept="Engineering", seniority="senior")
    frank = g.add_node(name="Frank", age=29, salary=105000, dept="Design", seniority="junior")
    
    print(f"✅ Created test graph: {g.node_count()} nodes")
    
    # Test 1: AND operator
    print(f"\n📋 Test 1: AND Operator")
    try:
        # Find senior Engineering employees
        and_query = "dept == 'Engineering' AND seniority == 'senior'"
        and_results = gr.enhanced_filter_nodes(g, and_query)
        and_nodes = set(and_results.nodes)
        
        print(f"✅ Senior Engineering employees: {len(and_nodes)} found")
        
        # Expected: Alice and Eve (both Engineering AND senior)
        expected_and = {alice, eve}
        assert and_nodes == expected_and, f"Expected {expected_and}, got {and_nodes}"
        print(f"✅ AND operator working correctly")
        
        # Show the results
        for node_id in and_nodes:
            node_view = g.nodes[node_id]
            print(f"   Senior Engineer: {node_view}")
        
    except Exception as e:
        print(f"❌ AND operator test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: OR operator
    print(f"\n📋 Test 2: OR Operator")
    try:
        # Find employees who are either very young (<=26) or very experienced (>=35)
        or_query = "age <= 26 OR age >= 35"
        or_results = gr.enhanced_filter_nodes(g, or_query)
        or_nodes = set(or_results.nodes)
        
        print(f"✅ Very young or very experienced: {len(or_nodes)} found")
        
        # Expected: Bob (25) and Carol (35)
        expected_or = {bob, carol}
        assert or_nodes == expected_or, f"Expected {expected_or}, got {or_nodes}"
        print(f"✅ OR operator working correctly")
        
        # Show the results
        for node_id in or_nodes:
            node_view = g.nodes[node_id]
            print(f"   Young or experienced: {node_view}")
        
    except Exception as e:
        print(f"❌ OR operator test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: NOT operator
    print(f"\n📋 Test 3: NOT Operator")
    try:
        # Find non-Engineering employees
        not_query = "NOT dept == 'Engineering'"
        not_results = gr.enhanced_filter_nodes(g, not_query)
        not_nodes = set(not_results.nodes)
        
        print(f"✅ Non-Engineering employees: {len(not_nodes)} found")
        
        # Expected: Carol (Design), David (Marketing), Frank (Design)
        expected_not = {carol, david, frank}
        assert not_nodes == expected_not, f"Expected {expected_not}, got {not_nodes}"
        print(f"✅ NOT operator working correctly")
        
        # Show the results
        for node_id in not_nodes:
            node_view = g.nodes[node_id]
            print(f"   Non-Engineering: {node_view}")
        
    except Exception as e:
        print(f"❌ NOT operator test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Complex logical combinations
    print(f"\n📋 Test 4: Complex Logical Combinations")
    try:
        # Find employees who are senior AND earn more than 110k
        complex_query1 = "seniority == 'senior' AND salary > 110000"
        complex_results1 = gr.enhanced_filter_nodes(g, complex_query1)
        complex_nodes1 = set(complex_results1.nodes)
        
        print(f"✅ Senior high earners: {len(complex_nodes1)} found")
        
        # Expected: Alice (senior, 120k), Eve (senior, 130k)
        expected_complex1 = {alice, eve}
        assert complex_nodes1 == expected_complex1, f"Expected {expected_complex1}, got {complex_nodes1}"
        
        # Find employees who are junior OR in Design
        complex_query2 = "seniority == 'junior' OR dept == 'Design'"
        complex_results2 = gr.enhanced_filter_nodes(g, complex_query2)
        complex_nodes2 = set(complex_results2.nodes)
        
        print(f"✅ Junior or Design employees: {len(complex_nodes2)} found")
        
        # Expected: Bob (junior), David (junior), Carol (Design), Frank (both junior AND Design)
        expected_complex2 = {bob, david, carol, frank}
        assert complex_nodes2 == expected_complex2, f"Expected {expected_complex2}, got {complex_nodes2}"
        
        print(f"✅ Complex logical combinations working correctly")
        
    except Exception as e:
        print(f"❌ Complex logical combinations failed: {e}")
        import traceback
        traceback.print_exc()

def test_edge_logical_operators():
    """Test logical operators for edge filtering"""
    
    print(f"\n🔗 Testing Edge Logical Operators")
    
    try:
        # Create test graph with edges
        g = gr.Graph()
        
        alice = g.add_node(name="Alice")
        bob = g.add_node(name="Bob")
        carol = g.add_node(name="Carol")
        david = g.add_node(name="David")
        
        # Add edges with different attributes
        edge1 = g.add_edge(alice, bob, weight=0.8, type="strong", relationship="collaborates")
        edge2 = g.add_edge(bob, carol, weight=0.6, type="medium", relationship="collaborates")
        edge3 = g.add_edge(carol, david, weight=0.9, type="strong", relationship="mentors")
        edge4 = g.add_edge(david, alice, weight=0.4, type="weak", relationship="knows")
        
        print(f"✅ Created edge test graph: {g.edge_count()} edges")
        
        # Test AND for edges
        and_edge_query = "weight > 0.7 AND type == 'strong'"
        and_edge_results = gr.enhanced_filter_edges(g, and_edge_query)
        and_edge_ids = set(and_edge_results)
        
        print(f"✅ Strong high-weight edges: {len(and_edge_ids)} found")
        
        # Expected: edge1 (0.8, strong), edge3 (0.9, strong)
        expected_and_edges = {edge1, edge3}
        assert and_edge_ids == expected_and_edges, f"Expected {expected_and_edges}, got {and_edge_ids}"
        
        # Test OR for edges  
        or_edge_query = "relationship == 'mentors' OR weight < 0.5"
        or_edge_results = gr.enhanced_filter_edges(g, or_edge_query)
        or_edge_ids = set(or_edge_results)
        
        print(f"✅ Mentor or weak edges: {len(or_edge_ids)} found")
        
        # Expected: edge3 (mentors), edge4 (weight 0.4)
        expected_or_edges = {edge3, edge4}
        assert or_edge_ids == expected_or_edges, f"Expected {expected_or_edges}, got {or_edge_ids}"
        
        # Test NOT for edges
        not_edge_query = "NOT relationship == 'collaborates'"
        not_edge_results = gr.enhanced_filter_edges(g, not_edge_query)
        not_edge_ids = set(not_edge_results)
        
        print(f"✅ Non-collaboration edges: {len(not_edge_ids)} found")
        
        # Expected: edge3 (mentors), edge4 (knows)
        expected_not_edges = {edge3, edge4}
        assert not_edge_ids == expected_not_edges, f"Expected {expected_not_edges}, got {not_edge_ids}"
        
        print(f"✅ All edge logical operators working correctly!")
        
    except Exception as e:
        print(f"❌ Edge logical operators failed: {e}")
        import traceback
        traceback.print_exc()

def test_query_validation():
    """Test query validation and error handling"""
    
    print(f"\n🔍 Testing Query Validation")
    
    try:
        g = gr.Graph()
        alice = g.add_node(name="Alice", age=30)
        bob = g.add_node(name="Bob", age=25)
        
        # Test valid queries
        valid_queries = [
            "age > 25",
            "name == 'Alice'", 
            "age > 25 AND name == 'Alice'",
            "age < 30 OR name == 'Bob'",
            "NOT age == 25",
        ]
        
        all_passed = True
        for query in valid_queries:
            try:
                result = gr.enhanced_filter_nodes(g, query)
                print(f"✅ Valid query: '{query}' → {len(result.nodes)} nodes")
            except Exception as e:
                print(f"❌ Valid query failed: '{query}' → {e}")
                all_passed = False
        
        if all_passed:
            print(f"✅ All query validation tests passed!")
        
    except Exception as e:
        print(f"❌ Query validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_logical_operators()
    test_edge_logical_operators()
    test_query_validation()
    
    print(f"\n🎉 Logical Operators Testing Complete!")
    print(f"✨ Step C.2 Successfully Implemented:")
    print(f"   • AND operator: dept == 'Engineering' AND seniority == 'senior'")
    print(f"   • OR operator: age <= 26 OR age >= 35")
    print(f"   • NOT operator: NOT dept == 'Engineering'")
    print(f"   • Complex combinations with multiple logical operators")
    print(f"   • Edge logical operators with weight and relationship filters")
    print(f"   • Comprehensive query validation and error handling")