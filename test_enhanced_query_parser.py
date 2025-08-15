#!/usr/bin/env python3
"""
Test script for enhanced query parser with 3+ term logical expressions and parentheses support.

Tests the key features:
1. 3+ term AND expressions: "A AND B AND C"
2. 3+ term OR expressions: "A OR B OR C" 
3. Parentheses grouping: "(A OR B) AND C"
4. Mixed operators: "A AND (B OR C)"
5. NOT with parentheses: "NOT (A OR B)"
"""

import groggy

def test_enhanced_query_parser():
    """Test the enhanced query parser with complex expressions."""
    
    print("🚀 Testing Enhanced Query Parser")
    print("=" * 50)
    
    # Create test graph with diverse data
    print("Creating test graph with sample data...")
    g = groggy.Graph()
    
    # Add nodes with multiple attributes for testing
    node_data = [
        {"name": "Alice", "age": 28, "dept": "Engineering", "salary": 85000, "active": True},
        {"name": "Bob", "age": 32, "dept": "Engineering", "salary": 95000, "active": True},
        {"name": "Charlie", "age": 45, "dept": "Engineering", "salary": 120000, "active": False},
        {"name": "Diana", "age": 29, "dept": "Sales", "salary": 75000, "active": True},
        {"name": "Eve", "age": 35, "dept": "Sales", "salary": 80000, "active": True},
        {"name": "Frank", "age": 50, "dept": "Sales", "salary": 110000, "active": False},
        {"name": "Grace", "age": 26, "dept": "Marketing", "salary": 70000, "active": True},
        {"name": "Henry", "age": 40, "dept": "Marketing", "salary": 90000, "active": True},
        {"name": "Ivy", "age": 55, "dept": "HR", "salary": 95000, "active": False},
        {"name": "Jack", "age": 30, "dept": "HR", "salary": 85000, "active": True},
    ]
    
    nodes = g.add_nodes(node_data)
    print(f"✅ Created {len(nodes)} nodes")
    
    # Test cases for the enhanced query parser
    test_cases = [
        # 3+ term AND expressions
        {
            "query": 'dept == "Engineering" AND age > 30 AND salary > 90000',
            "description": "3-term AND: Senior well-paid engineers",
            "expected_names": ["Bob", "Charlie"]
        },
        
        # 3+ term OR expressions  
        {
            "query": 'dept == "Sales" OR dept == "Marketing" OR dept == "HR"',
            "description": "3-term OR: Non-engineering departments",
            "expected_names": ["Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]
        },
        
        # Parentheses grouping
        {
            "query": '(age < 30 OR age > 50) AND active == true',
            "description": "Parentheses: Young or old active employees",
            "expected_names": ["Alice", "Diana", "Grace"]
        },
        
        # Mixed operators
        {
            "query": 'dept == "Engineering" AND (age < 30 OR salary > 100000)',
            "description": "Mixed: Engineers who are young or high-paid",
            "expected_names": ["Alice", "Charlie"]
        },
        
        # NOT with parentheses
        {
            "query": 'NOT (dept == "Engineering" OR dept == "Sales")',
            "description": "NOT with parentheses: Non-engineering, non-sales",
            "expected_names": ["Grace", "Henry", "Ivy", "Jack"]
        },
        
        # Complex 4-term expression
        {
            "query": 'age > 25 AND age < 50 AND salary > 70000 AND active == true',
            "description": "4-term AND: Active mid-career employees with good salary",
            "expected_names": ["Alice", "Bob", "Diana", "Eve", "Grace", "Henry", "Jack"]
        },
        
        # Complex nested parentheses
        {
            "query": '(dept == "Engineering" OR dept == "Sales") AND (age > 30 AND salary > 80000)',
            "description": "Nested: Senior well-paid tech/sales",
            "expected_names": ["Bob", "Charlie", "Eve", "Frank"]
        }
    ]
    
    print(f"\n🔧 Testing {len(test_cases)} complex query expressions:")
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        description = test_case["description"] 
        expected_names = set(test_case["expected_names"])
        
        print(f"\n  Test {i}: {description}")
        print(f"    Query: {query}")
        
        try:
            # Test the enhanced parser
            result = g.filter_nodes(query)
            result_names = set()
            
            # Extract names from results using table view
            try:
                # Use table() method to get node data
                table = result.table()
                rows, columns = table._build_table_data()
                for row in rows:
                    name = row.get('name')
                    if name:
                        result_names.add(name)
            except Exception as table_error:
                # Fallback: try to get node count and validate manually
                print(f"    Table approach failed ({table_error}), using count validation")
                count = result.node_count()
                print(f"    Found {count} nodes (cannot extract names for validation)")
            
            print(f"    Found: {sorted(result_names)}")
            print(f"    Expected: {sorted(expected_names)}")
            
            if result_names == expected_names:
                print(f"    ✅ PASS")
            else:
                print(f"    ❌ FAIL - Mismatch!")
                print(f"       Missing: {sorted(expected_names - result_names)}")
                print(f"       Extra: {sorted(result_names - expected_names)}")
                all_passed = False
                
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            all_passed = False
    
    # Test error cases
    print(f"\n🔧 Testing error handling:")
    
    error_cases = [
        "age > 30 AND",  # Incomplete expression
        "(age > 30",     # Missing closing parenthesis  
        "age > 30)",     # Missing opening parenthesis
        "AND age > 30",  # Invalid start
    ]
    
    for query in error_cases:
        print(f"    Testing error case: {query}")
        try:
            result = g.filter_nodes(query)
            print(f"    ❌ Should have failed but didn't!")
            all_passed = False
        except Exception as e:
            print(f"    ✅ Correctly failed: {type(e).__name__}")
    
    # Performance test
    print(f"\n📈 Performance test with complex query:")
    import time
    
    complex_query = '(dept == "Engineering" AND age > 25) OR (dept == "Sales" AND salary > 75000) OR (dept == "Marketing" AND active == true)'
    
    start_time = time.time()
    for _ in range(100):  # Run 100 times
        result = g.filter_nodes(complex_query)
    elapsed = time.time() - start_time
    
    print(f"    Complex query × 100: {elapsed*1000:.2f}ms total ({elapsed*10:.3f}ms per query)")
    print(f"    Query: {complex_query}")
    print(f"    Results: {result.node_count()} nodes")
    
    # Summary
    print(f"\n🎯 Enhanced Query Parser Test Results:")
    if all_passed:
        print(f"    ✅ ALL TESTS PASSED")
        print(f"    🚀 3+ term expressions: WORKING")
        print(f"    🚀 Parentheses support: WORKING") 
        print(f"    🚀 Mixed operators: WORKING")
        print(f"    🚀 Error handling: WORKING")
        print(f"    🚀 Performance: GOOD")
    else:
        print(f"    ❌ SOME TESTS FAILED")
        print(f"    💡 Check implementation details")
    
    return all_passed

if __name__ == "__main__":
    test_enhanced_query_parser()