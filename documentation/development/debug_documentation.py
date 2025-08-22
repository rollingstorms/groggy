#!/usr/bin/env python3
"""
Debug which documented features actually work vs don't work
"""

def test_with_error_handling():
    print("🔍 DEBUGGING DOCUMENTATION ISSUES")
    print("=" * 50)
    
    issues_found = []
    features_working = []
    
    import groggy as gr
    
    # Setup basic graph
    g = gr.Graph()
    alice = g.add_node(name="Alice", age=30, department="Engineering")
    bob = g.add_node(name="Bob", age=25, department="Design") 
    g.add_edge(alice, bob, weight=0.8)
    
    nodes_table = g.nodes.table()
    age_column = nodes_table['age']
    
    print("🧪 Testing documented features...")
    
    # Test 1: Table.mean() method
    try:
        result = nodes_table.mean('age')
        features_working.append("nodes_table.mean('age')")
        print(f"✅ nodes_table.mean('age') = {result}")
    except Exception as e:
        issues_found.append({"feature": "nodes_table.mean('age')", "error": str(e)})
        print(f"❌ nodes_table.mean('age') - {e}")
    
    # Test 2: Table.sum() method  
    try:
        result = nodes_table.sum('age')
        features_working.append("nodes_table.sum('age')")
        print(f"✅ nodes_table.sum('age') = {result}")
    except Exception as e:
        issues_found.append({"feature": "nodes_table.sum('age')", "error": str(e)})
        print(f"❌ nodes_table.sum('age') - {e}")
    
    # Test 3: Array describe method
    try:
        result = age_column.describe()
        features_working.append("age_column.describe()")
        print(f"✅ age_column.describe() = {type(result)}")
    except Exception as e:
        issues_found.append({"feature": "age_column.describe()", "error": str(e)})
        print(f"❌ age_column.describe() - {e}")
    
    # Test 4: Table describe method
    try:
        result = nodes_table.describe()
        features_working.append("nodes_table.describe()")
        print(f"✅ nodes_table.describe() = {type(result)}")
    except Exception as e:
        issues_found.append({"feature": "nodes_table.describe()", "error": str(e)})
        print(f"❌ nodes_table.describe() - {e}")
    
    # Test 5: Graph-aware table filtering
    try:
        result = nodes_table.filter_by_degree(g, 'node_id', min_degree=1)
        features_working.append("nodes_table.filter_by_degree()")
        print(f"✅ filter_by_degree works")
    except Exception as e:
        issues_found.append({"feature": "nodes_table.filter_by_degree()", "error": str(e)})
        print(f"❌ nodes_table.filter_by_degree() - {e}")
    
    try:
        result = nodes_table.filter_by_connectivity(g, 'node_id', [alice], mode='direct')
        features_working.append("nodes_table.filter_by_connectivity()")
        print(f"✅ filter_by_connectivity works")
    except Exception as e:
        issues_found.append({"feature": "nodes_table.filter_by_connectivity()", "error": str(e)})
        print(f"❌ nodes_table.filter_by_connectivity() - {e}")
    
    try:
        result = nodes_table.filter_by_distance(g, 'node_id', [alice], max_distance=2)
        features_working.append("nodes_table.filter_by_distance()")
        print(f"✅ filter_by_distance works")
    except Exception as e:
        issues_found.append({"feature": "nodes_table.filter_by_distance()", "error": str(e)})
        print(f"❌ nodes_table.filter_by_distance() - {e}")
    
    # Test 6: Matrix axis operations with different parameter styles
    adj_matrix = g.adjacency()
    
    try:
        result = adj_matrix.sum_axis(1)
        features_working.append("adj_matrix.sum_axis(1)")
        print(f"✅ adj_matrix.sum_axis(1) works")
    except Exception as e:
        issues_found.append({"feature": "adj_matrix.sum_axis(1)", "error": str(e)})
        print(f"❌ adj_matrix.sum_axis(1) - {e}")
    
    try:
        result = adj_matrix.sum_axis(axis=1)
        features_working.append("adj_matrix.sum_axis(axis=1)")
        print(f"✅ adj_matrix.sum_axis(axis=1) works")
    except Exception as e:
        issues_found.append({"feature": "adj_matrix.sum_axis(axis=1)", "error": str(e)})
        print(f"❌ adj_matrix.sum_axis(axis=1) - {e}")
    
    # Test 7: Array methods we documented
    try:
        result = age_column.min()
        features_working.append("age_column.min()")
        print(f"✅ age_column.min() = {result}")
    except Exception as e:
        issues_found.append({"feature": "age_column.min()", "error": str(e)})
        print(f"❌ age_column.min() - {e}")
    
    try:
        result = age_column.max()
        features_working.append("age_column.max()")
        print(f"✅ age_column.max() = {result}")
    except Exception as e:
        issues_found.append({"feature": "age_column.max()", "error": str(e)})
        print(f"❌ age_column.max() - {e}")
    
    try:
        result = age_column.sum()
        features_working.append("age_column.sum()")
        print(f"✅ age_column.sum() = {result}")
    except Exception as e:
        issues_found.append({"feature": "age_column.sum()", "error": str(e)})
        print(f"❌ age_column.sum() - {e}")
    
    # Test 8: Table methods we documented
    try:
        result = nodes_table.head(3)
        features_working.append("nodes_table.head(3)")
        print(f"✅ nodes_table.head(3) works")
    except Exception as e:
        issues_found.append({"feature": "nodes_table.head(3)", "error": str(e)})
        print(f"❌ nodes_table.head(3) - {e}")
    
    try:
        result = nodes_table.tail(2)
        features_working.append("nodes_table.tail(2)")
        print(f"✅ nodes_table.tail(2) works")
    except Exception as e:
        issues_found.append({"feature": "nodes_table.tail(2)", "error": str(e)})
        print(f"❌ nodes_table.tail(2) - {e}")
    
    # Test 9: Node/edge access methods from API docs
    try:
        # These methods were documented but may not exist
        result = g.get_node(alice)
        features_working.append("g.get_node(alice)")
        print(f"✅ g.get_node() works")
    except Exception as e:
        issues_found.append({"feature": "g.get_node(alice)", "error": str(e)})
        print(f"❌ g.get_node(alice) - {e}")
    
    try:
        result = g.get_edge(alice, bob)
        features_working.append("g.get_edge(alice, bob)")
        print(f"✅ g.get_edge() works")
    except Exception as e:
        issues_found.append({"feature": "g.get_edge(alice, bob)", "error": str(e)})
        print(f"❌ g.get_edge(alice, bob) - {e}")
    
    try:
        g.update_node(alice, {"promoted": True})
        features_working.append("g.update_node()")
        print(f"✅ g.update_node() works")
    except Exception as e:
        issues_found.append({"feature": "g.update_node(alice, dict)", "error": str(e)})
        print(f"❌ g.update_node() - {e}")
    
    # Test 10: Attribute setting methods we may have referenced
    try:
        g.set_node_attribute(alice, "promoted", True)
        features_working.append("g.set_node_attribute()")
        print(f"✅ g.set_node_attribute() works")
    except Exception as e:
        issues_found.append({"feature": "g.set_node_attribute()", "error": str(e)})
        print(f"❌ g.set_node_attribute() - {e}")
    
    try:
        g.set_edge_attribute(alice, bob, "verified", True)
        features_working.append("g.set_edge_attribute()")
        print(f"✅ g.set_edge_attribute() works")
    except Exception as e:
        issues_found.append({"feature": "g.set_edge_attribute()", "error": str(e)})
        print(f"❌ g.set_edge_attribute() - {e}")
    
    # Test 11: Matrix properties we documented
    try:
        result = adj_matrix.is_sparse
        features_working.append("adj_matrix.is_sparse")
        print(f"✅ adj_matrix.is_sparse = {result}")
    except Exception as e:
        issues_found.append({"feature": "adj_matrix.is_sparse", "error": str(e)})
        print(f"❌ adj_matrix.is_sparse - {e}")
    
    try:
        result = adj_matrix.to_numpy()
        features_working.append("adj_matrix.to_numpy()")
        print(f"✅ adj_matrix.to_numpy() works")
    except Exception as e:
        issues_found.append({"feature": "adj_matrix.to_numpy()", "error": str(e)})
        print(f"❌ adj_matrix.to_numpy() - {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DOCUMENTATION DEBUG SUMMARY")
    print("=" * 50)
    
    print(f"\n✅ WORKING FEATURES ({len(features_working)}):")
    for feature in features_working:
        print(f"   • {feature}")
    
    print(f"\n❌ BROKEN FEATURES ({len(issues_found)}):")
    for issue in issues_found:
        print(f"   • {issue['feature']}: {issue['error']}")
    
    if issues_found:
        print(f"\n⚠️  DOCUMENTATION NEEDS FIXES:")
        print(f"   - Remove or fix {len(issues_found)} non-working examples")
        print(f"   - Update method signatures to match implementation")
    else:
        print(f"\n🎉 ALL DOCUMENTED FEATURES WORK!")
    
    return issues_found, features_working

if __name__ == "__main__":
    issues, working = test_with_error_handling()