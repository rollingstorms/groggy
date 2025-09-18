#!/usr/bin/env python3
"""
Test script for new groupby functionality

This script tests the new groupby methods:
- subgraph.group_by('attr', 'nodes')
- subgraph.group_by('attr', 'edges') 
- g.nodes.group_by('attr')
- g.edges.group_by('attr')
"""

import sys
import os

# Add the python-groggy package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-groggy', 'python'))

try:
    import groggy as gr
    print("✅ Successfully imported groggy")
except ImportError as e:
    print(f"❌ Failed to import groggy: {e}")
    print("💡 Make sure to build the project first: cd python-groggy && maturin develop")
    sys.exit(1)

def test_subgraph_groupby():
    """Test the new Subgraph.group_by() methods"""
    print("🧪 Testing Subgraph.group_by() methods...")
    
    # Create a test graph with some structure
    g = gr.Graph()
    
    # Add nodes with attributes
    n1 = g.add_node(type="user", age=25)
    n2 = g.add_node(type="user", age=30) 
    n3 = g.add_node(type="admin", age=35)
    n4 = g.add_node(type="admin", age=40)
    
    # Add edges with attributes
    e1 = g.add_edge(n1, n2, relation="friend", weight=1.0)
    e2 = g.add_edge(n2, n3, relation="reports_to", weight=2.0)
    e3 = g.add_edge(n3, n4, relation="friend", weight=1.5)
    
    # Get a subgraph to test groupby on
    subgraph = g.view()
    
    # Test group_by nodes
    print("  📋 Testing group_by('type', 'nodes')...")
    node_groups = subgraph.group_by('type', 'nodes')
    print(f"    📊 Got {len(node_groups)} node groups")
    
    # Test group_by edges  
    print("  📋 Testing group_by('relation', 'edges')...")
    edge_groups = subgraph.group_by('relation', 'edges')
    print(f"    📊 Got {len(edge_groups)} edge groups")
    
    print("  ✅ Subgraph.group_by() tests passed!")
    return True

def test_nodes_accessor_groupby():
    """Test the new NodesAccessor.group_by() method"""
    print("🧪 Testing NodesAccessor.group_by() method...")
    
    # Create a test graph
    g = gr.Graph()
    
    # Add nodes with attributes 
    n1 = g.add_node(department="engineering", level="senior")
    n2 = g.add_node(department="engineering", level="junior")
    n3 = g.add_node(department="marketing", level="senior")
    n4 = g.add_node(department="marketing", level="junior")
    
    # Test nodes accessor groupby
    print("  📋 Testing g.nodes.group_by('department')...")
    dept_groups = g.nodes.group_by('department')
    print(f"    📊 Got {len(dept_groups)} department groups")
    
    print("  📋 Testing g.nodes.group_by('level')...")
    level_groups = g.nodes.group_by('level')
    print(f"    📊 Got {len(level_groups)} level groups")
    
    print("  ✅ NodesAccessor.group_by() tests passed!")
    return True

def test_edges_accessor_groupby():
    """Test the new EdgesAccessor.group_by() method"""
    print("🧪 Testing EdgesAccessor.group_by() method...")
    
    # Create a test graph
    g = gr.Graph()
    
    # Add nodes
    n1 = g.add_node(name="alice")
    n2 = g.add_node(name="bob")
    n3 = g.add_node(name="charlie")
    n4 = g.add_node(name="diana")
    
    # Add edges with attributes
    e1 = g.add_edge(n1, n2, type="friendship", strength="strong")
    e2 = g.add_edge(n2, n3, type="friendship", strength="weak")
    e3 = g.add_edge(n3, n4, type="collaboration", strength="strong")
    e4 = g.add_edge(n4, n1, type="collaboration", strength="medium")
    
    # Test edges accessor groupby
    print("  📋 Testing g.edges.group_by('type')...")
    type_groups = g.edges.group_by('type')
    print(f"    📊 Got {len(type_groups)} type groups")
    
    print("  📋 Testing g.edges.group_by('strength')...")
    strength_groups = g.edges.group_by('strength')
    print(f"    📊 Got {len(strength_groups)} strength groups")
    
    print("  ✅ EdgesAccessor.group_by() tests passed!")
    return True

def test_method_chaining():
    """Test method chaining with the new groupby functionality"""
    print("🧪 Testing method chaining...")
    
    # Create a test graph
    g = gr.Graph()
    
    # Add nodes with attributes
    n1 = g.add_node(category="A", value=10)
    n2 = g.add_node(category="A", value=20)
    n3 = g.add_node(category="B", value=15)
    n4 = g.add_node(category="B", value=25)
    
    # Add some edges
    e1 = g.add_edge(n1, n2, weight=1.0)
    e2 = g.add_edge(n2, n3, weight=2.0)
    e3 = g.add_edge(n3, n4, weight=1.5)
    
    # Test method chaining: group nodes by category, then group again
    print("  📋 Testing chained groupby operations...")
    grouped = g.nodes.group_by('category')
    print(f"    📊 Initial grouping: {len(grouped)} groups")
    
    # Test groupby on SubgraphArray (should delegate to each subgraph)
    chained = grouped.group_by('value', 'nodes')  
    print(f"    📊 Chained grouping: {len(chained)} groups")
    
    print("  ✅ Method chaining tests passed!")
    return True

def run_all_tests():
    """Run all test functions and report results"""
    print("🚀 Testing new GroupBy functionality")
    print("=" * 50)
    
    tests = [
        test_subgraph_groupby,
        test_nodes_accessor_groupby,
        test_edges_accessor_groupby,
        test_method_chaining
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ Test {test_func.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print("=" * 50)
    print(f"🏁 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False
        
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)