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
    g = groggy.Graph()
    
    # Add nodes with attributes
    n1 = g.add_node(None, {"type": "user", "age": 25})
    n2 = g.add_node(None, {"type": "user", "age": 30}) 
    n3 = g.add_node(None, {"type": "admin", "age": 35})
    n4 = g.add_node(None, {"type": "admin", "age": 40})

def test_nodes_accessor_groupby():
    """Test g.nodes.group_by() functionality"""
    print("\n🧪 Testing NodesAccessor.group_by() method...")
    
    # Create a simple graph
    g = gr.Graph()
    
    # Add nodes with department attribute
    g.add_node(0, {"department": "engineering", "level": "senior"})
    g.add_node(1, {"department": "engineering", "level": "junior"})
    g.add_node(2, {"department": "sales", "level": "senior"})
    g.add_node(3, {"department": "sales", "level": "junior"})
    
    # Add some edges
    g.add_edge(0, 1, {})
    g.add_edge(2, 3, {})
    
    try:
        # Test nodes accessor groupby
        dept_groups = g.nodes.group_by('department')
        print(f"✅ g.nodes.group_by('department') returned {len(dept_groups)} groups")
        
        for i, group in enumerate(dept_groups):
            print(f"   Group {i}: {group.node_count()} nodes, {group.edge_count()} edges")
            
        # Test level groupby
        level_groups = g.nodes.group_by('level')
        print(f"✅ g.nodes.group_by('level') returned {len(level_groups)} groups")
        
        for i, group in enumerate(level_groups):
            print(f"   Group {i}: {group.node_count()} nodes, {group.edge_count()} edges")
            
    except Exception as e:
        print(f"❌ NodesAccessor groupby test failed: {e}")
        return False
    
    return True

def test_edges_accessor_groupby():
    """Test g.edges.group_by() functionality"""
    print("\n🧪 Testing EdgesAccessor.group_by() method...")
    
    # Create a simple graph
    g = gr.Graph()
    
    # Add nodes
    g.add_node(0, {})
    g.add_node(1, {})
    g.add_node(2, {})
    g.add_node(3, {})
    
    # Add edges with different types
    g.add_edge(0, 1, {"type": "friendship", "strength": "strong"})
    g.add_edge(1, 2, {"type": "friendship", "strength": "weak"})
    g.add_edge(2, 3, {"type": "business", "strength": "strong"})
    g.add_edge(3, 0, {"type": "business", "strength": "weak"})
    
    try:
        # Test edges accessor groupby by type
        type_groups = g.edges.group_by('type')
        print(f"✅ g.edges.group_by('type') returned {len(type_groups)} groups")
        
        for i, group in enumerate(type_groups):
            print(f"   Group {i}: {group.node_count()} nodes, {group.edge_count()} edges")
            
        # Test groupby by strength
        strength_groups = g.edges.group_by('strength')
        print(f"✅ g.edges.group_by('strength') returned {len(strength_groups)} groups")
        
        for i, group in enumerate(strength_groups):
            print(f"   Group {i}: {group.node_count()} nodes, {group.edge_count()} edges")
            
    except Exception as e:
        print(f"❌ EdgesAccessor groupby test failed: {e}")
        return False
    
    return True

def test_method_chaining():
    """Test method chaining with new groupby methods"""
    print("\n🧪 Testing method chaining...")
    
    g = gr.Graph()
    
    # Add nodes with nested attributes
    g.add_node(0, {"dept": "eng", "team": "backend"})
    g.add_node(1, {"dept": "eng", "team": "frontend"})
    g.add_node(2, {"dept": "eng", "team": "backend"})
    g.add_node(3, {"dept": "sales", "team": "outbound"})
    
    g.add_edge(0, 1, {})
    g.add_edge(1, 2, {})
    g.add_edge(2, 3, {})
    
    try:
        # Test chaining: group by dept, then get tables
        dept_groups = g.nodes.group_by('dept')
        tables = dept_groups.table()
        print(f"✅ g.nodes.group_by('dept').table() returned {len(tables)} tables")
        
        # Test nested grouping with SubgraphArray
        nested_groups = dept_groups.group_by('team', 'nodes')
        print(f"✅ dept_groups.group_by('team', 'nodes') returned {len(nested_groups)} nested groups")
        
    except Exception as e:
        print(f"❌ Method chaining test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing new GroupBy functionality")
    print("=" * 50)
    
    tests = [
        test_subgraph_groupby,
        test_nodes_accessor_groupby,
        test_edges_accessor_groupby,
        test_method_chaining,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! New groupby functionality is working.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())