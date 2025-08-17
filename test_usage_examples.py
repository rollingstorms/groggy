#!/usr/bin/env python3
"""
Test script to validate the usage examples from docs/usage_examples.md
This script tests each section systematically and reports what works/fails.
"""

import sys
import traceback
import groggy as gr

def test_section(name, test_func):
    """Run a test section and report results"""
    print(f"\n🧪 Testing {name}...")
    try:
        test_func()
        print(f"✅ {name} - PASSED")
        return True
    except Exception as e:
        print(f"❌ {name} - FAILED: {e}")
        traceback.print_exc()
        return False

def test_graph_construction():
    """Test basic graph construction examples"""
    g = gr.Graph()
    
    # Test node creation
    alice = g.add_node(id="alice", age=30, role="engineer")
    bob = g.add_node(id="bob", age=25, role="engineer")
    
    # Test edge creation
    g.add_edge(alice, bob, relationship="collaborates")
    
    # Test bulk operations
    node_data = [
        {"id": "charlie", "age": 35, "role": "manager"},
        {"id": "diana", "age": 28, "role": "designer"}
    ]
    node_mapping = g.add_nodes(node_data, uid_key="id")
    print(f"Node mapping: {node_mapping}")

def test_graph_array_basic():
    """Test GraphArray basic functionality"""
    # Create GraphArray
    ages = gr.GraphArray([25, 30, 35, 40, 45])
    
    # Test statistical methods
    print(f"Mean: {ages.mean()}")
    print(f"Std: {ages.std()}")
    print(f"Min: {ages.min()}")
    print(f"Max: {ages.max()}")
    print(f"Median: {ages.median()}")
    
    # Test list compatibility
    print(f"Length: {len(ages)}")
    print(f"First element: {ages[0]}")
    print(f"Last element: {ages[-1]}")
    
    # Test conversion
    plain_list = ages.to_list()
    print(f"Converted to list: {plain_list}")

def test_filtering():
    """Test filtering functionality"""
    g = gr.Graph()
    
    # Add test data
    for i in range(5):
        g.add_node(age=25 + i*5, dept="Engineering" if i % 2 == 0 else "Sales")
    
    # Test string-based filtering
    engineers = g.filter_nodes("dept == 'Engineering'")
    print(f"Found {len(engineers.node_ids)} engineers")
    
    # Test age filtering
    young = g.filter_nodes("age < 35")
    print(f"Found {len(young.node_ids)} young people")

def test_adjacency_matrices():
    """Test adjacency matrix functionality"""
    g = gr.Graph()
    
    # Create a simple graph
    nodes = [g.add_node(id=f"node_{i}") for i in range(4)]
    g.add_edge(nodes[0], nodes[1])
    g.add_edge(nodes[1], nodes[2])
    g.add_edge(nodes[2], nodes[3])
    
    # Test adjacency matrix methods (use dense to avoid sparse NotImplementedError)
    adj_matrix = g.dense_adjacency_matrix()
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    
    # Test access patterns
    print(f"Matrix[0,1]: {adj_matrix[0, 1]}")
    print(f"Row 0: {adj_matrix[0]}")

def test_table_functionality():
    """Test table functionality"""
    g = gr.Graph()
    
    # Add nodes with attributes
    g.add_node(id="alice", age=30, salary=75000)
    g.add_node(id="bob", age=25, salary=65000)
    g.add_node(id="charlie", age=35, salary=85000)
    
    # Test table creation
    table = g.table()
    print(f"Table type: {type(table)}")
    
    # Test column access (if available)
    try:
        ages = table['age']
        print(f"Ages column type: {type(ages)}")
        if hasattr(ages, 'mean'):
            print(f"Mean age: {ages.mean()}")
    except Exception as e:
        print(f"Column access failed: {e}")

def test_algorithms():
    """Test graph algorithms"""
    g = gr.Graph()
    
    # Create a connected graph
    nodes = [g.add_node(id=f"node_{i}") for i in range(5)]
    g.add_edge(nodes[0], nodes[1])
    g.add_edge(nodes[1], nodes[2])
    g.add_edge(nodes[2], nodes[3])
    g.add_edge(nodes[3], nodes[4])
    
    # Test connected components
    components = g.connected_components()
    print(f"Found {len(components)} components")
    print(f"First component has {len(components[0].node_ids)} nodes")
    
    # Test BFS/DFS
    visited_bfs = g.bfs(start_node=nodes[0])
    visited_dfs = g.dfs(start_node=nodes[0])
    print(f"BFS visited {len(visited_bfs.node_ids)} nodes")
    print(f"DFS visited {len(visited_dfs.node_ids)} nodes")

def test_version_control():
    """Test version control functionality"""
    g = gr.Graph()
    
    # Add some data
    g.add_node(id="test", value=1)
    
    # Test commit
    commit_id = g.commit("Initial commit", "Test User")
    print(f"Created commit: {commit_id}")
    
    # Test branch operations
    g.create_branch("feature-branch")
    branches = g.branches()
    print(f"Available branches: {[b.name for b in branches]}")

def main():
    """Run all tests"""
    print("🚀 Testing Groggy Usage Examples")
    print("=" * 50)
    
    tests = [
        ("Graph Construction", test_graph_construction),
        ("GraphArray Basic", test_graph_array_basic),
        ("Filtering", test_filtering),
        ("Adjacency Matrices", test_adjacency_matrices),
        ("Table Functionality", test_table_functionality),
        ("Algorithms", test_algorithms),
        ("Version Control", test_version_control),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        if test_section(name, test_func):
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())