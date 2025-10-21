#!/usr/bin/env python3
"""
Comprehensive test script for Groggy Graph
Generated on: 2025-09-07 21:42:37

This script tests ALL methods of the Graph class with proper argument patterns.
Edit the TODO sections to provide correct arguments for each method.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr
import traceback
from datetime import datetime

def create_test_objects():
    """Create test objects for Graph testing"""
    print("🏗️ Creating test objects...")
    
    # Core graph with rich data
    g = gr.Graph()
    nodes_data = g.add_nodes([
        {'name': 'Alice', 'age': 25, 'salary': 75000, 'active': True, 'team': 'Engineering', 'level': 'Senior'},
        {'name': 'Bob', 'age': 30, 'salary': 85000, 'active': True, 'team': 'Sales', 'level': 'Manager'},
        {'name': 'Charlie', 'age': 35, 'salary': 95000, 'active': False, 'team': 'Marketing', 'level': 'Director'},
        {'name': 'Diana', 'age': 28, 'salary': 80000, 'active': True, 'team': 'Engineering', 'level': 'Senior'},
        {'name': 'Eve', 'age': 32, 'salary': 90000, 'active': True, 'team': 'Product', 'level': 'Manager'},
    ])
    
    edges_data = g.add_edges([
        (nodes_data[0], nodes_data[1], {'weight': 1.5, 'type': 'collaboration', 'strength': 'strong'}),
        (nodes_data[1], nodes_data[2], {'weight': 2.0, 'type': 'reports_to', 'strength': 'formal'}),
        (nodes_data[2], nodes_data[3], {'weight': 0.8, 'type': 'peer', 'strength': 'weak'}),
        (nodes_data[0], nodes_data[3], {'weight': 1.2, 'type': 'collaboration', 'strength': 'medium'}),
        (nodes_data[1], nodes_data[4], {'weight': 1.8, 'type': 'cross_team', 'strength': 'strong'}),
    ])
    
    # Create the specific test object for Graph
    test_obj = g
    
    return test_obj, nodes_data, edges_data

def test_method(obj, method_name, method_func, nodes_data, edges_data):
    """Test a single method with error handling"""
    print(f"Testing {method_name}...")
    
    try:
        # Call the method - EDIT THE ARGUMENTS AS NEEDED
        if method_name == 'PLACEHOLDER_METHOD':
            # Example: result = method_func(arg1, arg2, kwarg1=value)
            result = method_func()
        else:
            # Default call with no arguments
            result = method_func()
        
        print(f"  ✅ {method_name}() → {type(result).__name__}: {result}")
        return True, result
        
    except Exception as e:
        print(f"  ❌ {method_name}() → Error: {str(e)}")
        return False, str(e)


def test___len__(test_obj, nodes_data, edges_data):
    """Test Graph.__len__()"""
    # No arguments needed
    try:
        if hasattr(test_obj, '__len__'):
            method = getattr(test_obj, '__len__')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ __len__() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if '__len__' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ __len__() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ __len__() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ __len__ not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ __len__() → Error: {str(e)}")
        return False, str(e)

def test___repr__(test_obj, nodes_data, edges_data):
    """Test Graph.__repr__()"""
    # No arguments needed
    try:
        if hasattr(test_obj, '__repr__'):
            method = getattr(test_obj, '__repr__')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ __repr__() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if '__repr__' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ __repr__() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ __repr__() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ __repr__ not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ __repr__() → Error: {str(e)}")
        return False, str(e)

def test___str__(test_obj, nodes_data, edges_data):
    """Test Graph.__str__()"""
    # No arguments needed
    try:
        if hasattr(test_obj, '__str__'):
            method = getattr(test_obj, '__str__')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ __str__() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if '__str__' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ __str__() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ __str__() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ __str__ not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ __str__() → Error: {str(e)}")
        return False, str(e)

def test_add_edge(test_obj, nodes_data, edges_data):
    """Test Graph.add_edge(source, target, uid_key=None, **kwargs)"""
    # Arguments for add_edge(source, target, uid_key=None, **kwargs)
    try:
        if hasattr(test_obj, 'add_edge'):
            method = getattr(test_obj, 'add_edge')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0, 1)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ add_edge() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'add_edge' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ add_edge() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ add_edge() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ add_edge not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ add_edge() → Error: {str(e)}")
        return False, str(e)

def test_add_edges(test_obj, nodes_data, edges_data):
    """Test Graph.add_edges(edges, node_mapping=None, _uid_key=None, warm_cache=None)"""
    # Arguments for add_edges(edges, node_mapping=None, _uid_key=None, warm_cache=None)
    try:
        if hasattr(test_obj, 'add_edges'):
            method = getattr(test_obj, 'add_edges')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method([(0, 1)])
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ add_edges() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'add_edges' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ add_edges() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ add_edges() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ add_edges not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ add_edges() → Error: {str(e)}")
        return False, str(e)

def test_add_graph(test_obj, nodes_data, edges_data):
    """Test Graph.add_graph(other)"""
    # Arguments for add_graph(other)
    try:
        if hasattr(test_obj, 'add_graph'):
            method = getattr(test_obj, 'add_graph')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(gr.complete_graph(3))
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ add_graph() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'add_graph' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ add_graph() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ add_graph() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ add_graph not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ add_graph() → Error: {str(e)}")
        return False, str(e)

def test_add_node(test_obj, nodes_data, edges_data):
    """Test Graph.add_node(**kwargs)"""
    # Arguments for add_node(**kwargs)
    try:
        if hasattr(test_obj, 'add_node'):
            method = getattr(test_obj, 'add_node')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(name='test_node', age=25)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ add_node() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'add_node' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ add_node() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ add_node() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ add_node not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ add_node() → Error: {str(e)}")
        return False, str(e)

def test_add_nodes(test_obj, nodes_data, edges_data):
    """Test Graph.add_nodes(data, uid_key=None)"""
    # Arguments for add_nodes(data, uid_key=None)
    try:
        if hasattr(test_obj, 'add_nodes'):
            method = getattr(test_obj, 'add_nodes')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method([{'name': 'test1'}, {'name': 'test2'}])
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ add_nodes() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'add_nodes' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ add_nodes() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ add_nodes() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ add_nodes not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ add_nodes() → Error: {str(e)}")
        return False, str(e)

def test_adjacency(test_obj, nodes_data, edges_data):
    """Test Graph.adjacency()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'adjacency'):
            method = getattr(test_obj, 'adjacency')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ adjacency() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'adjacency' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ adjacency() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ adjacency() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ adjacency not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ adjacency() → Error: {str(e)}")
        return False, str(e)

def test_adjacency_matrix(test_obj, nodes_data, edges_data):
    """Test Graph.adjacency_matrix()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'adjacency_matrix'):
            method = getattr(test_obj, 'adjacency_matrix')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ adjacency_matrix() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'adjacency_matrix' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ adjacency_matrix() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ adjacency_matrix() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ adjacency_matrix not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ adjacency_matrix() → Error: {str(e)}")
        return False, str(e)

def test_aggregate(test_obj, nodes_data, edges_data):
    """Test Graph.aggregate(attribute, operation, target=None, _node_ids=None)"""
    # Arguments for aggregate(attribute, operation, target=None, _node_ids=None)
    try:
        if hasattr(test_obj, 'aggregate'):
            method = getattr(test_obj, 'aggregate')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('level', 'mean')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ aggregate() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'aggregate' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ aggregate() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ aggregate() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ aggregate not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ aggregate() → Error: {str(e)}")
        return False, str(e)

def test_all_edge_attribute_names(test_obj, nodes_data, edges_data):
    """Test Graph.all_edge_attribute_names()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'all_edge_attribute_names'):
            method = getattr(test_obj, 'all_edge_attribute_names')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ all_edge_attribute_names() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'all_edge_attribute_names' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ all_edge_attribute_names() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ all_edge_attribute_names() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ all_edge_attribute_names not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ all_edge_attribute_names() → Error: {str(e)}")
        return False, str(e)

def test_all_node_attribute_names(test_obj, nodes_data, edges_data):
    """Test Graph.all_node_attribute_names()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'all_node_attribute_names'):
            method = getattr(test_obj, 'all_node_attribute_names')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ all_node_attribute_names() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'all_node_attribute_names' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ all_node_attribute_names() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ all_node_attribute_names() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ all_node_attribute_names not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ all_node_attribute_names() → Error: {str(e)}")
        return False, str(e)

def test_bfs(test_obj, nodes_data, edges_data):
    """Test Graph.bfs(start, max_depth=None, inplace=None, attr_name=None)"""
    # Arguments for bfs(start, max_depth=None, inplace=None, attr_name=None)
    try:
        if hasattr(test_obj, 'bfs'):
            method = getattr(test_obj, 'bfs')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ bfs() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'bfs' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ bfs() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ bfs() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ bfs not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ bfs() → Error: {str(e)}")
        return False, str(e)

def test_branches(test_obj, nodes_data, edges_data):
    """Test Graph.branches()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'branches'):
            method = getattr(test_obj, 'branches')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ branches() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'branches' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ branches() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ branches() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ branches not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ branches() → Error: {str(e)}")
        return False, str(e)

def test_checkout_branch(test_obj, nodes_data, edges_data):
    """Test Graph.checkout_branch(branch_name)"""
    # Arguments for checkout_branch(branch_name)
    try:
        if hasattr(test_obj, 'checkout_branch'):
            method = getattr(test_obj, 'checkout_branch')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('test_branch')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ checkout_branch() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'checkout_branch' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ checkout_branch() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ checkout_branch() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ checkout_branch not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ checkout_branch() → Error: {str(e)}")
        return False, str(e)

def test_commit(test_obj, nodes_data, edges_data):
    """Test Graph.commit(message, author)"""
    # Arguments for commit(message, author)
    try:
        if hasattr(test_obj, 'commit'):
            method = getattr(test_obj, 'commit')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('Test commit', 'test@example.com')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ commit() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'commit' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ commit() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ commit() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ commit not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ commit() → Error: {str(e)}")
        return False, str(e)

def test_commit_history(test_obj, nodes_data, edges_data):
    """Test Graph.commit_history()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'commit_history'):
            method = getattr(test_obj, 'commit_history')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ commit_history() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'commit_history' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ commit_history() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ commit_history() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ commit_history not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ commit_history() → Error: {str(e)}")
        return False, str(e)

def test_contains_edge(test_obj, nodes_data, edges_data):
    """Test Graph.contains_edge(edge)"""
    # Arguments for contains_edge(edge)
    try:
        if hasattr(test_obj, 'contains_edge'):
            method = getattr(test_obj, 'contains_edge')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ contains_edge() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'contains_edge' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ contains_edge() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ contains_edge() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ contains_edge not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ contains_edge() → Error: {str(e)}")
        return False, str(e)

def test_contains_node(test_obj, nodes_data, edges_data):
    """Test Graph.contains_node(node)"""
    # Arguments for contains_node(node)
    try:
        if hasattr(test_obj, 'contains_node'):
            method = getattr(test_obj, 'contains_node')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ contains_node() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'contains_node' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ contains_node() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ contains_node() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ contains_node not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ contains_node() → Error: {str(e)}")
        return False, str(e)

def test_create_branch(test_obj, nodes_data, edges_data):
    """Test Graph.create_branch(branch_name)"""
    # Arguments for create_branch(branch_name)
    try:
        if hasattr(test_obj, 'create_branch'):
            method = getattr(test_obj, 'create_branch')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('test_branch')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ create_branch() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'create_branch' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ create_branch() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ create_branch() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ create_branch not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ create_branch() → Error: {str(e)}")
        return False, str(e)

def test_dense_adjacency_matrix(test_obj, nodes_data, edges_data):
    """Test Graph.dense_adjacency_matrix()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'dense_adjacency_matrix'):
            method = getattr(test_obj, 'dense_adjacency_matrix')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ dense_adjacency_matrix() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'dense_adjacency_matrix' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ dense_adjacency_matrix() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ dense_adjacency_matrix() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ dense_adjacency_matrix not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ dense_adjacency_matrix() → Error: {str(e)}")
        return False, str(e)

def test_density(test_obj, nodes_data, edges_data):
    """Test Graph.density()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'density'):
            method = getattr(test_obj, 'density')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ density() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'density' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ density() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ density() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ density not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ density() → Error: {str(e)}")
        return False, str(e)

def test_dfs(test_obj, nodes_data, edges_data):
    """Test Graph.dfs(start, max_depth=None, inplace=None, attr_name=None)"""
    # Arguments for dfs(start, max_depth=None, inplace=None, attr_name=None)
    try:
        if hasattr(test_obj, 'dfs'):
            method = getattr(test_obj, 'dfs')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ dfs() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'dfs' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ dfs() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ dfs() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ dfs not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ dfs() → Error: {str(e)}")
        return False, str(e)

def test_edge_attribute_keys(test_obj, nodes_data, edges_data):
    """Test Graph.edge_attribute_keys(edge_id)"""
    # Arguments for edge_attribute_keys(edge_id)
    try:
        if hasattr(test_obj, 'edge_attribute_keys'):
            method = getattr(test_obj, 'edge_attribute_keys')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ edge_attribute_keys() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'edge_attribute_keys' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ edge_attribute_keys() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ edge_attribute_keys() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ edge_attribute_keys not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ edge_attribute_keys() → Error: {str(e)}")
        return False, str(e)

def test_edge_count(test_obj, nodes_data, edges_data):
    """Test Graph.edge_count()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'edge_count'):
            method = getattr(test_obj, 'edge_count')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ edge_count() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'edge_count' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ edge_count() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ edge_count() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ edge_count not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ edge_count() → Error: {str(e)}")
        return False, str(e)

def test_edge_endpoints(test_obj, nodes_data, edges_data):
    """Test Graph.edge_endpoints(edge)"""
    # Arguments for edge_endpoints(edge)
    try:
        if hasattr(test_obj, 'edge_endpoints'):
            method = getattr(test_obj, 'edge_endpoints')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ edge_endpoints() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'edge_endpoints' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ edge_endpoints() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ edge_endpoints() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ edge_endpoints not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ edge_endpoints() → Error: {str(e)}")
        return False, str(e)

def test_edge_ids(test_obj, nodes_data, edges_data):
    """Test Graph.edge_ids(property)"""
    # Error getting signature: GraphArray(len=5, dtype=int64) is not a callable object
    try:
        if hasattr(test_obj, 'edge_ids'):
            method = getattr(test_obj, 'edge_ids')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ edge_ids() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'edge_ids' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ edge_ids() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ edge_ids() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ edge_ids not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ edge_ids() → Error: {str(e)}")
        return False, str(e)

def test_edges(test_obj, nodes_data, edges_data):
    """Test Graph.edges(property)"""
    # Error getting signature: <builtins.EdgesAccessor object at 0x1055228f0> is not a callable object
    try:
        if hasattr(test_obj, 'edges'):
            method = getattr(test_obj, 'edges')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ edges() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'edges' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ edges() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ edges() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ edges not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ edges() → Error: {str(e)}")
        return False, str(e)

def test_filter_edges(test_obj, nodes_data, edges_data):
    """Test Graph.filter_edges(filter)"""
    # Arguments for filter_edges(filter)
    try:
        if hasattr(test_obj, 'filter_edges'):
            method = getattr(test_obj, 'filter_edges')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(gr.EdgeFilter.attribute_equals('type', 'collaboration'))
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ filter_edges() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'filter_edges' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ filter_edges() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ filter_edges() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ filter_edges not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ filter_edges() → Error: {str(e)}")
        return False, str(e)

def test_filter_nodes(test_obj, nodes_data, edges_data):
    """Test Graph.filter_nodes(filter)"""
    # Arguments for filter_nodes(filter)
    try:
        if hasattr(test_obj, 'filter_nodes'):
            method = getattr(test_obj, 'filter_nodes')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(gr.NodeFilter.attribute_equals('name', 'Alice'))
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ filter_nodes() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'filter_nodes' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ filter_nodes() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ filter_nodes() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ filter_nodes not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ filter_nodes() → Error: {str(e)}")
        return False, str(e)

def test_get_edge_attr(test_obj, nodes_data, edges_data):
    """Test Graph.get_edge_attr(edge, attr, default=None)"""
    # Arguments for get_edge_attr(edge, attr, default=None)
    try:
        if hasattr(test_obj, 'get_edge_attr'):
            method = getattr(test_obj, 'get_edge_attr')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0, 'strength')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ get_edge_attr() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'get_edge_attr' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ get_edge_attr() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ get_edge_attr() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ get_edge_attr not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ get_edge_attr() → Error: {str(e)}")
        return False, str(e)

def test_get_edge_attrs(test_obj, nodes_data, edges_data):
    """Test Graph.get_edge_attrs(edges, attrs)"""
    # Arguments for get_edge_attrs(edges, attrs)
    try:
        if hasattr(test_obj, 'get_edge_attrs'):
            method = getattr(test_obj, 'get_edge_attrs')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(edges_data[0], ['weight', 'type'])
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ get_edge_attrs() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'get_edge_attrs' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ get_edge_attrs() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ get_edge_attrs() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ get_edge_attrs not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ get_edge_attrs() → Error: {str(e)}")
        return False, str(e)

def test_get_node_attr(test_obj, nodes_data, edges_data):
    """Test Graph.get_node_attr(node, attr, default=None)"""
    # Arguments for get_node_attr(node, attr, default=None)
    try:
        if hasattr(test_obj, 'get_node_attr'):
            method = getattr(test_obj, 'get_node_attr')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0, 'age')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ get_node_attr() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'get_node_attr' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ get_node_attr() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ get_node_attr() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ get_node_attr not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ get_node_attr() → Error: {str(e)}")
        return False, str(e)

def test_get_node_attrs(test_obj, nodes_data, edges_data):
    """Test Graph.get_node_attrs(nodes, attrs)"""
    # Arguments for get_node_attrs(nodes, attrs)
    try:
        if hasattr(test_obj, 'get_node_attrs'):
            method = getattr(test_obj, 'get_node_attrs')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(nodes_data[0], ['active', 'name'])
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ get_node_attrs() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'get_node_attrs' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ get_node_attrs() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ get_node_attrs() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ get_node_attrs not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ get_node_attrs() → Error: {str(e)}")
        return False, str(e)

def test_get_node_mapping(test_obj, nodes_data, edges_data):
    """Test Graph.get_node_mapping(uid_key, return_inverse=False)"""
    # Arguments for get_node_mapping(uid_key, return_inverse=False)
    try:
        if hasattr(test_obj, 'get_node_mapping'):
            method = getattr(test_obj, 'get_node_mapping')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('Alice')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ get_node_mapping() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'get_node_mapping' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ get_node_mapping() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ get_node_mapping() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ get_node_mapping not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ get_node_mapping() → Error: {str(e)}")
        return False, str(e)

def test_group_by(test_obj, nodes_data, edges_data):
    """Test Graph.group_by(attribute, aggregation_attr, operation)"""
    # Arguments for group_by(attribute, aggregation_attr, operation)
    try:
        if hasattr(test_obj, 'group_by'):
            method = getattr(test_obj, 'group_by')
            if callable(method):
                # TODO: Edit arguments as needed
                                # result = method(# # TODO: Need columns list)
                pass  # TODO: Fix arguments and uncomment
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ group_by() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'group_by' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ group_by() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ group_by() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ group_by not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ group_by() → Error: {str(e)}")
        return False, str(e)

def test_group_nodes_by_attribute(test_obj, nodes_data, edges_data):
    """Test Graph.group_nodes_by_attribute(attribute, aggregation_attr, operation)"""
    # Arguments for group_nodes_by_attribute(attribute, aggregation_attr, operation)
    try:
        if hasattr(test_obj, 'group_nodes_by_attribute'):
            method = getattr(test_obj, 'group_nodes_by_attribute')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('age', 'salary', 'mean')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ group_nodes_by_attribute() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'group_nodes_by_attribute' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ group_nodes_by_attribute() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ group_nodes_by_attribute() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ group_nodes_by_attribute not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ group_nodes_by_attribute() → Error: {str(e)}")
        return False, str(e)

def test_has_edge(test_obj, nodes_data, edges_data):
    """Test Graph.has_edge(edge_id)"""
    # Arguments for has_edge(edge_id)
    try:
        if hasattr(test_obj, 'has_edge'):
            method = getattr(test_obj, 'has_edge')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ has_edge() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'has_edge' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ has_edge() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ has_edge() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ has_edge not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ has_edge() → Error: {str(e)}")
        return False, str(e)

def test_has_edge_attribute(test_obj, nodes_data, edges_data):
    """Test Graph.has_edge_attribute(edge_id, attr_name)"""
    # Arguments for has_edge_attribute(edge_id, attr_name)
    try:
        if hasattr(test_obj, 'has_edge_attribute'):
            method = getattr(test_obj, 'has_edge_attribute')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0, 'strength')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ has_edge_attribute() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'has_edge_attribute' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ has_edge_attribute() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ has_edge_attribute() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ has_edge_attribute not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ has_edge_attribute() → Error: {str(e)}")
        return False, str(e)

def test_has_node(test_obj, nodes_data, edges_data):
    """Test Graph.has_node(node_id)"""
    # Arguments for has_node(node_id)
    try:
        if hasattr(test_obj, 'has_node'):
            method = getattr(test_obj, 'has_node')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ has_node() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'has_node' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ has_node() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ has_node() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ has_node not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ has_node() → Error: {str(e)}")
        return False, str(e)

def test_has_node_attribute(test_obj, nodes_data, edges_data):
    """Test Graph.has_node_attribute(node_id, attr_name)"""
    # Arguments for has_node_attribute(node_id, attr_name)
    try:
        if hasattr(test_obj, 'has_node_attribute'):
            method = getattr(test_obj, 'has_node_attribute')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0, 'level')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ has_node_attribute() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'has_node_attribute' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ has_node_attribute() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ has_node_attribute() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ has_node_attribute not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ has_node_attribute() → Error: {str(e)}")
        return False, str(e)

def test_has_uncommitted_changes(test_obj, nodes_data, edges_data):
    """Test Graph.has_uncommitted_changes()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'has_uncommitted_changes'):
            method = getattr(test_obj, 'has_uncommitted_changes')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ has_uncommitted_changes() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'has_uncommitted_changes' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ has_uncommitted_changes() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ has_uncommitted_changes() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ has_uncommitted_changes not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ has_uncommitted_changes() → Error: {str(e)}")
        return False, str(e)

def test_historical_view(test_obj, nodes_data, edges_data):
    """Test Graph.historical_view(commit_id)"""
    # Arguments for historical_view(commit_id)
    try:
        if hasattr(test_obj, 'historical_view'):
            method = getattr(test_obj, 'historical_view')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(1)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ historical_view() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'historical_view' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ historical_view() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ historical_view() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ historical_view not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ historical_view() → Error: {str(e)}")
        return False, str(e)

def test_is_connected(test_obj, nodes_data, edges_data):
    """Test Graph.is_connected()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'is_connected'):
            method = getattr(test_obj, 'is_connected')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ is_connected() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'is_connected' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ is_connected() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ is_connected() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ is_connected not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ is_connected() → Error: {str(e)}")
        return False, str(e)

def test_is_directed(test_obj, nodes_data, edges_data):
    """Test Graph.is_directed(property)"""
    # Error getting signature: False is not a callable object
    try:
        if hasattr(test_obj, 'is_directed'):
            method = getattr(test_obj, 'is_directed')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ is_directed() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'is_directed' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ is_directed() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ is_directed() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ is_directed not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ is_directed() → Error: {str(e)}")
        return False, str(e)

def test_is_undirected(test_obj, nodes_data, edges_data):
    """Test Graph.is_undirected(property)"""
    # Error getting signature: True is not a callable object
    try:
        if hasattr(test_obj, 'is_undirected'):
            method = getattr(test_obj, 'is_undirected')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ is_undirected() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'is_undirected' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ is_undirected() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ is_undirected() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ is_undirected not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ is_undirected() → Error: {str(e)}")
        return False, str(e)

def test_laplacian_matrix(test_obj, nodes_data, edges_data):
    """Test Graph.laplacian_matrix(normalized=None)"""
    # Arguments for laplacian_matrix(normalized=None)
    try:
        if hasattr(test_obj, 'laplacian_matrix'):
            method = getattr(test_obj, 'laplacian_matrix')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ laplacian_matrix() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'laplacian_matrix' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ laplacian_matrix() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ laplacian_matrix() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ laplacian_matrix not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ laplacian_matrix() → Error: {str(e)}")
        return False, str(e)

def test_neighborhood(test_obj, nodes_data, edges_data):
    """Test Graph.neighborhood(center_nodes, radius=None, max_nodes=None)"""
    # Arguments for neighborhood(center_nodes, radius=None, max_nodes=None)
    try:
        if hasattr(test_obj, 'neighborhood'):
            method = getattr(test_obj, 'neighborhood')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method([0], radius=2)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ neighborhood() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'neighborhood' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ neighborhood() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ neighborhood() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ neighborhood not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ neighborhood() → Error: {str(e)}")
        return False, str(e)

def test_neighborhood_statistics(test_obj, nodes_data, edges_data):
    """Test Graph.neighborhood_statistics()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'neighborhood_statistics'):
            method = getattr(test_obj, 'neighborhood_statistics')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ neighborhood_statistics() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'neighborhood_statistics' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ neighborhood_statistics() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ neighborhood_statistics() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ neighborhood_statistics not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ neighborhood_statistics() → Error: {str(e)}")
        return False, str(e)

def test_neighbors(test_obj, nodes_data, edges_data):
    """Test Graph.neighbors(nodes=None)"""
    # Arguments for neighbors(nodes=None)
    try:
        if hasattr(test_obj, 'neighbors'):
            method = getattr(test_obj, 'neighbors')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ neighbors() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'neighbors' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ neighbors() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ neighbors() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ neighbors not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ neighbors() → Error: {str(e)}")
        return False, str(e)

def test_node_attribute_keys(test_obj, nodes_data, edges_data):
    """Test Graph.node_attribute_keys(node_id)"""
    # Arguments for node_attribute_keys(node_id)
    try:
        if hasattr(test_obj, 'node_attribute_keys'):
            method = getattr(test_obj, 'node_attribute_keys')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ node_attribute_keys() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'node_attribute_keys' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ node_attribute_keys() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ node_attribute_keys() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ node_attribute_keys not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ node_attribute_keys() → Error: {str(e)}")
        return False, str(e)

def test_node_count(test_obj, nodes_data, edges_data):
    """Test Graph.node_count()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'node_count'):
            method = getattr(test_obj, 'node_count')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ node_count() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'node_count' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ node_count() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ node_count() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ node_count not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ node_count() → Error: {str(e)}")
        return False, str(e)

def test_node_ids(test_obj, nodes_data, edges_data):
    """Test Graph.node_ids(property)"""
    # Error getting signature: GraphArray(len=5, dtype=int64) is not a callable object
    try:
        if hasattr(test_obj, 'node_ids'):
            method = getattr(test_obj, 'node_ids')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ node_ids() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'node_ids' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ node_ids() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ node_ids() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ node_ids not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ node_ids() → Error: {str(e)}")
        return False, str(e)

def test_nodes(test_obj, nodes_data, edges_data):
    """Test Graph.nodes(property)"""
    # Error getting signature: <builtins.NodesAccessor object at 0x1055255f0> is not a callable object
    try:
        if hasattr(test_obj, 'nodes'):
            method = getattr(test_obj, 'nodes')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ nodes() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'nodes' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ nodes() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ nodes() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ nodes not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ nodes() → Error: {str(e)}")
        return False, str(e)

def test_remove_edge(test_obj, nodes_data, edges_data):
    """Test Graph.remove_edge(edge)"""
    # Arguments for remove_edge(edge)
    try:
        if hasattr(test_obj, 'remove_edge'):
            method = getattr(test_obj, 'remove_edge')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ remove_edge() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'remove_edge' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ remove_edge() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ remove_edge() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ remove_edge not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ remove_edge() → Error: {str(e)}")
        return False, str(e)

def test_remove_edges(test_obj, nodes_data, edges_data):
    """Test Graph.remove_edges(edges)"""
    # Arguments for remove_edges(edges)
    try:
        if hasattr(test_obj, 'remove_edges'):
            method = getattr(test_obj, 'remove_edges')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method([edges_data[0]])
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ remove_edges() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'remove_edges' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ remove_edges() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ remove_edges() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ remove_edges not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ remove_edges() → Error: {str(e)}")
        return False, str(e)

def test_remove_node(test_obj, nodes_data, edges_data):
    """Test Graph.remove_node(node)"""
    # Arguments for remove_node(node)
    try:
        if hasattr(test_obj, 'remove_node'):
            method = getattr(test_obj, 'remove_node')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ remove_node() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'remove_node' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ remove_node() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ remove_node() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ remove_node not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ remove_node() → Error: {str(e)}")
        return False, str(e)

def test_remove_nodes(test_obj, nodes_data, edges_data):
    """Test Graph.remove_nodes(nodes)"""
    # Arguments for remove_nodes(nodes)
    try:
        if hasattr(test_obj, 'remove_nodes'):
            method = getattr(test_obj, 'remove_nodes')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method([nodes_data[0]])
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ remove_nodes() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'remove_nodes' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ remove_nodes() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ remove_nodes() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ remove_nodes not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ remove_nodes() → Error: {str(e)}")
        return False, str(e)

def test_resolve_string_id_to_node(test_obj, nodes_data, edges_data):
    """Test Graph.resolve_string_id_to_node(string_id, uid_key)"""
    # Arguments for resolve_string_id_to_node(string_id, uid_key)
    try:
        if hasattr(test_obj, 'resolve_string_id_to_node'):
            method = getattr(test_obj, 'resolve_string_id_to_node')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('Alice', 'name')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ resolve_string_id_to_node() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'resolve_string_id_to_node' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ resolve_string_id_to_node() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ resolve_string_id_to_node() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ resolve_string_id_to_node not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ resolve_string_id_to_node() → Error: {str(e)}")
        return False, str(e)

def test_set_edge_attr(test_obj, nodes_data, edges_data):
    """Test Graph.set_edge_attr(edge, attr, value)"""
    # Arguments for set_edge_attr(edge, attr, value)
    try:
        if hasattr(test_obj, 'set_edge_attr'):
            method = getattr(test_obj, 'set_edge_attr')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(edges_data[0], 'new_attr', 'new_value')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ set_edge_attr() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'set_edge_attr' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ set_edge_attr() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ set_edge_attr() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ set_edge_attr not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ set_edge_attr() → Error: {str(e)}")
        return False, str(e)

def test_set_edge_attrs(test_obj, nodes_data, edges_data):
    """Test Graph.set_edge_attrs(attrs_dict)"""
    # Arguments for set_edge_attrs(attrs_dict)
    try:
        if hasattr(test_obj, 'set_edge_attrs'):
            method = getattr(test_obj, 'set_edge_attrs')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method({edges_data[0]: {'new_attr': 'value'}})
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ set_edge_attrs() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'set_edge_attrs' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ set_edge_attrs() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ set_edge_attrs() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ set_edge_attrs not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ set_edge_attrs() → Error: {str(e)}")
        return False, str(e)

def test_set_node_attr(test_obj, nodes_data, edges_data):
    """Test Graph.set_node_attr(node, attr, value)"""
    # Arguments for set_node_attr(node, attr, value)
    try:
        if hasattr(test_obj, 'set_node_attr'):
            method = getattr(test_obj, 'set_node_attr')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(nodes_data[0], 'new_attr', 'new_value')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ set_node_attr() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'set_node_attr' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ set_node_attr() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ set_node_attr() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ set_node_attr not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ set_node_attr() → Error: {str(e)}")
        return False, str(e)

def test_set_node_attrs(test_obj, nodes_data, edges_data):
    """Test Graph.set_node_attrs(attrs_dict)"""
    # Arguments for set_node_attrs(attrs_dict)
    try:
        if hasattr(test_obj, 'set_node_attrs'):
            method = getattr(test_obj, 'set_node_attrs')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method({nodes_data[0]: {'new_attr': 'value'}})
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ set_node_attrs() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'set_node_attrs' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ set_node_attrs() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ set_node_attrs() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ set_node_attrs not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ set_node_attrs() → Error: {str(e)}")
        return False, str(e)

def test_shortest_path(test_obj, nodes_data, edges_data):
    """Test Graph.shortest_path(source, target, weight_attribute=None, inplace=None, attr_name=None)"""
    # Arguments for shortest_path(source, target, weight_attribute=None, inplace=None, attr_name=None)
    try:
        if hasattr(test_obj, 'shortest_path'):
            method = getattr(test_obj, 'shortest_path')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0, 1)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ shortest_path() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'shortest_path' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ shortest_path() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ shortest_path() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ shortest_path not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ shortest_path() → Error: {str(e)}")
        return False, str(e)

def test_sparse_adjacency_matrix(test_obj, nodes_data, edges_data):
    """Test Graph.sparse_adjacency_matrix()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'sparse_adjacency_matrix'):
            method = getattr(test_obj, 'sparse_adjacency_matrix')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ sparse_adjacency_matrix() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'sparse_adjacency_matrix' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ sparse_adjacency_matrix() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ sparse_adjacency_matrix() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ sparse_adjacency_matrix not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ sparse_adjacency_matrix() → Error: {str(e)}")
        return False, str(e)

def test_table(test_obj, nodes_data, edges_data):
    """Test Graph.table()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'table'):
            method = getattr(test_obj, 'table')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ table() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'table' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ table() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ table() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ table not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ table() → Error: {str(e)}")
        return False, str(e)

def test_to_networkx(test_obj, nodes_data, edges_data):
    """Test Graph.to_networkx(directed: bool = False, include_attributes: bool = True)"""
    # Arguments for to_networkx(directed: bool = False, include_attributes: bool = True)
    try:
        if hasattr(test_obj, 'to_networkx'):
            method = getattr(test_obj, 'to_networkx')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ to_networkx() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'to_networkx' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ to_networkx() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ to_networkx() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ to_networkx not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ to_networkx() → Error: {str(e)}")
        return False, str(e)

def test_transition_matrix(test_obj, nodes_data, edges_data):
    """Test Graph.transition_matrix()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'transition_matrix'):
            method = getattr(test_obj, 'transition_matrix')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ transition_matrix() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'transition_matrix' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ transition_matrix() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ transition_matrix() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ transition_matrix not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ transition_matrix() → Error: {str(e)}")
        return False, str(e)

def test_view(test_obj, nodes_data, edges_data):
    """Test Graph.view()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'view'):
            method = getattr(test_obj, 'view')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ view() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'view' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ view() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ view() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ view not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ view() → Error: {str(e)}")
        return False, str(e)

def test_weighted_adjacency_matrix(test_obj, nodes_data, edges_data):
    """Test Graph.weighted_adjacency_matrix(weight_attr)"""
    # Arguments for weighted_adjacency_matrix(weight_attr)
    try:
        if hasattr(test_obj, 'weighted_adjacency_matrix'):
            method = getattr(test_obj, 'weighted_adjacency_matrix')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('type')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ weighted_adjacency_matrix() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'weighted_adjacency_matrix' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ weighted_adjacency_matrix() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ weighted_adjacency_matrix() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ weighted_adjacency_matrix not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ weighted_adjacency_matrix() → Error: {str(e)}")
        return False, str(e)

def run_all_tests():
    """Run all Graph method tests"""
    print(f"# Graph Comprehensive Test Suite")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing 71 methods\n")
    
    # Create test objects
    test_obj, nodes_data, edges_data = create_test_objects()
    
    if test_obj is None:
        print("❌ Failed to create test object")
        return
    
    results = []
    working_count = 0
    total_count = 0
    
    print(f"## Testing Graph Methods\n")
    
    # Run all method tests
    # Test __len__
    success, result = test___len__(test_obj, nodes_data, edges_data)
    results.append({'method': '__len__', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test __repr__
    success, result = test___repr__(test_obj, nodes_data, edges_data)
    results.append({'method': '__repr__', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test __str__
    success, result = test___str__(test_obj, nodes_data, edges_data)
    results.append({'method': '__str__', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test add_edge
    success, result = test_add_edge(test_obj, nodes_data, edges_data)
    results.append({'method': 'add_edge', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test add_edges
    success, result = test_add_edges(test_obj, nodes_data, edges_data)
    results.append({'method': 'add_edges', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test add_graph
    success, result = test_add_graph(test_obj, nodes_data, edges_data)
    results.append({'method': 'add_graph', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test add_node
    success, result = test_add_node(test_obj, nodes_data, edges_data)
    results.append({'method': 'add_node', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test add_nodes
    success, result = test_add_nodes(test_obj, nodes_data, edges_data)
    results.append({'method': 'add_nodes', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test adjacency
    success, result = test_adjacency(test_obj, nodes_data, edges_data)
    results.append({'method': 'adjacency', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test adjacency_matrix
    success, result = test_adjacency_matrix(test_obj, nodes_data, edges_data)
    results.append({'method': 'adjacency_matrix', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test aggregate
    success, result = test_aggregate(test_obj, nodes_data, edges_data)
    results.append({'method': 'aggregate', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test all_edge_attribute_names
    success, result = test_all_edge_attribute_names(test_obj, nodes_data, edges_data)
    results.append({'method': 'all_edge_attribute_names', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test all_node_attribute_names
    success, result = test_all_node_attribute_names(test_obj, nodes_data, edges_data)
    results.append({'method': 'all_node_attribute_names', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test bfs
    success, result = test_bfs(test_obj, nodes_data, edges_data)
    results.append({'method': 'bfs', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test branches
    success, result = test_branches(test_obj, nodes_data, edges_data)
    results.append({'method': 'branches', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test checkout_branch
    success, result = test_checkout_branch(test_obj, nodes_data, edges_data)
    results.append({'method': 'checkout_branch', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test commit
    success, result = test_commit(test_obj, nodes_data, edges_data)
    results.append({'method': 'commit', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test commit_history
    success, result = test_commit_history(test_obj, nodes_data, edges_data)
    results.append({'method': 'commit_history', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test contains_edge
    success, result = test_contains_edge(test_obj, nodes_data, edges_data)
    results.append({'method': 'contains_edge', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test contains_node
    success, result = test_contains_node(test_obj, nodes_data, edges_data)
    results.append({'method': 'contains_node', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test create_branch
    success, result = test_create_branch(test_obj, nodes_data, edges_data)
    results.append({'method': 'create_branch', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test dense_adjacency_matrix
    success, result = test_dense_adjacency_matrix(test_obj, nodes_data, edges_data)
    results.append({'method': 'dense_adjacency_matrix', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test density
    success, result = test_density(test_obj, nodes_data, edges_data)
    results.append({'method': 'density', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test dfs
    success, result = test_dfs(test_obj, nodes_data, edges_data)
    results.append({'method': 'dfs', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test edge_attribute_keys
    success, result = test_edge_attribute_keys(test_obj, nodes_data, edges_data)
    results.append({'method': 'edge_attribute_keys', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test edge_count
    success, result = test_edge_count(test_obj, nodes_data, edges_data)
    results.append({'method': 'edge_count', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test edge_endpoints
    success, result = test_edge_endpoints(test_obj, nodes_data, edges_data)
    results.append({'method': 'edge_endpoints', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test edge_ids
    success, result = test_edge_ids(test_obj, nodes_data, edges_data)
    results.append({'method': 'edge_ids', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test edges
    success, result = test_edges(test_obj, nodes_data, edges_data)
    results.append({'method': 'edges', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test filter_edges
    success, result = test_filter_edges(test_obj, nodes_data, edges_data)
    results.append({'method': 'filter_edges', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test filter_nodes
    success, result = test_filter_nodes(test_obj, nodes_data, edges_data)
    results.append({'method': 'filter_nodes', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test get_edge_attr
    success, result = test_get_edge_attr(test_obj, nodes_data, edges_data)
    results.append({'method': 'get_edge_attr', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test get_edge_attrs
    success, result = test_get_edge_attrs(test_obj, nodes_data, edges_data)
    results.append({'method': 'get_edge_attrs', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test get_node_attr
    success, result = test_get_node_attr(test_obj, nodes_data, edges_data)
    results.append({'method': 'get_node_attr', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test get_node_attrs
    success, result = test_get_node_attrs(test_obj, nodes_data, edges_data)
    results.append({'method': 'get_node_attrs', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test get_node_mapping
    success, result = test_get_node_mapping(test_obj, nodes_data, edges_data)
    results.append({'method': 'get_node_mapping', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test group_by
    success, result = test_group_by(test_obj, nodes_data, edges_data)
    results.append({'method': 'group_by', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test group_nodes_by_attribute
    success, result = test_group_nodes_by_attribute(test_obj, nodes_data, edges_data)
    results.append({'method': 'group_nodes_by_attribute', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test has_edge
    success, result = test_has_edge(test_obj, nodes_data, edges_data)
    results.append({'method': 'has_edge', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test has_edge_attribute
    success, result = test_has_edge_attribute(test_obj, nodes_data, edges_data)
    results.append({'method': 'has_edge_attribute', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test has_node
    success, result = test_has_node(test_obj, nodes_data, edges_data)
    results.append({'method': 'has_node', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test has_node_attribute
    success, result = test_has_node_attribute(test_obj, nodes_data, edges_data)
    results.append({'method': 'has_node_attribute', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test has_uncommitted_changes
    success, result = test_has_uncommitted_changes(test_obj, nodes_data, edges_data)
    results.append({'method': 'has_uncommitted_changes', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test historical_view
    success, result = test_historical_view(test_obj, nodes_data, edges_data)
    results.append({'method': 'historical_view', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test is_connected
    success, result = test_is_connected(test_obj, nodes_data, edges_data)
    results.append({'method': 'is_connected', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test is_directed
    success, result = test_is_directed(test_obj, nodes_data, edges_data)
    results.append({'method': 'is_directed', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test is_undirected
    success, result = test_is_undirected(test_obj, nodes_data, edges_data)
    results.append({'method': 'is_undirected', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test laplacian_matrix
    success, result = test_laplacian_matrix(test_obj, nodes_data, edges_data)
    results.append({'method': 'laplacian_matrix', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test neighborhood
    success, result = test_neighborhood(test_obj, nodes_data, edges_data)
    results.append({'method': 'neighborhood', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test neighborhood_statistics
    success, result = test_neighborhood_statistics(test_obj, nodes_data, edges_data)
    results.append({'method': 'neighborhood_statistics', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test neighbors
    success, result = test_neighbors(test_obj, nodes_data, edges_data)
    results.append({'method': 'neighbors', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test node_attribute_keys
    success, result = test_node_attribute_keys(test_obj, nodes_data, edges_data)
    results.append({'method': 'node_attribute_keys', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test node_count
    success, result = test_node_count(test_obj, nodes_data, edges_data)
    results.append({'method': 'node_count', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test node_ids
    success, result = test_node_ids(test_obj, nodes_data, edges_data)
    results.append({'method': 'node_ids', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test nodes
    success, result = test_nodes(test_obj, nodes_data, edges_data)
    results.append({'method': 'nodes', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test remove_edge
    success, result = test_remove_edge(test_obj, nodes_data, edges_data)
    results.append({'method': 'remove_edge', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test remove_edges
    success, result = test_remove_edges(test_obj, nodes_data, edges_data)
    results.append({'method': 'remove_edges', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test remove_node
    success, result = test_remove_node(test_obj, nodes_data, edges_data)
    results.append({'method': 'remove_node', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test remove_nodes
    success, result = test_remove_nodes(test_obj, nodes_data, edges_data)
    results.append({'method': 'remove_nodes', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test resolve_string_id_to_node
    success, result = test_resolve_string_id_to_node(test_obj, nodes_data, edges_data)
    results.append({'method': 'resolve_string_id_to_node', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test set_edge_attr
    success, result = test_set_edge_attr(test_obj, nodes_data, edges_data)
    results.append({'method': 'set_edge_attr', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test set_edge_attrs
    success, result = test_set_edge_attrs(test_obj, nodes_data, edges_data)
    results.append({'method': 'set_edge_attrs', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test set_node_attr
    success, result = test_set_node_attr(test_obj, nodes_data, edges_data)
    results.append({'method': 'set_node_attr', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test set_node_attrs
    success, result = test_set_node_attrs(test_obj, nodes_data, edges_data)
    results.append({'method': 'set_node_attrs', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test shortest_path
    success, result = test_shortest_path(test_obj, nodes_data, edges_data)
    results.append({'method': 'shortest_path', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test sparse_adjacency_matrix
    success, result = test_sparse_adjacency_matrix(test_obj, nodes_data, edges_data)
    results.append({'method': 'sparse_adjacency_matrix', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test table
    success, result = test_table(test_obj, nodes_data, edges_data)
    results.append({'method': 'table', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test to_networkx
    success, result = test_to_networkx(test_obj, nodes_data, edges_data)
    results.append({'method': 'to_networkx', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test transition_matrix
    success, result = test_transition_matrix(test_obj, nodes_data, edges_data)
    results.append({'method': 'transition_matrix', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test view
    success, result = test_view(test_obj, nodes_data, edges_data)
    results.append({'method': 'view', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test weighted_adjacency_matrix
    success, result = test_weighted_adjacency_matrix(test_obj, nodes_data, edges_data)
    results.append({'method': 'weighted_adjacency_matrix', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    
    # Print summary
    print(f"\n# Graph Test Summary")
    print(f"**Results**: {working_count}/{total_count} methods working ({working_count/total_count*100:.1f}%)")
    
    # Show working methods
    working = [r for r in results if r['success']]
    failing = [r for r in results if not r['success']]
    
    print(f"\n**Working Methods ({len(working)}):**")
    for r in working:  # Show all
        print(f"  ✅ {r['method']}")
    
    print(f"\n**Failing Methods ({len(failing)}):**")  
    for r in failing:  # Show all
        print(f"  ❌ {r['method']}: {r['result']}")
    
    return results

if __name__ == "__main__":
    results = run_all_tests()
