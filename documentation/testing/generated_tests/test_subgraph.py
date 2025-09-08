#!/usr/bin/env python3
"""
Comprehensive test script for Groggy Subgraph
Generated on: 2025-09-07 21:42:37

This script tests ALL methods of the Subgraph class with proper argument patterns.
Edit the TODO sections to provide correct arguments for each method.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr
import traceback
from datetime import datetime

def create_test_objects():
    """Create test objects for Subgraph testing"""
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
    
    # Create the specific test object for Subgraph
    test_obj = g.view()
    
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


def test___getitem__(test_obj, nodes_data, edges_data):
    """Test Subgraph.__getitem__(key, /)"""
    # Error getting signature: local variable 'node_attrs' referenced before assignment
    try:
        if hasattr(test_obj, '__getitem__'):
            method = getattr(test_obj, '__getitem__')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ __getitem__() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if '__getitem__' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ __getitem__() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ __getitem__() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ __getitem__ not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ __getitem__() → Error: {str(e)}")
        return False, str(e)

def test___len__(test_obj, nodes_data, edges_data):
    """Test Subgraph.__len__()"""
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
    """Test Subgraph.__repr__()"""
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
    """Test Subgraph.__str__()"""
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

def test_bfs(test_obj, nodes_data, edges_data):
    """Test Subgraph.bfs(start, max_depth=None)"""
    # Arguments for bfs(start, max_depth=None)
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

def test_calculate_similarity(test_obj, nodes_data, edges_data):
    """Test Subgraph.calculate_similarity(other, metric='jaccard')"""
    # Arguments for calculate_similarity(other, metric='jaccard')
    try:
        if hasattr(test_obj, 'calculate_similarity'):
            method = getattr(test_obj, 'calculate_similarity')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(g.view())
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ calculate_similarity() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'calculate_similarity' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ calculate_similarity() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ calculate_similarity() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ calculate_similarity not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ calculate_similarity() → Error: {str(e)}")
        return False, str(e)

def test_child_meta_nodes(test_obj, nodes_data, edges_data):
    """Test Subgraph.child_meta_nodes()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'child_meta_nodes'):
            method = getattr(test_obj, 'child_meta_nodes')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ child_meta_nodes() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'child_meta_nodes' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ child_meta_nodes() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ child_meta_nodes() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ child_meta_nodes not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ child_meta_nodes() → Error: {str(e)}")
        return False, str(e)

def test_clustering_coefficient(test_obj, nodes_data, edges_data):
    """Test Subgraph.clustering_coefficient(_node_id=None)"""
    # Arguments for clustering_coefficient(_node_id=None)
    try:
        if hasattr(test_obj, 'clustering_coefficient'):
            method = getattr(test_obj, 'clustering_coefficient')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ clustering_coefficient() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'clustering_coefficient' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ clustering_coefficient() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ clustering_coefficient() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ clustering_coefficient not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ clustering_coefficient() → Error: {str(e)}")
        return False, str(e)

def test_collapse(test_obj, nodes_data, edges_data):
    """Test Subgraph.collapse(node_aggs=None, edge_aggs=None, edge_strategy='aggregate', node_strategy='extract', preset=None, include_edge_count=True, mark_entity_type=True, entity_type='meta', allow_missing_attributes=True)"""
    # Arguments for collapse(node_aggs=None, edge_aggs=None, edge_strategy='aggregate', node_strategy='extract', preset=None, include_edge_count=True, mark_entity_type=True, entity_type='meta', allow_missing_attributes=True)
    try:
        if hasattr(test_obj, 'collapse'):
            method = getattr(test_obj, 'collapse')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ collapse() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'collapse' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ collapse() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ collapse() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ collapse not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ collapse() → Error: {str(e)}")
        return False, str(e)

def test_collapse_to_node(test_obj, nodes_data, edges_data):
    """Test Subgraph.collapse_to_node(agg_functions)"""
    # Arguments for collapse_to_node(agg_functions)
    try:
        if hasattr(test_obj, 'collapse_to_node'):
            method = getattr(test_obj, 'collapse_to_node')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('collapsed_node', {'salary': 'mean', 'age': 'max'})
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ collapse_to_node() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'collapse_to_node' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ collapse_to_node() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ collapse_to_node() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ collapse_to_node not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ collapse_to_node() → Error: {str(e)}")
        return False, str(e)

def test_collapse_to_node_with_defaults(test_obj, nodes_data, edges_data):
    """Test Subgraph.collapse_to_node_with_defaults(agg_functions, defaults=None)"""
    # Arguments for collapse_to_node_with_defaults(agg_functions, defaults=None)
    try:
        if hasattr(test_obj, 'collapse_to_node_with_defaults'):
            method = getattr(test_obj, 'collapse_to_node_with_defaults')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('collapsed_node', {'salary': 'mean', 'age': 'max'})
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ collapse_to_node_with_defaults() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'collapse_to_node_with_defaults' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ collapse_to_node_with_defaults() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ collapse_to_node_with_defaults() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ collapse_to_node_with_defaults not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ collapse_to_node_with_defaults() → Error: {str(e)}")
        return False, str(e)

def test_connected_components(test_obj, nodes_data, edges_data):
    """Test Subgraph.connected_components()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'connected_components'):
            method = getattr(test_obj, 'connected_components')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ connected_components() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'connected_components' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ connected_components() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ connected_components() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ connected_components not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ connected_components() → Error: {str(e)}")
        return False, str(e)

def test_contains_edge(test_obj, nodes_data, edges_data):
    """Test Subgraph.contains_edge(edge_id)"""
    # Arguments for contains_edge(edge_id)
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
    """Test Subgraph.contains_node(node_id)"""
    # Arguments for contains_node(node_id)
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

def test_degree(test_obj, nodes_data, edges_data):
    """Test Subgraph.degree(nodes=None, *, full_graph=False)"""
    # Arguments for degree(nodes=None, *, full_graph=False)
    try:
        if hasattr(test_obj, 'degree'):
            method = getattr(test_obj, 'degree')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ degree() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'degree' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ degree() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ degree() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ degree not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ degree() → Error: {str(e)}")
        return False, str(e)

def test_density(test_obj, nodes_data, edges_data):
    """Test Subgraph.density()"""
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
    """Test Subgraph.dfs(start, max_depth=None)"""
    # Arguments for dfs(start, max_depth=None)
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

def test_edge_count(test_obj, nodes_data, edges_data):
    """Test Subgraph.edge_count()"""
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
    """Test Subgraph.edge_endpoints(edge_id)"""
    # Arguments for edge_endpoints(edge_id)
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
    """Test Subgraph.edge_ids(property)"""
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
    """Test Subgraph.edges(property)"""
    # Error getting signature: <builtins.EdgesAccessor object at 0x105227fb0> is not a callable object
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

def test_edges_table(test_obj, nodes_data, edges_data):
    """Test Subgraph.edges_table()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'edges_table'):
            method = getattr(test_obj, 'edges_table')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ edges_table() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'edges_table' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ edges_table() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ edges_table() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ edges_table not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ edges_table() → Error: {str(e)}")
        return False, str(e)

def test_entity_type(test_obj, nodes_data, edges_data):
    """Test Subgraph.entity_type()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'entity_type'):
            method = getattr(test_obj, 'entity_type')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ entity_type() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'entity_type' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ entity_type() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ entity_type() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ entity_type not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ entity_type() → Error: {str(e)}")
        return False, str(e)

def test_filter_edges(test_obj, nodes_data, edges_data):
    """Test Subgraph.filter_edges(filter)"""
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
    """Test Subgraph.filter_nodes(filter)"""
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

def test_get_edge_attribute(test_obj, nodes_data, edges_data):
    """Test Subgraph.get_edge_attribute(edge_id, attr_name)"""
    # Arguments for get_edge_attribute(edge_id, attr_name)
    try:
        if hasattr(test_obj, 'get_edge_attribute'):
            method = getattr(test_obj, 'get_edge_attribute')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0, 'attr_name')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ get_edge_attribute() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'get_edge_attribute' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ get_edge_attribute() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ get_edge_attribute() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ get_edge_attribute not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ get_edge_attribute() → Error: {str(e)}")
        return False, str(e)

def test_get_node_attribute(test_obj, nodes_data, edges_data):
    """Test Subgraph.get_node_attribute(node_id, attr_name)"""
    # Arguments for get_node_attribute(node_id, attr_name)
    try:
        if hasattr(test_obj, 'get_node_attribute'):
            method = getattr(test_obj, 'get_node_attribute')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0, 'attr_name')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ get_node_attribute() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'get_node_attribute' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ get_node_attribute() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ get_node_attribute() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ get_node_attribute not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ get_node_attribute() → Error: {str(e)}")
        return False, str(e)

def test_has_edge(test_obj, nodes_data, edges_data):
    """Test Subgraph.has_edge(edge_id)"""
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

def test_has_edge_between(test_obj, nodes_data, edges_data):
    """Test Subgraph.has_edge_between(source, target)"""
    # Arguments for has_edge_between(source, target)
    try:
        if hasattr(test_obj, 'has_edge_between'):
            method = getattr(test_obj, 'has_edge_between')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0, 1)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ has_edge_between() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'has_edge_between' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ has_edge_between() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ has_edge_between() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ has_edge_between not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ has_edge_between() → Error: {str(e)}")
        return False, str(e)

def test_has_meta_nodes(test_obj, nodes_data, edges_data):
    """Test Subgraph.has_meta_nodes()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'has_meta_nodes'):
            method = getattr(test_obj, 'has_meta_nodes')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ has_meta_nodes() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'has_meta_nodes' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ has_meta_nodes() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ has_meta_nodes() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ has_meta_nodes not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ has_meta_nodes() → Error: {str(e)}")
        return False, str(e)

def test_has_node(test_obj, nodes_data, edges_data):
    """Test Subgraph.has_node(node_id)"""
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

def test_has_path(test_obj, nodes_data, edges_data):
    """Test Subgraph.has_path(node1_id, node2_id)"""
    # Arguments for has_path(node1_id, node2_id)
    try:
        if hasattr(test_obj, 'has_path'):
            method = getattr(test_obj, 'has_path')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0, 1)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ has_path() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'has_path' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ has_path() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ has_path() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ has_path not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ has_path() → Error: {str(e)}")
        return False, str(e)

def test_hierarchy_level(test_obj, nodes_data, edges_data):
    """Test Subgraph.hierarchy_level(property)"""
    # Error getting signature: 0 is not a callable object
    try:
        if hasattr(test_obj, 'hierarchy_level'):
            method = getattr(test_obj, 'hierarchy_level')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ hierarchy_level() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'hierarchy_level' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ hierarchy_level() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ hierarchy_level() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ hierarchy_level not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ hierarchy_level() → Error: {str(e)}")
        return False, str(e)

def test_in_degree(test_obj, nodes_data, edges_data):
    """Test Subgraph.in_degree(nodes=None, full_graph=False)"""
    # Arguments for in_degree(nodes=None, full_graph=False)
    try:
        if hasattr(test_obj, 'in_degree'):
            method = getattr(test_obj, 'in_degree')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ in_degree() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'in_degree' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ in_degree() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ in_degree() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ in_degree not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ in_degree() → Error: {str(e)}")
        return False, str(e)

def test_induced_subgraph(test_obj, nodes_data, edges_data):
    """Test Subgraph.induced_subgraph(nodes)"""
    # Arguments for induced_subgraph(nodes)
    try:
        if hasattr(test_obj, 'induced_subgraph'):
            method = getattr(test_obj, 'induced_subgraph')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method([0, 1])
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ induced_subgraph() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'induced_subgraph' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ induced_subgraph() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ induced_subgraph() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ induced_subgraph not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ induced_subgraph() → Error: {str(e)}")
        return False, str(e)

def test_intersect_with(test_obj, nodes_data, edges_data):
    """Test Subgraph.intersect_with(_other)"""
    # Arguments for intersect_with(_other)
    try:
        if hasattr(test_obj, 'intersect_with'):
            method = getattr(test_obj, 'intersect_with')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(g.view())
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ intersect_with() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'intersect_with' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ intersect_with() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ intersect_with() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ intersect_with not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ intersect_with() → Error: {str(e)}")
        return False, str(e)

def test_is_connected(test_obj, nodes_data, edges_data):
    """Test Subgraph.is_connected()"""
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

def test_is_empty(test_obj, nodes_data, edges_data):
    """Test Subgraph.is_empty()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'is_empty'):
            method = getattr(test_obj, 'is_empty')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ is_empty() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'is_empty' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ is_empty() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ is_empty() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ is_empty not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ is_empty() → Error: {str(e)}")
        return False, str(e)

def test_merge_with(test_obj, nodes_data, edges_data):
    """Test Subgraph.merge_with(_other)"""
    # Arguments for merge_with(_other)
    try:
        if hasattr(test_obj, 'merge_with'):
            method = getattr(test_obj, 'merge_with')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(g.view())
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ merge_with() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'merge_with' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ merge_with() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ merge_with() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ merge_with not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ merge_with() → Error: {str(e)}")
        return False, str(e)

def test_meta_nodes(test_obj, nodes_data, edges_data):
    """Test Subgraph.meta_nodes()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'meta_nodes'):
            method = getattr(test_obj, 'meta_nodes')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ meta_nodes() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'meta_nodes' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ meta_nodes() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ meta_nodes() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ meta_nodes not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ meta_nodes() → Error: {str(e)}")
        return False, str(e)

def test_neighborhood(test_obj, nodes_data, edges_data):
    """Test Subgraph.neighborhood(central_nodes, hops)"""
    # Arguments for neighborhood(central_nodes, hops)
    try:
        if hasattr(test_obj, 'neighborhood'):
            method = getattr(test_obj, 'neighborhood')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method([0], 2)
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

def test_neighbors(test_obj, nodes_data, edges_data):
    """Test Subgraph.neighbors(node_id)"""
    # Arguments for neighbors(node_id)
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

def test_node_count(test_obj, nodes_data, edges_data):
    """Test Subgraph.node_count()"""
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
    """Test Subgraph.node_ids(property)"""
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
    """Test Subgraph.nodes(property)"""
    # Error getting signature: <builtins.NodesAccessor object at 0x105547e70> is not a callable object
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

def test_out_degree(test_obj, nodes_data, edges_data):
    """Test Subgraph.out_degree(nodes=None, full_graph=False)"""
    # Arguments for out_degree(nodes=None, full_graph=False)
    try:
        if hasattr(test_obj, 'out_degree'):
            method = getattr(test_obj, 'out_degree')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ out_degree() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'out_degree' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ out_degree() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ out_degree() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ out_degree not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ out_degree() → Error: {str(e)}")
        return False, str(e)

def test_parent_meta_node(test_obj, nodes_data, edges_data):
    """Test Subgraph.parent_meta_node()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'parent_meta_node'):
            method = getattr(test_obj, 'parent_meta_node')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ parent_meta_node() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'parent_meta_node' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ parent_meta_node() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ parent_meta_node() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ parent_meta_node not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ parent_meta_node() → Error: {str(e)}")
        return False, str(e)

def test_set_edge_attrs(test_obj, nodes_data, edges_data):
    """Test Subgraph.set_edge_attrs(attrs_dict)"""
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

def test_set_node_attrs(test_obj, nodes_data, edges_data):
    """Test Subgraph.set_node_attrs(attrs_dict)"""
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

def test_shortest_path_subgraph(test_obj, nodes_data, edges_data):
    """Test Subgraph.shortest_path_subgraph(source, target)"""
    # Arguments for shortest_path_subgraph(source, target)
    try:
        if hasattr(test_obj, 'shortest_path_subgraph'):
            method = getattr(test_obj, 'shortest_path_subgraph')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(0, 1)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ shortest_path_subgraph() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'shortest_path_subgraph' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ shortest_path_subgraph() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ shortest_path_subgraph() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ shortest_path_subgraph not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ shortest_path_subgraph() → Error: {str(e)}")
        return False, str(e)

def test_subgraph_from_edges(test_obj, nodes_data, edges_data):
    """Test Subgraph.subgraph_from_edges(edges)"""
    # Arguments for subgraph_from_edges(edges)
    try:
        if hasattr(test_obj, 'subgraph_from_edges'):
            method = getattr(test_obj, 'subgraph_from_edges')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method([0])
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ subgraph_from_edges() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'subgraph_from_edges' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ subgraph_from_edges() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ subgraph_from_edges() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ subgraph_from_edges not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ subgraph_from_edges() → Error: {str(e)}")
        return False, str(e)

def test_subtract_from(test_obj, nodes_data, edges_data):
    """Test Subgraph.subtract_from(_other)"""
    # Arguments for subtract_from(_other)
    try:
        if hasattr(test_obj, 'subtract_from'):
            method = getattr(test_obj, 'subtract_from')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(g.view())
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ subtract_from() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'subtract_from' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ subtract_from() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ subtract_from() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ subtract_from not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ subtract_from() → Error: {str(e)}")
        return False, str(e)

def test_summary(test_obj, nodes_data, edges_data):
    """Test Subgraph.summary()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'summary'):
            method = getattr(test_obj, 'summary')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ summary() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'summary' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ summary() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ summary() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ summary not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ summary() → Error: {str(e)}")
        return False, str(e)

def test_table(test_obj, nodes_data, edges_data):
    """Test Subgraph.table()"""
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

def test_to_graph(test_obj, nodes_data, edges_data):
    """Test Subgraph.to_graph()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'to_graph'):
            method = getattr(test_obj, 'to_graph')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ to_graph() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'to_graph' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ to_graph() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ to_graph() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ to_graph not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ to_graph() → Error: {str(e)}")
        return False, str(e)

def test_to_networkx(test_obj, nodes_data, edges_data):
    """Test Subgraph.to_networkx()"""
    # No arguments needed
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

def test_transitivity(test_obj, nodes_data, edges_data):
    """Test Subgraph.transitivity()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'transitivity'):
            method = getattr(test_obj, 'transitivity')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ transitivity() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'transitivity' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ transitivity() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ transitivity() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ transitivity not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ transitivity() → Error: {str(e)}")
        return False, str(e)

def run_all_tests():
    """Run all Subgraph method tests"""
    print(f"# Subgraph Comprehensive Test Suite")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing 57 methods\n")
    
    # Create test objects
    test_obj, nodes_data, edges_data = create_test_objects()
    
    if test_obj is None:
        print("❌ Failed to create test object")
        return
    
    results = []
    working_count = 0
    total_count = 0
    
    print(f"## Testing Subgraph Methods\n")
    
    # Run all method tests
    # Test __getitem__
    success, result = test___getitem__(test_obj, nodes_data, edges_data)
    results.append({'method': '__getitem__', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
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
    
    # Test bfs
    success, result = test_bfs(test_obj, nodes_data, edges_data)
    results.append({'method': 'bfs', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test calculate_similarity
    success, result = test_calculate_similarity(test_obj, nodes_data, edges_data)
    results.append({'method': 'calculate_similarity', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test child_meta_nodes
    success, result = test_child_meta_nodes(test_obj, nodes_data, edges_data)
    results.append({'method': 'child_meta_nodes', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test clustering_coefficient
    success, result = test_clustering_coefficient(test_obj, nodes_data, edges_data)
    results.append({'method': 'clustering_coefficient', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test collapse
    success, result = test_collapse(test_obj, nodes_data, edges_data)
    results.append({'method': 'collapse', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test collapse_to_node
    success, result = test_collapse_to_node(test_obj, nodes_data, edges_data)
    results.append({'method': 'collapse_to_node', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test collapse_to_node_with_defaults
    success, result = test_collapse_to_node_with_defaults(test_obj, nodes_data, edges_data)
    results.append({'method': 'collapse_to_node_with_defaults', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test connected_components
    success, result = test_connected_components(test_obj, nodes_data, edges_data)
    results.append({'method': 'connected_components', 'success': success, 'result': result})
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
    
    # Test degree
    success, result = test_degree(test_obj, nodes_data, edges_data)
    results.append({'method': 'degree', 'success': success, 'result': result})
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
    
    # Test edges_table
    success, result = test_edges_table(test_obj, nodes_data, edges_data)
    results.append({'method': 'edges_table', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test entity_type
    success, result = test_entity_type(test_obj, nodes_data, edges_data)
    results.append({'method': 'entity_type', 'success': success, 'result': result})
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
    
    # Test get_edge_attribute
    success, result = test_get_edge_attribute(test_obj, nodes_data, edges_data)
    results.append({'method': 'get_edge_attribute', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test get_node_attribute
    success, result = test_get_node_attribute(test_obj, nodes_data, edges_data)
    results.append({'method': 'get_node_attribute', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test has_edge
    success, result = test_has_edge(test_obj, nodes_data, edges_data)
    results.append({'method': 'has_edge', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test has_edge_between
    success, result = test_has_edge_between(test_obj, nodes_data, edges_data)
    results.append({'method': 'has_edge_between', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test has_meta_nodes
    success, result = test_has_meta_nodes(test_obj, nodes_data, edges_data)
    results.append({'method': 'has_meta_nodes', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test has_node
    success, result = test_has_node(test_obj, nodes_data, edges_data)
    results.append({'method': 'has_node', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test has_path
    success, result = test_has_path(test_obj, nodes_data, edges_data)
    results.append({'method': 'has_path', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test hierarchy_level
    success, result = test_hierarchy_level(test_obj, nodes_data, edges_data)
    results.append({'method': 'hierarchy_level', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test in_degree
    success, result = test_in_degree(test_obj, nodes_data, edges_data)
    results.append({'method': 'in_degree', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test induced_subgraph
    success, result = test_induced_subgraph(test_obj, nodes_data, edges_data)
    results.append({'method': 'induced_subgraph', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test intersect_with
    success, result = test_intersect_with(test_obj, nodes_data, edges_data)
    results.append({'method': 'intersect_with', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test is_connected
    success, result = test_is_connected(test_obj, nodes_data, edges_data)
    results.append({'method': 'is_connected', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test is_empty
    success, result = test_is_empty(test_obj, nodes_data, edges_data)
    results.append({'method': 'is_empty', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test merge_with
    success, result = test_merge_with(test_obj, nodes_data, edges_data)
    results.append({'method': 'merge_with', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test meta_nodes
    success, result = test_meta_nodes(test_obj, nodes_data, edges_data)
    results.append({'method': 'meta_nodes', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test neighborhood
    success, result = test_neighborhood(test_obj, nodes_data, edges_data)
    results.append({'method': 'neighborhood', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test neighbors
    success, result = test_neighbors(test_obj, nodes_data, edges_data)
    results.append({'method': 'neighbors', 'success': success, 'result': result})
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
    
    # Test out_degree
    success, result = test_out_degree(test_obj, nodes_data, edges_data)
    results.append({'method': 'out_degree', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test parent_meta_node
    success, result = test_parent_meta_node(test_obj, nodes_data, edges_data)
    results.append({'method': 'parent_meta_node', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test set_edge_attrs
    success, result = test_set_edge_attrs(test_obj, nodes_data, edges_data)
    results.append({'method': 'set_edge_attrs', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test set_node_attrs
    success, result = test_set_node_attrs(test_obj, nodes_data, edges_data)
    results.append({'method': 'set_node_attrs', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test shortest_path_subgraph
    success, result = test_shortest_path_subgraph(test_obj, nodes_data, edges_data)
    results.append({'method': 'shortest_path_subgraph', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test subgraph_from_edges
    success, result = test_subgraph_from_edges(test_obj, nodes_data, edges_data)
    results.append({'method': 'subgraph_from_edges', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test subtract_from
    success, result = test_subtract_from(test_obj, nodes_data, edges_data)
    results.append({'method': 'subtract_from', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test summary
    success, result = test_summary(test_obj, nodes_data, edges_data)
    results.append({'method': 'summary', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test table
    success, result = test_table(test_obj, nodes_data, edges_data)
    results.append({'method': 'table', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test to_graph
    success, result = test_to_graph(test_obj, nodes_data, edges_data)
    results.append({'method': 'to_graph', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test to_networkx
    success, result = test_to_networkx(test_obj, nodes_data, edges_data)
    results.append({'method': 'to_networkx', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test transitivity
    success, result = test_transitivity(test_obj, nodes_data, edges_data)
    results.append({'method': 'transitivity', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    
    # Print summary
    print(f"\n# Subgraph Test Summary")
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
