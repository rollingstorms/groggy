#!/usr/bin/env python3
"""
Comprehensive test script for Groggy GraphTable
Generated on: 2025-09-07 21:42:37

This script tests ALL methods of the GraphTable class with proper argument patterns.
Edit the TODO sections to provide correct arguments for each method.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr
import traceback
from datetime import datetime

def create_test_objects():
    """Create test objects for GraphTable testing"""
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
    
    # Create the specific test object for GraphTable
    test_obj = g.table()
    
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
    """Test GraphTable.__getitem__(key, /)"""
    # Arguments for __getitem__(key, /)
    try:
        if hasattr(test_obj, '__getitem__'):
            method = getattr(test_obj, '__getitem__')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('name')
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
    """Test GraphTable.__len__()"""
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
    """Test GraphTable.__repr__()"""
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
    """Test GraphTable.__str__()"""
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

def test_edges(test_obj, nodes_data, edges_data):
    """Test GraphTable.edges(property)"""
    # Error getting signature: EdgesTable[5 x 6] is not a callable object
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

def test_from_federated_bundles(test_obj, nodes_data, edges_data):
    """Test GraphTable.from_federated_bundles(bundle_paths, domain_names=None)"""
    # Arguments for from_federated_bundles(bundle_paths, domain_names=None)
    try:
        if hasattr(test_obj, 'from_federated_bundles'):
            method = getattr(test_obj, 'from_federated_bundles')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(['bundle1.json', 'bundle2.json'])
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ from_federated_bundles() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'from_federated_bundles' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ from_federated_bundles() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ from_federated_bundles() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ from_federated_bundles not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ from_federated_bundles() → Error: {str(e)}")
        return False, str(e)

def test_head(test_obj, nodes_data, edges_data):
    """Test GraphTable.head(n=5)"""
    # Arguments for head(n=5)
    try:
        if hasattr(test_obj, 'head'):
            method = getattr(test_obj, 'head')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(5)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ head() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'head' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ head() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ head() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ head not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ head() → Error: {str(e)}")
        return False, str(e)

def test_load_bundle(test_obj, nodes_data, edges_data):
    """Test GraphTable.load_bundle(path)"""
    # Arguments for load_bundle(path)
    try:
        if hasattr(test_obj, 'load_bundle'):
            method = getattr(test_obj, 'load_bundle')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('test_bundle.json')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ load_bundle() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'load_bundle' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ load_bundle() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ load_bundle() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ load_bundle not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ load_bundle() → Error: {str(e)}")
        return False, str(e)

def test_merge(test_obj, nodes_data, edges_data):
    """Test GraphTable.merge(tables)"""
    # Arguments for merge(tables)
    try:
        if hasattr(test_obj, 'merge'):
            method = getattr(test_obj, 'merge')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(test_obj)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ merge() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'merge' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ merge() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ merge() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ merge not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ merge() → Error: {str(e)}")
        return False, str(e)

def test_merge_with(test_obj, nodes_data, edges_data):
    """Test GraphTable.merge_with(other, strategy)"""
    # Arguments for merge_with(other, strategy)
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

def test_merge_with_strategy(test_obj, nodes_data, edges_data):
    """Test GraphTable.merge_with_strategy(tables, strategy)"""
    # Arguments for merge_with_strategy(tables, strategy)
    try:
        if hasattr(test_obj, 'merge_with_strategy'):
            method = getattr(test_obj, 'merge_with_strategy')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(test_obj, 'left')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ merge_with_strategy() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'merge_with_strategy' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ merge_with_strategy() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ merge_with_strategy() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ merge_with_strategy not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ merge_with_strategy() → Error: {str(e)}")
        return False, str(e)

def test_ncols(test_obj, nodes_data, edges_data):
    """Test GraphTable.ncols()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'ncols'):
            method = getattr(test_obj, 'ncols')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ ncols() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'ncols' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ ncols() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ ncols() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ ncols not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ ncols() → Error: {str(e)}")
        return False, str(e)

def test_nodes(test_obj, nodes_data, edges_data):
    """Test GraphTable.nodes(property)"""
    # Error getting signature: NodesTable[5 x 7] is not a callable object
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

def test_nrows(test_obj, nodes_data, edges_data):
    """Test GraphTable.nrows()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'nrows'):
            method = getattr(test_obj, 'nrows')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ nrows() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'nrows' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ nrows() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ nrows() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ nrows not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ nrows() → Error: {str(e)}")
        return False, str(e)

def test_save_bundle(test_obj, nodes_data, edges_data):
    """Test GraphTable.save_bundle(path)"""
    # Arguments for save_bundle(path)
    try:
        if hasattr(test_obj, 'save_bundle'):
            method = getattr(test_obj, 'save_bundle')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method('test_bundle.json')
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ save_bundle() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'save_bundle' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ save_bundle() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ save_bundle() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ save_bundle not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ save_bundle() → Error: {str(e)}")
        return False, str(e)

def test_shape(test_obj, nodes_data, edges_data):
    """Test GraphTable.shape()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'shape'):
            method = getattr(test_obj, 'shape')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ shape() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'shape' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ shape() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ shape() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ shape not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ shape() → Error: {str(e)}")
        return False, str(e)

def test_stats(test_obj, nodes_data, edges_data):
    """Test GraphTable.stats()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'stats'):
            method = getattr(test_obj, 'stats')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ stats() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'stats' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ stats() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ stats() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ stats not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ stats() → Error: {str(e)}")
        return False, str(e)

def test_tail(test_obj, nodes_data, edges_data):
    """Test GraphTable.tail(n=5)"""
    # Arguments for tail(n=5)
    try:
        if hasattr(test_obj, 'tail'):
            method = getattr(test_obj, 'tail')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method(5)
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ tail() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'tail' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ tail() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ tail() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ tail not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ tail() → Error: {str(e)}")
        return False, str(e)

def test_to_graph(test_obj, nodes_data, edges_data):
    """Test GraphTable.to_graph()"""
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

def test_validate(test_obj, nodes_data, edges_data):
    """Test GraphTable.validate()"""
    # No arguments needed
    try:
        if hasattr(test_obj, 'validate'):
            method = getattr(test_obj, 'validate')
            if callable(method):
                # TODO: Edit arguments as needed
                                result = method()
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ validate() → {type(result).__name__}: {result}")
                # For filter methods, also print alternative results if they exist
                if 'validate' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{i}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ validate() variant {i-1} → {type(alt_result).__name__}: {alt_result}")
                return True, result
            else:
                print(f"  ⚠️ validate() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ validate not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ validate() → Error: {str(e)}")
        return False, str(e)

def run_all_tests():
    """Run all GraphTable method tests"""
    print(f"# GraphTable Comprehensive Test Suite")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing 20 methods\n")
    
    # Create test objects
    test_obj, nodes_data, edges_data = create_test_objects()
    
    if test_obj is None:
        print("❌ Failed to create test object")
        return
    
    results = []
    working_count = 0
    total_count = 0
    
    print(f"## Testing GraphTable Methods\n")
    
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
    
    # Test edges
    success, result = test_edges(test_obj, nodes_data, edges_data)
    results.append({'method': 'edges', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test from_federated_bundles
    success, result = test_from_federated_bundles(test_obj, nodes_data, edges_data)
    results.append({'method': 'from_federated_bundles', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test head
    success, result = test_head(test_obj, nodes_data, edges_data)
    results.append({'method': 'head', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test load_bundle
    success, result = test_load_bundle(test_obj, nodes_data, edges_data)
    results.append({'method': 'load_bundle', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test merge
    success, result = test_merge(test_obj, nodes_data, edges_data)
    results.append({'method': 'merge', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test merge_with
    success, result = test_merge_with(test_obj, nodes_data, edges_data)
    results.append({'method': 'merge_with', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test merge_with_strategy
    success, result = test_merge_with_strategy(test_obj, nodes_data, edges_data)
    results.append({'method': 'merge_with_strategy', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test ncols
    success, result = test_ncols(test_obj, nodes_data, edges_data)
    results.append({'method': 'ncols', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test nodes
    success, result = test_nodes(test_obj, nodes_data, edges_data)
    results.append({'method': 'nodes', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test nrows
    success, result = test_nrows(test_obj, nodes_data, edges_data)
    results.append({'method': 'nrows', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test save_bundle
    success, result = test_save_bundle(test_obj, nodes_data, edges_data)
    results.append({'method': 'save_bundle', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test shape
    success, result = test_shape(test_obj, nodes_data, edges_data)
    results.append({'method': 'shape', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test stats
    success, result = test_stats(test_obj, nodes_data, edges_data)
    results.append({'method': 'stats', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test tail
    success, result = test_tail(test_obj, nodes_data, edges_data)
    results.append({'method': 'tail', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test to_graph
    success, result = test_to_graph(test_obj, nodes_data, edges_data)
    results.append({'method': 'to_graph', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    # Test validate
    success, result = test_validate(test_obj, nodes_data, edges_data)
    results.append({'method': 'validate', 'success': success, 'result': result})
    if success:
        working_count += 1
    total_count += 1
    
    
    # Print summary
    print(f"\n# GraphTable Test Summary")
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
