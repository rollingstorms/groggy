#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr
import inspect
from datetime import datetime

def get_test_args_for_method(obj, method_name, method_func):
    """Intelligently determine test arguments for a method based on its name and signature"""
    args = []
    kwargs = {}
    
    try:
        # Get method signature
        sig = inspect.signature(method_func)
        param_names = list(sig.parameters.keys())
        
        # Common argument patterns based on method names
        if method_name in ['has_node', 'degree', 'neighbors', 'in_degree', 'out_degree', 'contains_node', 'remove_node']:
            # Methods that need a node ID
            if hasattr(obj, 'node_count') and obj.node_count() > 0:
                args = [0]  # Use first node
        elif method_name in ['get_node_attr', 'set_node_attr', 'has_node_attribute']:
            # Methods that need node ID and attribute name
            if hasattr(obj, 'node_count') and obj.node_count() > 0:
                args = [0, "name"]  # Common attribute
        elif method_name in ['has_edge', 'get_edge_attr', 'set_edge_attr', 'contains_edge', 'remove_edge', 'edge_endpoints']:
            # Methods that need edge info
            if hasattr(obj, 'edge_count') and obj.edge_count() > 0:
                args = [0]  # Use first edge ID
        elif method_name in ['head', 'tail']:
            # Methods that take a count
            args = [3]
        elif method_name in ['sort_by']:
            # Methods that need column names
            if hasattr(obj, 'column_names'):
                try:
                    cols = obj.column_names()
                    if cols:
                        args = [cols[0]]
                except:
                    pass
        elif method_name in ['select']:
            # Methods that need column list
            if hasattr(obj, 'column_names'):
                try:
                    cols = obj.column_names()
                    if cols:
                        args = [cols[:1]]  # First column as list
                except:
                    pass
        elif method_name == '__getitem__':
            # Indexing operations
            if hasattr(obj, '__len__') and len(obj) > 0:
                if hasattr(obj, 'ncols'):  # Matrix-like
                    args = [0, 0]
                else:  # Array-like or table column access
                    try:
                        if hasattr(obj, 'column_names'):
                            cols = obj.column_names()
                            if cols:
                                args = [cols[0]]  # Column name
                        else:
                            args = [0]  # Index
                    except:
                        args = [0]
        elif method_name in ['subgraph']:
            # Methods that need node lists
            if hasattr(obj, 'node_ids'):
                try:
                    node_ids = obj.node_ids()
                    if node_ids:
                        args = [node_ids[:2]]  # First two nodes
                except:
                    pass
        elif method_name in ['add_node']:
            # Add operations
            kwargs = {"name": "test"}
        elif method_name in ['add_edge']:
            # Edge operations
            if hasattr(obj, 'node_count') and obj.node_count() >= 2:
                args = [0, 1]
        elif method_name in ['add_nodes']:
            # Bulk add operations
            args = [[{"name": "test1"}, {"name": "test2"}]]
        elif method_name in ['add_edges']:
            # Bulk edge operations
            if hasattr(obj, 'node_count') and obj.node_count() >= 2:
                args = [[(0, 1), (0, 2)]]
        elif method_name in ['filter_nodes', 'filter_edges']:
            # Filter operations - skip for now as they need complex filters
            return [], {}
        elif method_name in ['bfs', 'dfs', 'shortest_path']:
            # Graph traversal - need start nodes
            if hasattr(obj, 'node_count') and obj.node_count() > 0:
                if 'shortest_path' in method_name:
                    args = [0, 1] if obj.node_count() > 1 else [0, 0]
                else:
                    args = [0]
        elif method_name in ['get_node_attrs', 'set_node_attrs']:
            # Bulk attribute operations
            if hasattr(obj, 'node_count') and obj.node_count() > 0:
                if 'get' in method_name:
                    args = [[0], ["name"]]
                else:
                    args = [{"0": {"name": "test"}}]
        elif method_name in ['get_edge_attrs', 'set_edge_attrs']:
            # Bulk edge attribute operations
            if hasattr(obj, 'edge_count') and obj.edge_count() > 0:
                if 'get' in method_name:
                    args = [[0], ["weight"]]
                else:
                    args = [{"0": {"weight": 1.0}}]
        elif method_name in ['weighted_adjacency_matrix']:
            args = ["weight"]
        elif method_name in ['aggregate', 'group_by', 'group_nodes_by_attribute']:
            # Skip complex aggregation methods
            return [], {}
        
    except Exception:
        # If introspection fails, use no arguments
        pass
    
    return args, kwargs

def discover_delegated_methods(obj):
    """Discover methods that might be available through __getattr__ delegation"""
    delegated_methods = []
    
    # Common method names to test for delegation
    common_methods = [
        # BaseTable methods that might be delegated
        'column_names', 'columns', 'dtypes', 'schema', 'describe_columns',
        'filter_by', 'group_by_attr', 'aggregate_by', 'pivot',
        'join', 'union', 'intersect', 'difference',
        'sample', 'shuffle', 'sort_by_multiple', 'rename_columns',
        'add_column', 'remove_column', 'reorder_columns',
        'fill_null', 'drop_null', 'replace_values',
        'to_csv', 'to_json', 'to_parquet', 'from_csv', 'from_json',
        
        # Graph methods that might be delegated to subgraphs
        'centrality', 'pagerank', 'clustering_coefficient', 'betweenness',
        'closeness', 'eigenvector_centrality', 'katz_centrality',
        'triangles', 'transitivity', 'diameter', 'radius',
        'min_spanning_tree', 'max_flow', 'min_cut',
        'strongly_connected_components', 'weakly_connected_components',
        'topological_sort', 'is_dag', 'has_cycle',
        
        # Array methods that might be delegated
        'cumsum', 'cumprod', 'cummax', 'cummin', 'diff',
        'rolling_mean', 'rolling_sum', 'rolling_max', 'rolling_min',
        'interpolate', 'resample', 'normalize', 'standardize',
        'binning', 'discretize', 'clip', 'abs', 'log', 'exp', 'sqrt',
        
        # Matrix methods that might be delegated
        'transpose', 'inverse', 'determinant', 'eigenvalues', 'eigenvectors',
        'svd', 'qr', 'cholesky', 'lu', 'rank', 'trace', 'norm',
        'condition_number', 'spectral_radius', 'frobenius_norm'
    ]
    
    for method_name in common_methods:
        try:
            # Try to get the attribute - if __getattr__ delegation exists, this will succeed
            attr = getattr(obj, method_name, None)
            if attr is not None and method_name not in dir(obj):
                # This method exists but wasn't in dir() - likely delegated
                if callable(attr):
                    args, kwargs = get_test_args_for_method(obj, method_name, attr)
                    delegated_methods.append((method_name, args, kwargs, "delegated"))
                else:
                    delegated_methods.append((method_name, [], {}, "delegated_property"))
        except Exception:
            # Method doesn't exist or caused an error during access
            continue
    
    return delegated_methods

def test_object_methods(obj, object_name):
    """Test all methods of an object systematically, including delegated methods"""
    print(f"\n## {object_name} Methods\n")
    
    # Discover all methods automatically
    methods_to_test = []
    
    # Get all callable methods and properties from dir()
    direct_methods = []
    for name in dir(obj):
        if not name.startswith('_') or name in ['__str__', '__repr__', '__len__', '__getitem__', '__iter__']:
            try:
                attr = getattr(obj, name)
                if callable(attr):
                    # Try to determine appropriate test arguments
                    args, kwargs = get_test_args_for_method(obj, name, attr)
                    direct_methods.append((name, args, kwargs, "direct"))
                elif not name.startswith('_'):  # Properties
                    direct_methods.append((name, [], {}, "direct_property"))
            except Exception:
                # Skip methods we can't introspect
                continue
    
    # Discover methods available through __getattr__ delegation
    delegated_methods = discover_delegated_methods(obj)
    
    # Combine both sets of methods
    methods_to_test = direct_methods + delegated_methods
    
    # Sort methods by name for better organization
    methods_to_test.sort(key=lambda x: x[0])
    
    working_methods = []
    failing_methods = []
    
    print("| Method | Type | Status | Error |")
    print("|--------|------|--------|-------|")
    
    for method_data in methods_to_test:
        if len(method_data) == 4:
            method_name, args, kwargs, method_type = method_data
        else:
            # Fallback for old format
            method_name, args, kwargs = method_data
            method_type = "direct"
        
        # Create type indicator
        type_indicator = {
            "direct": "🔵",
            "direct_property": "🔵📄", 
            "delegated": "🟡", 
            "delegated_property": "🟡📄"
        }.get(method_type, "🔵")
        
        try:
            # Handle nested method calls like "nodes.table"
            if "." in method_name:
                parts = method_name.split(".")
                test_obj = obj
                for part in parts[:-1]:
                    test_obj = getattr(test_obj, part)
                method = getattr(test_obj, parts[-1])
            else:
                method = getattr(obj, method_name)
            
            if callable(method):
                result = method(*args, **kwargs)
            else:
                result = method
                
            working_methods.append(method_name)
            print(f"| `{method_name}` | {type_indicator} | ✅ Working | - |")
            
        except Exception as e:
            failing_methods.append((method_name, str(e)))
            error_short = str(e)[:60] + "..." if len(str(e)) > 60 else str(e)
            print(f"| `{method_name}` | {type_indicator} | ❌ Failing | {error_short} |")
    
    # Count methods by type
    direct_count = len([m for m in methods_to_test if len(m) > 3 and m[3].startswith("direct")])
    delegated_count = len([m for m in methods_to_test if len(m) > 3 and m[3].startswith("delegated")])
    
    print(f"\n**Summary**: {len(working_methods)} working, {len(failing_methods)} failing out of {len(methods_to_test)} total methods")
    print(f"- Direct methods: {direct_count}")
    print(f"- Delegated methods: {delegated_count}\n")
    
    return working_methods, failing_methods

def main():
    """Generate systematic method testing report"""
    print("# Groggy Systematic Method Testing Report")
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis report systematically tests every method of every major object type in Groggy using Python introspection.")
    print("It discovers both direct methods (from `dir()`) and delegated methods (available through `__getattr__`).\n")
    
    print("## Legend")
    print("- 🔵 Direct method (available in `dir()`)")  
    print("- 🔵📄 Direct property")
    print("- 🟡 Delegated method (available through `__getattr__`)")
    print("- 🟡📄 Delegated property")
    print("- ✅ Method works correctly")
    print("- ❌ Method fails with error\n")
    
    # Create test data
    g = gr.Graph()
    test_nodes = g.add_nodes([
        {'name': 'Alice', 'age': 25, 'salary': 75000, 'active': True, 'team': 'Engineering'},
        {'name': 'Bob', 'age': 30, 'salary': 85000, 'active': True, 'team': 'Sales'},
        {'name': 'Charlie', 'age': 35, 'salary': 95000, 'active': False, 'team': 'Marketing'},
        {'name': 'Diana', 'age': 28, 'salary': 80000, 'active': True, 'team': 'Engineering'},
    ])
    
    # Create edges with attributes
    test_edges = g.add_edges([
        (test_nodes[0], test_nodes[1], {'weight': 1.5, 'type': 'collaboration'}),
        (test_nodes[1], test_nodes[2], {'weight': 2.0, 'type': 'reports_to'}),
        (test_nodes[2], test_nodes[3], {'weight': 0.8, 'type': 'peer'}),
    ])
    
    all_working = []
    all_failing = []
    
    # Test Graph methods
    working, failing = test_object_methods(g, "Graph")
    all_working.extend([(m, "Graph") for m in working])
    all_failing.extend([(m, e, "Graph") for m, e in failing])
    
    # Test Table methods
    try:
        nodes_table = g.nodes.table()
        working, failing = test_object_methods(nodes_table, "NodesTable")
        all_working.extend([(m, "NodesTable") for m in working])
        all_failing.extend([(m, e, "NodesTable") for m, e in failing])
        
        edges_table = g.edges.table()
        working, failing = test_object_methods(edges_table, "EdgesTable")
        all_working.extend([(m, "EdgesTable") for m in working])
        all_failing.extend([(m, e, "EdgesTable") for m, e in failing])
    except Exception as e:
        print(f"## Table Testing Failed\n\nError: {e}\n")
    
    # Test Array methods
    try:
        nodes_table = g.nodes.table()
        if hasattr(nodes_table, '__getitem__'):
            node_ids_array = nodes_table["node_id"]
            working, failing = test_object_methods(node_ids_array, "GraphArray")
            all_working.extend([(m, "GraphArray") for m in working])
            all_failing.extend([(m, e, "GraphArray") for m, e in failing])
    except Exception as e:
        print(f"## Array Testing Failed\n\nError: {e}\n")
    
    # Test Matrix methods
    try:
        adj_matrix = g.adjacency_matrix()
        working, failing = test_object_methods(adj_matrix, "GraphMatrix")
        all_working.extend([(m, "GraphMatrix") for m in working])
        all_failing.extend([(m, e, "GraphMatrix") for m, e in failing])
    except Exception as e:
        print(f"## Matrix Testing Failed\n\nError: {e}\n")
    
    # Test Subgraph methods
    try:
        subgraph = g.subgraph(test_nodes[:2])
        working, failing = test_object_methods(subgraph, "Subgraph")
        all_working.extend([(m, "Subgraph") for m in working])
        all_failing.extend([(m, e, "Subgraph") for m, e in failing])
    except Exception as e:
        print(f"## Subgraph Testing Failed\n\nError: {e}\n")
    
    # Summary
    print("# Overall Summary")
    print(f"\n- **Total Working Methods**: {len(all_working)}")
    print(f"- **Total Failing Methods**: {len(all_failing)}")
    print(f"- **Total Methods Tested**: {len(all_working) + len(all_failing)}")
    print(f"- **Success Rate**: {len(all_working)/(len(all_working)+len(all_failing))*100:.1f}%")
    
    print("\n## Methods Needing Work")
    print("\nThese methods currently fail and need implementation or bug fixes:\n")
    
    for method, error, obj_type in all_failing:
        print(f"- **{obj_type}.{method}**: {error}")
    
    print(f"\n---\n*Generated by systematic method testing using Python introspection*")

if __name__ == "__main__":
    main()