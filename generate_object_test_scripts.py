#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr
import inspect
import os
from datetime import datetime

def create_comprehensive_test_objects():
    """Create a comprehensive set of test objects covering all scenarios"""
    print("🏗️ Creating comprehensive test objects for script generation...")
    
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
    
    objects = {
        'Graph': g,
        'NodesTable': g.nodes.table(),
        'EdgesTable': g.edges.table(),
        'GraphArray': g.nodes.table()['node_id'],  # Use node_id array
        'GraphMatrix': g.adjacency_matrix(),
        'GraphTable': g.table(),
        'Subgraph': g.view(),
        'NeighborhoodResult': g.neighborhood([nodes_data[0]], radius=2),
        'BaseTable': g.nodes.table().base_table(),
    }
    
    return objects, nodes_data, edges_data

def get_object_methods(obj):
    """Get all public methods for an object"""
    methods = []
    for name in dir(obj):
        if not name.startswith('_') or name in ['__str__', '__repr__', '__len__', '__getitem__', '__iter__']:
            try:
                attr = getattr(obj, name)
                methods.append((name, attr))
            except:
                continue
    return methods

def get_smart_arguments(obj, method_name, method_func, test_nodes, test_edges):
    """Generate smart test arguments based on object type and method"""
    try:
        # Handle dunder methods first - these have specific patterns
        if method_name in ['__len__', '__str__', '__repr__', '__iter__']:
            return "# No arguments needed", []
        
        if method_name == '__getitem__':
            # Try to determine indexing type based on object
            obj_type = str(type(obj))
            if 'GraphTable' in obj_type or 'Subgraph' in obj_type:
                # These need column names or attribute names
                if hasattr(obj, 'column_names'):
                    try:
                        cols = obj.column_names()
                        if cols:
                            return f"# Arguments for {method_name}(key, /)", [f"'{cols[0]}'"]
                    except:
                        pass
                # For subgraph, try node attribute names
                if 'Subgraph' in obj_type and node_attrs:
                    return f"# Arguments for {method_name}(key, /)", [f"'{node_attrs[0]}'"]
                # Default to a common attribute name
                return f"# Arguments for {method_name}(key, /)", ["'name'"]
            elif hasattr(obj, 'column_names'):
                try:
                    cols = obj.column_names()
                    if cols:
                        return f"# Arguments for {method_name}(key, /)", [f"'{cols[0]}'"]
                except:
                    pass
            return f"# Arguments for {method_name}(key, /)", ["0"]
        
        sig = inspect.signature(method_func)
        params = list(sig.parameters.values())
        
        if len(params) == 0:
            return "# No arguments needed", []
        
        # Extract object data safely
        node_ids = []
        edge_ids = []
        node_attrs = []
        edge_attrs = []
        column_names = []
        
        try:
            if hasattr(obj, 'node_ids'):
                node_ids = list(obj.node_ids().to_list()[:3])
            elif hasattr(obj, 'all_node_attribute_names'):
                # For Graph objects
                try:
                    nodes_table = obj.nodes.table() 
                    node_ids = list(nodes_table.node_ids().to_list()[:3])
                except:
                    node_ids = test_nodes[:3] if test_nodes else [0, 1, 2]
        except:
            node_ids = test_nodes[:3] if test_nodes else [0, 1, 2]
        
        try:
            if hasattr(obj, 'edge_ids'):
                edge_ids = list(obj.edge_ids().to_list()[:3])
            elif hasattr(obj, 'all_edge_attribute_names'):
                try:
                    edges_table = obj.edges.table()
                    edge_ids = list(edges_table.edge_ids().to_list()[:3])
                except:
                    edge_ids = test_edges[:3] if test_edges else [0, 1, 2]
        except:
            edge_ids = test_edges[:3] if test_edges else [0, 1, 2]
        
        try:
            if hasattr(obj, 'all_node_attribute_names'):
                node_attrs = obj.all_node_attribute_names()[:3]
        except:
            node_attrs = ['name', 'age', 'team']
        
        try:
            if hasattr(obj, 'all_edge_attribute_names'):
                edge_attrs = obj.all_edge_attribute_names()[:3]
        except:
            edge_attrs = ['weight', 'type', 'strength']
        
        try:
            if hasattr(obj, 'column_names'):
                column_names = obj.column_names()[:3]
        except:
            column_names = []
        
        # Generate argument suggestions based on method patterns
        arg_comment = f"# Arguments for {method_name}{sig}"
        suggestions = []
        
        # Filter methods need special handling
        if method_name == 'filter_nodes':
            suggestions = [
                "gr.NodeFilter.attribute_equals('name', 'Alice')",
                "# Alternative: lambda n: n.get('age', 0) > 25"
            ]
            return arg_comment, suggestions
        
        elif method_name == 'filter_edges':
            suggestions = [
                "gr.EdgeFilter.attribute_equals('type', 'collaboration')",
                "# Alternative: lambda e: e.get('weight', 0) > 1.0"
            ]
            return arg_comment, suggestions
        
        elif method_name == 'filter':
            # Generic table/array filter
            if column_names:
                suggestions = [f"lambda x: x.get('{column_names[0]}', None) is not None"]
            else:
                suggestions = ["lambda x: x > 0"]
            return arg_comment, suggestions
        
        # Table/array methods that are commonly no-arg
        elif method_name in ['nrows', 'ncols', 'shape', 'column_names', 'copy', 'clear', 
                           'keys', 'values', 'items', 'popitem', 'rich_display']:
            suggestions = []  # no args
            return arg_comment, suggestions
        
        # Methods with optional n parameter
        elif method_name in ['head', 'tail']:
            suggestions = ["5"]  # default n
            return arg_comment, suggestions
        
        # Table selection methods
        elif method_name == 'select':
            if column_names:
                suggestions = [f"['{column_names[0]}']"]
            else:
                suggestions = ["['column_name']"]
            return arg_comment, suggestions
        
        elif method_name in ['has_column', 'drop_columns']:
            if column_names:
                if method_name == 'has_column':
                    suggestions = [f"'{column_names[0]}'"]
                else:  # drop_columns
                    suggestions = [f"['{column_names[0]}']"]
            else:
                suggestions = ["'column_name'"] if method_name == 'has_column' else ["['column_name']"]
            return arg_comment, suggestions
        
        # Dict-like methods
        elif method_name in ['get', 'pop', 'setdefault']:
            obj_type = str(type(obj))
            if method_name == 'get':
                if 'GraphMatrix' in obj_type:
                    suggestions = ["'size'"]  # GraphMatrix is a dict-like with known keys
                else:
                    suggestions = ["'key'"]
            elif method_name == 'pop':
                if 'GraphMatrix' in obj_type:
                    suggestions = ["'size'"]  # Pop a known key
                else:
                    suggestions = ["'key'"]
            else:  # setdefault
                suggestions = ["'new_key'", "'default_value'"]
            return arg_comment, suggestions
        
        elif method_name == 'update':
            suggestions = ["{'new_key': 'new_value'}"]
            return arg_comment, suggestions
        
        # Graph methods that need specific arguments
        elif method_name == 'group_nodes_by_attribute':
            if node_attrs:
                suggestions = [f"'{node_attrs[0]}'", "'salary'", "'mean'"]
            else:
                suggestions = ["'attribute_name'", "'aggregation_attr'", "'operation'"]
            return arg_comment, suggestions
        
        elif method_name in ['get_node_attrs', 'get_edge_attrs']:
            # These need node/edge IDs and lists of attribute names
            if method_name == 'get_node_attrs' and node_attrs:
                suggestions = ["nodes_data[0]", f"['{node_attrs[0]}', '{node_attrs[1] if len(node_attrs) > 1 else node_attrs[0]}']"]
            elif method_name == 'get_edge_attrs' and edge_attrs:
                suggestions = ["edges_data[0]", f"['{edge_attrs[0]}', '{edge_attrs[1] if len(edge_attrs) > 1 else edge_attrs[0]}']"]
            else:
                suggestions = ["id_value", "['attr1', 'attr2']"]
            return arg_comment, suggestions
        
        elif method_name in ['set_node_attrs', 'set_edge_attrs']:
            # These need proper dict format: {id: {attr: value}} or {"nodes": [...], "values": [...]}
            if method_name == 'set_node_attrs':
                suggestions = ["{nodes_data[0]: {'new_attr': 'value'}}"]
            else:  # set_edge_attrs
                suggestions = ["{edges_data[0]: {'new_attr': 'value'}}"]
            return arg_comment, suggestions
        
        elif method_name in ['remove_nodes', 'remove_edges']:
            # Use actual IDs that exist in the graph
            if method_name == 'remove_nodes':
                suggestions = ["[nodes_data[0]]"]
            else:  # remove_edges
                suggestions = ["[edges_data[0]]"]
            return arg_comment, suggestions
        
        elif method_name == 'add_graph':
            # Add graph needs just the graph object, not the mapping parameters
            suggestions = ["gr.complete_graph(3)"]
            return arg_comment, suggestions
        
        elif method_name == 'resolve_string_id_to_node':
            # This needs the string ID and the uid_key parameter
            suggestions = ["'Alice'", "'name'"]
            return arg_comment, suggestions
        
        elif method_name == 'get_node_mapping':
            suggestions = ["'Alice'"]
            return arg_comment, suggestions
        
        elif method_name == 'historical_view':
            suggestions = ["1"]  # commit_id
            return arg_comment, suggestions
        
        # Table filter methods need string predicates, not lambdas
        elif method_name == 'filter':
            obj_type = str(type(obj))
            if 'Table' in obj_type:
                if column_names:
                    # Use a column that's likely to have filterable values
                    if 'salary' in column_names:
                        suggestions = ["'salary > 80000'"]
                    elif 'age' in column_names:
                        suggestions = ["'age > 25'"]
                    elif 'node_id' in column_names or 'edge_id' in column_names:
                        suggestions = ["'node_id >= 0'" if 'node_id' in column_names else "'edge_id >= 0'"]
                    else:
                        suggestions = [f"'{column_names[0]} is not null'"]
                else:
                    suggestions = ["'column_name > 0'"]
            else:
                # For other objects, use lambda functions
                if column_names:
                    suggestions = [f"lambda x: x.get('{column_names[0]}', None) is not None"]
                else:
                    suggestions = ["lambda x: x > 0"]
            return arg_comment, suggestions
        
        elif method_name == 'filter_by_attr':
            if node_attrs:
                suggestions = [f"'{node_attrs[0]}'", "'Alice'"]
            elif edge_attrs:
                suggestions = [f"'{edge_attrs[0]}'", "'collaboration'"]
            elif column_names:
                suggestions = [f"'{column_names[0]}'", "'value'"]
            else:
                suggestions = ["'attr_name'", "'value'"]
            return arg_comment, suggestions
        
        elif method_name in ['filter_by_sources', 'filter_by_targets']:
            if node_ids:
                suggestions = [f"[{node_ids[0]}]"]
            else:
                suggestions = ["[0, 1]"]
            return arg_comment, suggestions
        
        elif method_name == 'unique_attr_values':
            if node_attrs:
                suggestions = [f"'{node_attrs[0]}'"]
            elif edge_attrs:
                suggestions = [f"'{edge_attrs[0]}'"]
            elif column_names:
                suggestions = [f"'{column_names[0]}'"]
            else:
                suggestions = ["'attr_name'"]
            return arg_comment, suggestions
        
        elif method_name == 'with_attributes':
            # This needs a mapping of attribute names to attributes
            if node_attrs:
                suggestions = [f"'{node_attrs[0]}'", f"['{node_attrs[0]}', '{node_attrs[1] if len(node_attrs) > 1 else node_attrs[0]}']"]
            else:
                suggestions = ["'attr_name'", "['attr1', 'attr2']"]
            return arg_comment, suggestions
        
        # Subgraph methods
        elif method_name in ['get_node_attribute', 'get_edge_attribute']:
            if method_name == 'get_node_attribute' and node_ids and node_attrs:
                suggestions = [f"{node_ids[0]}", f"'{node_attrs[0]}'"]
            elif method_name == 'get_edge_attribute' and edge_ids and edge_attrs:
                suggestions = [f"{edge_ids[0]}", f"'{edge_attrs[0]}'"]
            else:
                suggestions = ["0", "'attr_name'"]
            return arg_comment, suggestions
        
        elif method_name == 'induced_subgraph':
            if node_ids:
                suggestions = [f"[{node_ids[0]}, {node_ids[1] if len(node_ids) > 1 else node_ids[0]}]"]
            else:
                suggestions = ["[0, 1]"]
            return arg_comment, suggestions
        
        elif method_name == 'subgraph_from_edges':
            if edge_ids:
                suggestions = [f"[{edge_ids[0]}]"]
            else:
                suggestions = ["[0]"]
            return arg_comment, suggestions
        
        elif method_name in ['intersect_with', 'merge_with', 'subtract_from']:
            # These need another subgraph - create a simple one
            suggestions = ["g.view()"]  # Use the parent graph's view
            return arg_comment, suggestions
        
        elif method_name in ['collapse_to_node', 'collapse_to_node_with_defaults']:
            # These need node_name and agg_functions dict
            suggestions = ["'collapsed_node'", "{'salary': 'mean', 'age': 'max'}"]
            return arg_comment, suggestions
        
        elif method_name == 'calculate_similarity':
            # This needs another subgraph, not a string
            suggestions = ["g.view()"]
            return arg_comment, suggestions
        
        # Dict-like methods that need keys
        elif method_name == 'fromkeys':
            suggestions = ["['key1', 'key2']", "'default_value'"]
            return arg_comment, suggestions
        
        # GraphTable methods
        elif method_name in ['save_bundle', 'load_bundle']:
            suggestions = ["'test_bundle.json'"]
            return arg_comment, suggestions
        
        elif method_name == 'from_federated_bundles':
            suggestions = ["['bundle1.json', 'bundle2.json']"]
            return arg_comment, suggestions
        
        elif method_name in ['merge', 'merge_with', 'merge_with_strategy']:
            # Create another table for merging
            if method_name == 'merge_with_strategy':
                suggestions = ["test_obj", "'left'"]  # Use self for testing
            else:
                suggestions = ["test_obj"]  # Use self for testing
            return arg_comment, suggestions
        
        elif method_name in ['bfs', 'dfs']:
            if node_ids:
                suggestions = [f"{node_ids[0]}"]
            else:
                suggestions = ["# TODO: Need start node"]
        elif method_name in ['has_node', 'contains_node', 'node_attribute_keys']:
            if node_ids:
                suggestions = [f"{node_ids[0]}"]
            else:
                suggestions = ["# TODO: Need node_id"]
        elif method_name in ['get_node_attribute', 'get_node_attr']:
            if node_ids and node_attrs:
                suggestions = [f"{node_ids[0]}", f"'{node_attrs[0]}'"]
            else:
                suggestions = ["# TODO: Need node_id, attr_name"]
        elif method_name in ['has_node_attribute']:
            if node_ids and node_attrs:
                suggestions = [f"{node_ids[0]}", f"'{node_attrs[0]}'"]
            else:
                suggestions = ["# TODO: Need node_id, attr_name"]
        elif method_name in ['has_edge', 'contains_edge', 'edge_attribute_keys', 'edge_endpoints']:
            if edge_ids:
                suggestions = [f"{edge_ids[0]}"]
            else:
                suggestions = ["# TODO: Need edge_id"]
        elif method_name in ['get_edge_attribute', 'get_edge_attr']:
            if edge_ids and edge_attrs:
                suggestions = [f"{edge_ids[0]}", f"'{edge_attrs[0]}'"]
            else:
                suggestions = ["# TODO: Need edge_id, attr_name"]
        elif method_name in ['has_edge_attribute']:
            if edge_ids and edge_attrs:
                suggestions = [f"{edge_ids[0]}", f"'{edge_attrs[0]}'"]
            else:
                suggestions = ["# TODO: Need edge_id, attr_name"]
        elif method_name in ['neighbors']:
            if node_ids:
                suggestions = [f"{node_ids[0]}"]
            else:
                suggestions = ["# TODO: Need node_id"]
        elif method_name in ['shortest_path', 'shortest_path_subgraph']:
            if len(node_ids) >= 2:
                suggestions = [f"{node_ids[0]}", f"{node_ids[1]}"]
            else:
                suggestions = ["# TODO: Need source, target nodes"]
        elif method_name in ['has_edge_between']:
            if len(node_ids) >= 2:
                suggestions = [f"{node_ids[0]}", f"{node_ids[1]}"]
            else:
                suggestions = ["# TODO: Need source, target nodes"]
        elif method_name in ['has_path']:
            if len(node_ids) >= 2:
                suggestions = [f"{node_ids[0]}", f"{node_ids[1]}"]
            else:
                suggestions = ["# TODO: Need node1_id, node2_id"]
        elif method_name == 'add_node':
            suggestions = ["name='test_node'", "age=25"]
        elif method_name == 'add_nodes':
            suggestions = ["[{'name': 'test1'}, {'name': 'test2'}]"]
        elif method_name == 'add_edge':
            if len(node_ids) >= 2:
                suggestions = [f"{node_ids[0]}", f"{node_ids[1]}"]
            else:
                suggestions = ["# TODO: Need source, target nodes"]
        elif method_name == 'add_edges':
            if len(node_ids) >= 2:
                suggestions = [f"[({node_ids[0]}, {node_ids[1]})]"]
            else:
                suggestions = ["# TODO: Need edges list"]
        elif method_name in ['remove_node']:
            if node_ids:
                suggestions = [f"{node_ids[0]}"]
            else:
                suggestions = ["# TODO: Need node_id"]
        elif method_name in ['remove_edge']:
            if edge_ids:
                suggestions = [f"{edge_ids[0]}"]
            else:
                suggestions = ["# TODO: Need edge_id"]
        elif method_name in ['set_node_attr', 'set_node_attribute']:
            # Use nodes_data indices which are the actual node IDs that exist
            suggestions = ["nodes_data[0]", "'new_attr'", "'new_value'"]
            return arg_comment, suggestions
        elif method_name in ['set_edge_attr', 'set_edge_attribute']:
            # Use edges_data indices which are the actual edge IDs that exist 
            suggestions = ["edges_data[0]", "'new_attr'", "'new_value'"]
            return arg_comment, suggestions
        elif method_name == 'aggregate':
            if node_attrs:
                suggestions = [f"'{node_attrs[0]}'", "'mean'"]
            else:
                suggestions = ["# TODO: Need attribute, operation"]
        elif method_name in ['group_by']:
            if column_names:
                suggestions = [f"['{column_names[0]}']"]
            else:
                suggestions = ["# TODO: Need columns list"]
        elif method_name == 'sort_by':
            if column_names:
                suggestions = [f"'{column_names[0]}'"]
            else:
                suggestions = ["# TODO: Need column name"]
        elif method_name == 'select':
            if column_names:
                suggestions = [f"['{column_names[0]}']"]
            else:
                suggestions = ["# TODO: Need columns list"]
        elif method_name in ['slice']:
            suggestions = ["0", "5"]  # start, end
        elif method_name in ['quantile', 'percentile']:
            suggestions = ["0.5"]  # median
        elif method_name == 'fill_na':
            suggestions = ["0"]
        elif method_name in ['neighborhood']:
            obj_type = str(type(obj))
            if 'Subgraph' in obj_type:
                # Subgraph.neighborhood needs center_nodes and hops
                if node_ids:
                    suggestions = [f"[{node_ids[0]}]", "2"]  # hops parameter
                else:
                    suggestions = ["[0]", "2"]
            else:
                # Graph.neighborhood has radius parameter
                if node_ids:
                    suggestions = [f"[{node_ids[0]}]", "radius=2"]
                else:
                    suggestions = ["[0]", "radius=2"]
            return arg_comment, suggestions
        elif method_name == 'commit':
            suggestions = ["'Test commit'", "'test@example.com'"]
        elif method_name in ['checkout_branch', 'create_branch']:
            suggestions = ["'test_branch'"]
        elif method_name == 'weighted_adjacency_matrix':
            if edge_attrs:
                suggestions = [f"'{edge_attrs[0]}'"]
            else:
                suggestions = ["'weight'"]
        elif method_name in ['get_cell']:
            suggestions = ["0", "0"]  # row, col
        elif method_name in ['get_row', 'get_column']:
            suggestions = ["0"]  # row/col index
        elif method_name in ['power']:
            suggestions = ["2"]  # square
        elif method_name in ['identity', 'zeros']:
            suggestions = ["3"] if method_name == 'identity' else ["3", "3"]
        else:
            # Generic parameter inference
            for param in params:
                if param.default != inspect.Parameter.empty:
                    continue
                elif param.annotation in [int, 'int']:
                    suggestions.append("1")
                elif param.annotation in [float, 'float']:
                    suggestions.append("1.0")
                elif param.annotation in [str, 'str']:
                    suggestions.append("'test'")
                elif param.annotation in [bool, 'bool']:
                    suggestions.append("True")
                else:
                    suggestions.append("# TODO: Fill argument")
        
        return arg_comment, suggestions
        
    except Exception as e:
        return f"# Error getting signature: {e}", []

def generate_test_script(obj_name, obj, methods, test_nodes, test_edges):
    """Generate a complete test script for an object"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Comprehensive test script for Groggy {obj_name}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This script tests ALL methods of the {obj_name} class with proper argument patterns.
Edit the TODO sections to provide correct arguments for each method.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr
import traceback
from datetime import datetime

def create_test_objects():
    """Create test objects for {obj_name} testing"""
    print("🏗️ Creating test objects...")
    
    # Core graph with rich data
    g = gr.Graph()
    nodes_data = g.add_nodes([
        {{'name': 'Alice', 'age': 25, 'salary': 75000, 'active': True, 'team': 'Engineering', 'level': 'Senior'}},
        {{'name': 'Bob', 'age': 30, 'salary': 85000, 'active': True, 'team': 'Sales', 'level': 'Manager'}},
        {{'name': 'Charlie', 'age': 35, 'salary': 95000, 'active': False, 'team': 'Marketing', 'level': 'Director'}},
        {{'name': 'Diana', 'age': 28, 'salary': 80000, 'active': True, 'team': 'Engineering', 'level': 'Senior'}},
        {{'name': 'Eve', 'age': 32, 'salary': 90000, 'active': True, 'team': 'Product', 'level': 'Manager'}},
    ])
    
    edges_data = g.add_edges([
        (nodes_data[0], nodes_data[1], {{'weight': 1.5, 'type': 'collaboration', 'strength': 'strong'}}),
        (nodes_data[1], nodes_data[2], {{'weight': 2.0, 'type': 'reports_to', 'strength': 'formal'}}),
        (nodes_data[2], nodes_data[3], {{'weight': 0.8, 'type': 'peer', 'strength': 'weak'}}),
        (nodes_data[0], nodes_data[3], {{'weight': 1.2, 'type': 'collaboration', 'strength': 'medium'}}),
        (nodes_data[1], nodes_data[4], {{'weight': 1.8, 'type': 'cross_team', 'strength': 'strong'}}),
    ])
    
    # Create the specific test object for {obj_name}
'''
    
    # Add object-specific creation code
    if obj_name == 'Graph':
        script_content += "    test_obj = g\n"
    elif obj_name == 'NodesTable':
        script_content += "    test_obj = g.nodes.table()\n"
    elif obj_name == 'EdgesTable':
        script_content += "    test_obj = g.edges.table()\n"
    elif obj_name == 'GraphArray':
        script_content += "    test_obj = g.nodes.table()['node_id']\n"
    elif obj_name == 'GraphMatrix':
        script_content += "    test_obj = g.adjacency_matrix()\n"
    elif obj_name == 'GraphTable':
        script_content += "    test_obj = g.table()\n"
    elif obj_name == 'Subgraph':
        script_content += "    test_obj = g.view()\n"
    elif obj_name == 'NeighborhoodResult':
        script_content += "    test_obj = g.neighborhood([nodes_data[0]], radius=2)\n"
    elif obj_name == 'BaseTable':
        script_content += "    test_obj = g.nodes.table().base_table()\n"
    else:
        script_content += f"    # TODO: Create {obj_name} object\n    test_obj = None\n"
    
    script_content += f'''    
    return test_obj, nodes_data, edges_data

def test_method(obj, method_name, method_func, nodes_data, edges_data):
    """Test a single method with error handling"""
    print(f"Testing {{method_name}}...")
    
    try:
        # Call the method - EDIT THE ARGUMENTS AS NEEDED
        if method_name == 'PLACEHOLDER_METHOD':
            # Example: result = method_func(arg1, arg2, kwarg1=value)
            result = method_func()
        else:
            # Default call with no arguments
            result = method_func()
        
        print(f"  ✅ {{method_name}}() → {{type(result).__name__}}: {{result}}")
        return True, result
        
    except Exception as e:
        print(f"  ❌ {{method_name}}() → Error: {{str(e)}}")
        return False, str(e)

'''
    
    # Add individual test methods
    for method_name, method_func in methods:
        try:
            sig = inspect.signature(method_func) if callable(method_func) else "(property)"
        except:
            sig = "(...)"
        
        arg_comment, suggestions = get_smart_arguments(obj, method_name, method_func, test_nodes, test_edges)
        args_str = ", ".join(suggestions) if suggestions else ""
        
        # Generate method calls - handle multiple entry points for certain methods
        method_calls = []
        if method_name in ['filter_nodes', 'filter_edges']:
            # Test multiple entry points for filter methods
            for i, suggestion in enumerate(suggestions[:2]):  # Test up to 2 variants
                if suggestion.startswith('#'):
                    continue
                call_name = f"result{i+1}" if i > 0 else "result"
                method_calls.append(f"                {call_name} = method({suggestion})")
            if not method_calls:
                method_calls.append("                pass  # TODO: Fix arguments and uncomment")
        else:
            # Clean up arguments string to avoid syntax errors
            if args_str and any(x in args_str for x in ['# TODO', '# Need']):
                # If arguments contain TODOs, comment them out and provide pass
                args_str = f"# {args_str}"
                method_calls.append(f"                # result = method({args_str})\n                pass  # TODO: Fix arguments and uncomment")
            else:
                method_calls.append(f"                result = method({args_str})" if args_str else "                result = method()")
        
        method_call = "\n".join(method_calls)
        
        script_content += f'''
def test_{method_name}(test_obj, nodes_data, edges_data):
    """Test {obj_name}.{method_name}{sig}"""
    {arg_comment}
    try:
        if hasattr(test_obj, '{method_name}'):
            method = getattr(test_obj, '{method_name}')
            if callable(method):
                # TODO: Edit arguments as needed
                {method_call}
            else:
                # Property access
                result = method
            
            if 'result' in locals():
                print(f"  ✅ {method_name}() → {{type(result).__name__}}: {{result}}")
                # For filter methods, also print alternative results if they exist
                if '{method_name}' in ['filter_nodes', 'filter_edges']:
                    for i in range(2, 5):  # Check result2, result3, result4
                        alt_var = f'result{{i}}'
                        if alt_var in locals():
                            alt_result = locals()[alt_var]
                            print(f"  ✅ {method_name}() variant {{i-1}} → {{type(alt_result).__name__}}: {{alt_result}}")
                return True, result
            else:
                print(f"  ⚠️ {method_name}() → Skipped (needs argument fixes)")
                return False, "Skipped - needs argument fixes"
        else:
            print(f"  ⚠️ {method_name} not found on object")
            return False, "Method not found"
            
    except Exception as e:
        print(f"  ❌ {method_name}() → Error: {{str(e)}}")
        return False, str(e)
'''

    # Add main execution
    script_content += f'''
def run_all_tests():
    """Run all {obj_name} method tests"""
    print(f"# {obj_name} Comprehensive Test Suite")
    print(f"Generated: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"Testing {len(methods)} methods\\n")
    
    # Create test objects
    test_obj, nodes_data, edges_data = create_test_objects()
    
    if test_obj is None:
        print("❌ Failed to create test object")
        return
    
    results = []
    working_count = 0
    total_count = 0
    
    print(f"## Testing {obj_name} Methods\\n")
    
    # Run all method tests
'''
    
    for method_name, _ in methods:
        script_content += f'''    # Test {method_name}
    success, result = test_{method_name}(test_obj, nodes_data, edges_data)
    results.append({{'method': '{method_name}', 'success': success, 'result': result}})
    if success:
        working_count += 1
    total_count += 1
    
'''
    
    script_content += f'''    
    # Print summary
    print(f"\\n# {obj_name} Test Summary")
    print(f"**Results**: {{working_count}}/{{total_count}} methods working ({{working_count/total_count*100:.1f}}%)")
    
    # Show working methods
    working = [r for r in results if r['success']]
    failing = [r for r in results if not r['success']]
    
    print(f"\\n**Working Methods ({{len(working)}}):**")
    for r in working:  # Show all
        print(f"  ✅ {{r['method']}}")
    
    print(f"\\n**Failing Methods ({{len(failing)}}):**")  
    for r in failing:  # Show all
        print(f"  ❌ {{r['method']}}: {{r['result']}}")
    
    return results

if __name__ == "__main__":
    results = run_all_tests()
'''
    
    return script_content

def main():
    """Generate test scripts for all Groggy object types"""
    print("🚀 Generating comprehensive test scripts for all Groggy objects...")
    
    # Create test objects
    test_objects, test_nodes, test_edges = create_comprehensive_test_objects()
    
    # Create tests directory in documentation
    test_dir = "documentation/testing/generated_tests"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"📁 Created directory: {test_dir}")
    
    generated_files = []
    
    # Generate test script for each object type
    for obj_name, obj in test_objects.items():
        print(f"\\n📝 Generating test script for {obj_name}...")
        
        # Get all methods for this object
        methods = get_object_methods(obj)
        print(f"   Found {len(methods)} methods")
        
        # Generate the test script
        script_content = generate_test_script(obj_name, obj, methods, test_nodes, test_edges)
        
        # Write to file
        filename = f"{test_dir}/test_{obj_name.lower()}.py"
        with open(filename, 'w') as f:
            f.write(script_content)
        
        print(f"   ✅ Generated: {filename}")
        generated_files.append(filename)
    
    # Generate master test runner
    print(f"\\n📝 Generating master test runner...")
    
    runner_content = f'''#!/usr/bin/env python3
"""
Master test runner for all generated Groggy object tests
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This script runs all individual object test scripts and aggregates results.
"""

import os
import subprocess
import sys
from datetime import datetime

def run_test_script(script_path):
    """Run a single test script and capture output"""
    print(f"\\n🧪 Running {{os.path.basename(script_path)}}...")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"  ✅ Completed successfully")
            return True, result.stdout
        else:
            print(f"  ❌ Failed with return code {{result.returncode}}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout after 5 minutes")
        return False, "Timeout"
    except Exception as e:
        print(f"  💥 Error running script: {{e}}")
        return False, str(e)

def main():
    """Run all generated test scripts"""
    print("🚀 Master Test Runner for Groggy Object Tests")
    print(f"Started: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}\\n")
    
    test_scripts = [
'''
    
    for filename in generated_files:
        runner_content += f'        "{filename}",\n'
    
    runner_content += f'''    ]
    
    results = []
    
    for script in test_scripts:
        if os.path.exists(script):
            success, output = run_test_script(script)
            results.append({{'script': script, 'success': success, 'output': output}})
        else:
            print(f"⚠️ Script not found: {{script}}")
            results.append({{'script': script, 'success': False, 'output': 'File not found'}})
    
    # Generate summary
    print(f"\\n# Master Test Summary")
    print(f"Completed: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\\n**Overall Results**: {{len(successful)}}/{{len(results)}} test scripts completed successfully")
    
    if successful:
        print(f"\\n**Successful Scripts:**")
        for r in successful:
            print(f"  ✅ {{os.path.basename(r['script'])}}")
    
    if failed:
        print(f"\\n**Failed Scripts:**")
        for r in failed:
            print(f"  ❌ {{os.path.basename(r['script'])}}: {{r['output'][:100]}}...")
    
    # Save detailed results
    with open('master_test_results.txt', 'w') as f:
        f.write(f"Master Test Results - {{datetime.now()}}\\n")
        f.write("=" * 50 + "\\n\\n")
        
        for r in results:
            f.write(f"Script: {{r['script']}}\\n")
            f.write(f"Success: {{r['success']}}\\n")
            f.write(f"Output:\\n{{r['output']}}\\n")
            f.write("-" * 30 + "\\n\\n")
    
    print(f"\\n📄 Detailed results saved to: master_test_results.txt")
    return results

if __name__ == "__main__":
    results = main()
'''
    
    runner_filename = f"{test_dir}/run_all_tests.py"
    with open(runner_filename, 'w') as f:
        f.write(runner_content)
    
    print(f"   ✅ Generated: {runner_filename}")
    
    # Make scripts executable
    for filename in generated_files + [runner_filename]:
        os.chmod(filename, 0o755)
    
    print(f"\\n🎉 Successfully generated {len(generated_files)} test scripts + 1 master runner")
    print(f"\\n📋 Generated files:")
    for filename in sorted(generated_files + [runner_filename]):
        print(f"  - {filename}")
    
    print(f"\\n🚀 To run all tests: python {runner_filename}")
    print(f"🔧 To run individual tests: python {test_dir}/test_<object>.py")
    print(f"\\n💡 Next steps:")
    print(f"  1. Edit the TODO sections in each test file with proper arguments")
    print(f"  2. Run individual tests to verify they work")
    print(f"  3. Run the master test runner to execute all tests")
    print(f"  4. Use results to create comprehensive API documentation")

if __name__ == "__main__":
    main()