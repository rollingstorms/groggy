#!/usr/bin/env python3
"""
Dynamic Repository Analyzer for Groggy

Treats the codebase as a graph:
- Nodes: Object types (Graph, NodesTable, etc.) 
- Edges: Methods (transformations between types)

Key features:
1. Dynamic object discovery (no hardcoding)
2. Intelligent argument injection from example objects
3. GraphTable export of API structure
4. Automatic test generation based on runtime analysis
5. Visual representation of repository structure
"""

import sys
import os
import inspect
import importlib
import traceback
import json
from typing import Any, Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime

# Add the python-groggy path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-groggy/python'))

try:
    import groggy
except ImportError as e:
    print(f"❌ Failed to import groggy: {e}")
    print("Make sure you're running from the groggy root directory")
    sys.exit(1)

@dataclass
class MethodEdge:
    """Represents a method as an edge in the repository graph"""
    source_type: str
    target_type: str
    method_name: str
    signature: str
    params: List[str]
    return_annotation: str
    success: bool = False
    error: Optional[str] = None
    execution_time: float = 0.0

@dataclass 
class ObjectNode:
    """Represents an object type as a node in the repository graph"""
    type_name: str
    module: str
    methods: List[str]
    properties: List[str]
    creation_success: bool = False
    creation_error: Optional[str] = None

class DynamicRepoAnalyzer:
    def __init__(self):
        self.discovered_types: Dict[str, type] = {}
        self.example_objects: Dict[str, Any] = {}
        self.method_graph: List[MethodEdge] = []
        self.object_nodes: List[ObjectNode] = []
        self.primitive_pool = self._create_primitive_pool()
        
    def _create_primitive_pool(self) -> Dict[str, Any]:
        """Create a pool of primitive values for method injection"""
        return {
            'int': [0, 1, 2, 5, 10, 100],
            'str': ['test', 'name', 'attr', 'Alice', 'Bob', 'id'],
            'float': [0.0, 1.0, 0.5, 3.14],
            'bool': [True, False],
            'list': [[], [1, 2, 3], ['a', 'b'], [0, 1]],
            'dict': [{}, {'key': 'value'}, {'attr': 'test'}],
        }
    
    def discover_groggy_types(self) -> Dict[str, type]:
        """Dynamically discover all Groggy object types"""
        print("🔍 Discovering Groggy object types...")
        
        discovered = {}
        
        # Scan the groggy module for classes
        for name in dir(groggy):
            obj = getattr(groggy, name)
            if inspect.isclass(obj) and not name.startswith('_'):
                # Check if it's a substantial class (has methods)
                methods = [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m, None))]
                if len(methods) > 2:  # More than just basic methods
                    discovered[name] = obj
                    print(f"  ✅ Found: {name} ({len(methods)} methods)")
        
        print(f"\n📊 Discovered {len(discovered)} object types")
        return discovered
    
    def create_example_objects(self) -> Dict[str, Any]:
        """Create example objects for each discovered type"""
        print("\n🏗️ Creating example objects...")
        
        examples = {}
        
        # Create a comprehensive graph with rich data for object creation
        try:
            # Rich sample data
            nodes_data = [
                ('Alice', {'age': 25, 'city': 'NYC', 'salary': 75000}),
                ('Bob', {'age': 30, 'city': 'LA', 'salary': 85000}), 
                ('Carol', {'age': 35, 'city': 'Chicago', 'salary': 95000}),
                ('Dave', {'age': 28, 'city': 'Austin', 'salary': 70000})
            ]
            
            edges_data = [
                ('Alice', 'Bob', {'relationship': 'friend', 'years': 5}),
                ('Bob', 'Carol', {'relationship': 'colleague', 'years': 2}),
                ('Carol', 'Dave', {'relationship': 'manager', 'years': 1}),
                ('Alice', 'Carol', {'relationship': 'friend', 'years': 3})
            ]
            
            # Create base graph - try different construction patterns
            g = None
            
            # Try pattern 1: Empty constructor then add nodes/edges
            try:
                g = groggy.Graph()
                # Check if add_node exists and what signature it has
                if hasattr(g, 'add_node'):
                    # Try different add_node patterns
                    try:
                        g.add_node(node_id=nodes_data[0][0], **nodes_data[0][1])
                    except:
                        try:
                            g.add_node(nodes_data[0][0], nodes_data[0][1])
                        except:
                            g.add_node(nodes_data[0][0])
                else:
                    # Maybe it needs data upfront
                    g = None
            except:
                g = None
            
            # Try pattern 2: Construction with data
            if g is None:
                try:
                    g = groggy.Graph(nodes=nodes_data, edges=edges_data)
                except:
                    try:
                        g = groggy.Graph(nodes=[n[0] for n in nodes_data], 
                                       edges=[(e[0], e[1]) for e in edges_data])
                    except:
                        g = groggy.Graph()
            
            if g is not None:
                examples['Graph'] = g
                try:
                    print(f"  ✅ Graph: {g.num_nodes()} nodes, {g.num_edges()} edges")
                except:
                    print(f"  ✅ Graph: created successfully")
            else:
                print(f"  ❌ Graph: could not create with any pattern")
            
        except Exception as e:
            print(f"  ❌ Graph creation failed: {e}")
        
        # Try to create other objects dynamically
        for type_name, type_class in self.discovered_types.items():
            if type_name == 'Graph':
                continue  # Already created
                
            try:
                # Try different creation strategies
                obj = None
                
                # Strategy 1: No args constructor
                try:
                    obj = type_class()
                    method = "no args"
                except:
                    pass
                
                # Strategy 2: From existing graph - be more aggressive
                if obj is None and 'Graph' in examples:
                    try:
                        graph = examples['Graph']
                        # Try multiple patterns for each type
                        creation_patterns = {
                            'NodesTable': ['nodes', 'nodes_table'],
                            'EdgesTable': ['edges', 'edges_table'], 
                            'GraphArray': ['adjacency', 'to_array'],
                            'GraphMatrix': ['adjacency_matrix', 'sparse_adjacency_matrix', 'dense_adjacency_matrix'],
                            'GraphTable': ['table', 'to_table'],
                            'Subgraph': ['view', 'subgraph'],
                            'NeighborhoodResult': [], # Will try neighborhood call later
                        }
                        
                        patterns = creation_patterns.get(type_name, [])
                        # Add generic patterns
                        patterns.extend([
                            type_name.lower(),
                            type_name.lower().replace('table', '_table'),
                            type_name.lower().replace('array', '_array'),
                            type_name.lower().replace('matrix', '_matrix')
                        ])
                        
                        for pattern in patterns:
                            if hasattr(graph, pattern):
                                try:
                                    attr = getattr(graph, pattern)
                                    if callable(attr):
                                        obj = attr()
                                        method = f"from graph.{pattern}()"
                                        break
                                    else:
                                        obj = attr
                                        method = f"from graph.{pattern}"
                                        break
                                except:
                                    continue
                        
                        # Special cases that need arguments - be more aggressive
                        if obj is None:
                            try:
                                if type_name == 'NeighborhoodResult':
                                    # Try multiple neighborhood patterns
                                    try:
                                        obj = graph.neighborhood([0], radius=1)
                                        method = "from graph.neighborhood([0], radius=1)"
                                    except:
                                        try:
                                            obj = graph.neighborhood([0])
                                            method = "from graph.neighborhood([0])"
                                        except:
                                            pass
                                            
                                elif type_name == 'Subgraph':
                                    # Try multiple subgraph creation methods
                                    subgraph_methods = [
                                        ('bfs', lambda: graph.bfs(0, max_depth=2)),
                                        ('dfs', lambda: graph.dfs(0, max_depth=2)),
                                        ('filter_nodes', lambda: graph.filter_nodes("age > 25")),
                                        ('filter_edges', lambda: graph.filter_edges("weight > 0.5")),
                                    ]
                                    
                                    for method_name, method_func in subgraph_methods:
                                        try:
                                            obj = method_func()
                                            method = f"from graph.{method_name}(...)"
                                            break
                                        except:
                                            continue
                                            
                                elif type_name in ['NodesTable', 'EdgesTable']:
                                    # Try accessor patterns
                                    accessor_methods = {
                                        'NodesTable': [
                                            lambda: graph.nodes(),
                                            lambda: getattr(graph, 'nodes_table', lambda: None)()
                                        ],
                                        'EdgesTable': [
                                            lambda: graph.edges(),
                                            lambda: getattr(graph, 'edges_table', lambda: None)()
                                        ]
                                    }
                                    
                                    for method_func in accessor_methods.get(type_name, []):
                                        try:
                                            result = method_func()
                                            if result is not None:
                                                obj = result
                                                method = f"from graph accessor"
                                                break
                                        except:
                                            continue
                                            
                            except Exception as e:
                                print(f"    Debug: {type_name} creation failed: {e}")
                                pass
                                
                    except:
                        pass
                
                if obj is not None:
                    examples[type_name] = obj
                    print(f"  ✅ {type_name}: created via {method}")
                else:
                    print(f"  ⚠️ {type_name}: could not create example object")
                    
            except Exception as e:
                print(f"  ❌ {type_name}: {e}")
        
        return examples
    
    def analyze_method(self, obj: Any, method_name: str, obj_type: str) -> MethodEdge:
        """Analyze a single method and attempt to execute it"""
        method = getattr(obj, method_name)
        
        # Skip methods that don't have inspectable signatures (built-ins)
        try:
            sig = inspect.signature(method)
        except ValueError:
            # Built-in method without signature - skip it
            return MethodEdge(
                source_type=obj_type,
                target_type="Unknown",
                method_name=method_name,
                signature="<built-in>",
                params=[],
                return_annotation="Any",
                success=False,
                error="Built-in method - no signature available"
            )
        
        # Create method edge metadata
        edge = MethodEdge(
            source_type=obj_type,
            target_type="Unknown",  # Will infer from return value
            method_name=method_name,
            signature=str(sig),
            params=list(sig.parameters.keys()),
            return_annotation=str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else "Any"
        )
        
        # Try to execute the method with smart argument injection
        try:
            start_time = datetime.now()
            
            args = self._inject_smart_arguments(sig, obj_type, method_name)
            result = method(*args)
            
            end_time = datetime.now()
            edge.execution_time = (end_time - start_time).total_seconds()
            
            # Infer target type from result
            if result is not None:
                result_type = type(result).__name__
                edge.target_type = result_type
            
            edge.success = True
            
        except Exception as e:
            edge.error = str(e)
            edge.success = False
        
        return edge
    
    def _inject_smart_arguments(self, sig: inspect.Signature, obj_type: str, method_name: str) -> List[Any]:
        """Intelligently inject arguments based on method signature and context"""
        args = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # Try to infer argument based on name and type hints
            arg_value = None
            
            # 1. Try annotation-based injection
            if param.annotation != inspect.Parameter.empty:
                annotation = param.annotation
                if hasattr(annotation, '__name__'):
                    type_name = annotation.__name__
                    if type_name in self.example_objects:
                        arg_value = self.example_objects[type_name]
                    elif type_name in self.primitive_pool:
                        arg_value = self.primitive_pool[type_name][0]
            
            # 2. Try name-based injection
            if arg_value is None:
                arg_value = self._infer_from_parameter_name(param_name, obj_type, method_name)
            
            # 3. Default fallback
            if arg_value is None:
                if param.default != inspect.Parameter.empty:
                    arg_value = param.default
                else:
                    arg_value = self._get_default_value_for_param(param_name)
            
            args.append(arg_value)
        
        return args
    
    def _infer_from_parameter_name(self, param_name: str, obj_type: str, method_name: str) -> Any:
        """Infer argument value from parameter name patterns"""
        name_lower = param_name.lower()
        
        # Node/edge ID patterns - Groggy uses integer IDs!
        if 'node' in name_lower and ('id' in name_lower or param_name == 'node'):
            # Try to get actual node IDs from the graph
            if 'Graph' in self.example_objects:
                try:
                    nodes_table = self.example_objects['Graph'].nodes()
                    if hasattr(nodes_table, 'node_ids'):
                        node_ids = list(nodes_table.node_ids())
                        if node_ids:
                            return node_ids[0]  # Return integer ID
                except:
                    pass
            return 0  # Default integer node ID
        
        if 'edge' in name_lower and 'id' in name_lower:
            if 'Graph' in self.example_objects:
                try:
                    edges_table = self.example_objects['Graph'].edges()
                    if hasattr(edges_table, 'edge_ids'):
                        edge_ids = list(edges_table.edge_ids())
                        if edge_ids:
                            return edge_ids[0]  # Return integer ID
                except:
                    pass
            return 0  # Default integer edge ID
        
        # Multiple nodes/edges
        if 'nodes' in name_lower:
            if 'Graph' in self.example_objects:
                try:
                    nodes_table = self.example_objects['Graph'].nodes()
                    if hasattr(nodes_table, 'node_ids'):
                        node_ids = list(nodes_table.node_ids())[:2]  # Get first 2
                        if node_ids:
                            return node_ids
                except:
                    pass
            return [0, 1]  # Default integer node IDs
        
        if 'edges' in name_lower:
            if 'Graph' in self.example_objects:
                try:
                    edges_table = self.example_objects['Graph'].edges()
                    if hasattr(edges_table, 'edge_ids'):
                        edge_ids = list(edges_table.edge_ids())[:2]  # Get first 2
                        if edge_ids:
                            return edge_ids
                except:
                    pass
            return [0, 1]  # Default integer edge IDs
        
        # Attribute patterns - get real attribute names
        if 'attr' in name_lower or 'key' in name_lower:
            if 'Graph' in self.example_objects:
                try:
                    if 'node' in obj_type.lower() or 'node' in method_name.lower():
                        attrs = self.example_objects['Graph'].all_node_attribute_names()
                        return attrs[0] if attrs else 'age'
                    elif 'edge' in obj_type.lower() or 'edge' in method_name.lower():
                        attrs = self.example_objects['Graph'].all_edge_attribute_names()
                        return attrs[0] if attrs else 'relationship'
                except:
                    pass
            return 'age' if 'node' in obj_type.lower() else 'relationship'
        
        # Attribute list patterns
        if 'attrs' in name_lower:
            if 'Graph' in self.example_objects:
                try:
                    if 'node' in obj_type.lower() or 'node' in method_name.lower():
                        attrs = self.example_objects['Graph'].all_node_attribute_names()
                        return list(attrs)[:2] if attrs else ['age']
                    elif 'edge' in obj_type.lower() or 'edge' in method_name.lower():
                        attrs = self.example_objects['Graph'].all_edge_attribute_names()
                        return list(attrs)[:2] if attrs else ['relationship']
                except:
                    pass
            return ['age'] if 'node' in obj_type.lower() else ['relationship']
        
        # Value patterns
        if 'value' in name_lower:
            return 'test_value'
        
        # Dictionary patterns for setting attributes
        if 'attrs_dict' in param_name or ('dict' in name_lower and 'attr' in name_lower):
            if 'node' in method_name.lower():
                return {0: {'test_attr': 'test_value'}}  # {node_id: {attr: value}}
            elif 'edge' in method_name.lower():
                return {0: {'test_attr': 'test_value'}}  # {edge_id: {attr: value}}
        
        # Filter patterns - use string predicates for Groggy
        if 'filter' in name_lower:
            if 'node' in obj_type.lower() or 'node' in method_name.lower():
                return "age > 25"  # String predicate for node filter
            elif 'edge' in obj_type.lower() or 'edge' in method_name.lower():
                return "relationship == 'friend'"  # String predicate for edge filter
            else:
                return "age > 25"  # Generic string predicate
        
        # Predicate patterns (for table operations)
        if 'predicate' in name_lower:
            return "age > 25"  # String predicate
        
        # Column selection patterns
        if 'columns' in name_lower:
            if obj_type in ['BaseTable', 'NodesTable', 'EdgesTable']:
                try:
                    if obj_type == 'BaseTable' and 'BaseTable' in self.example_objects:
                        cols = self.example_objects['BaseTable'].column_names()
                        return list(cols)[:2] if cols else ['col1', 'col2']
                    elif 'Graph' in self.example_objects:
                        if 'node' in method_name.lower():
                            attrs = self.example_objects['Graph'].all_node_attribute_names()
                            return list(attrs)[:2] if attrs else ['age']
                        else:
                            attrs = self.example_objects['Graph'].all_edge_attribute_names()
                            return list(attrs)[:2] if attrs else ['relationship']
                except:
                    pass
            return ['col1', 'col2']
        
        # Common patterns
        patterns = {
            'uid_key': 'name',
            'hops': 2,
            'limit': 10,
            'radius': 2,
            'max_nodes': 100,
            'max_depth': 3,
            'other': lambda: self.example_objects.get(obj_type),
            'message': 'test commit',
            'author': 'test_user',
            'branch_name': 'test_branch',
            'source': 0,  # Integer node ID
            'target': 1,  # Integer node ID
            'start': 0,   # Integer node ID
            'center_nodes': [0],  # List of integer node IDs
            'operation': 'sum',
            'attribute': 'age',
            'aggregation_attr': 'age',
            'weight_attribute': 'weight',
            'column': 'age',
            'n': 5,
            'ascending': True,
            'normalized': False,
            'directed': False,
            'include_attributes': True,
        }
        
        return patterns.get(param_name, None)
    
    def _get_default_value_for_param(self, param_name: str) -> Any:
        """Get sensible default value for a parameter"""
        if 'id' in param_name.lower():
            return 'test_id'
        elif 'name' in param_name.lower():
            return 'test_name'
        elif 'value' in param_name.lower():
            return 42
        else:
            return None
    
    def analyze_repository(self):
        """Main analysis method - discovers types, creates objects, analyzes methods"""
        print("🚀 Starting Dynamic Repository Analysis\n")
        
        # Step 1: Discover types
        self.discovered_types = self.discover_groggy_types()
        
        # Step 2: Create example objects
        self.example_objects = self.create_example_objects()
        
        # Step 3: Analyze all methods
        print(f"\n🔬 Analyzing methods across {len(self.example_objects)} object types...")
        
        total_methods = 0
        successful_methods = 0
        
        for obj_type, obj in self.example_objects.items():
            print(f"\n📋 Analyzing {obj_type}...")
            
            # Get all methods (excluding only private ones to be more comprehensive)
            # Only filter out obviously non-Groggy methods
            basic_python_methods = {'capitalize', 'casefold', 'center', 'count', 'encode', 
                                  'endswith', 'expandtabs', 'find', 'format', 'format_map',
                                  'index', 'isalnum', 'isalpha', 'isascii', 'isdecimal', 
                                  'isdigit', 'isidentifier', 'islower', 'isnumeric', 'isprintable',
                                  'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower',
                                  'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 
                                  'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split',
                                  'splitlines', 'startswith', 'strip', 'swapcase', 'title', 
                                  'translate', 'upper', 'zfill'}
            
            # Safely get methods, including magic methods
            try:
                # Get ALL methods, including magic methods like __len__, __str__, __repr__
                all_methods = []
                for name in dir(obj):
                    try:
                        attr = getattr(obj, name, None)
                        if callable(attr):
                            # Include magic methods that are commonly tested
                            magic_methods_to_include = {'__len__', '__str__', '__repr__', '__bool__', 
                                                      '__contains__', '__iter__', '__getitem__', 
                                                      '__setitem__', '__delitem__'}
                            
                            if not name.startswith('_') or name in magic_methods_to_include:
                                all_methods.append(name)
                    except (AttributeError, RuntimeError):
                        # Some attributes can't be accessed safely
                        continue
                        
            except (ValueError, AttributeError, RuntimeError) as e:
                print(f"  ⚠️ Could not introspect methods for {obj_type}: {e}")
                continue
            
            # Keep methods that are likely Groggy-specific (not basic Python string/int methods)
            methods = []
            for method_name in all_methods:
                # Skip only basic Python methods, keep everything else including __len__, __str__, etc. equivalents
                if method_name in basic_python_methods:
                    continue
                methods.append(method_name)
            
            print(f"  📊 Found {len(methods)} methods to test (filtered {len(all_methods) - len(methods)} basic Python methods)")
            
            # Analyze each method
            for method_name in methods:
                edge = self.analyze_method(obj, method_name, obj_type)
                self.method_graph.append(edge)
                
                if edge.success:
                    successful_methods += 1
                    status = "✅"
                else:
                    status = "❌"
                
                total_methods += 1
                print(f"  {status} {method_name}({', '.join(edge.params)})")
        
        print(f"\n📊 Analysis Complete:")
        print(f"   Total methods: {total_methods}")
        print(f"   Successful: {successful_methods} ({successful_methods/total_methods*100:.1f}%)")
        print(f"   Failed: {total_methods - successful_methods}")
        
    def export_results(self):
        """Export analysis results in multiple formats"""
        print("\n💾 Exporting results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Export as JSON
        results = {
            'timestamp': timestamp,
            'discovered_types': list(self.discovered_types.keys()),
            'example_objects': list(self.example_objects.keys()),
            'method_graph': [asdict(edge) for edge in self.method_graph],
            'summary': {
                'total_methods': len(self.method_graph),
                'successful_methods': sum(1 for e in self.method_graph if e.success),
                'object_types': len(self.example_objects)
            }
        }
        
        json_path = f"repo_analysis_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✅ JSON export: {json_path}")
        
        # 2. Export GraphTable representation (if possible)
        try:
            self._export_as_graphtable()
        except Exception as e:
            print(f"  ⚠️ GraphTable export failed: {e}")
        
        # 3. Export test results summary
        self._export_test_summary(timestamp)
        
        return results

    def _export_as_graphtable(self):
        """Export the repository structure as a GraphTable"""
        print("  🔄 Creating GraphTable representation...")
        
        if 'Graph' not in self.example_objects:
            raise Exception("No Graph object available for GraphTable creation")
        
        # Create a new graph to represent the repository structure
        repo_graph = groggy.Graph()
        
        # Add nodes for each object type
        node_id = 0
        type_to_id = {}
        
        for obj_type in self.example_objects.keys():
            # Count methods for this type
            type_methods = [e for e in self.method_graph if e.source_type == obj_type]
            successful_methods = sum(1 for m in type_methods if m.success)
            total_methods = len(type_methods)
            success_rate = (successful_methods / total_methods * 100) if total_methods > 0 else 0
            
            # Add node with metadata
            node_attrs = {
                'object_type': obj_type,
                'total_methods': total_methods,
                'successful_methods': successful_methods,
                'success_rate': success_rate,
                'node_type': 'object_type'
            }
            
            # Try to add node - handle different Graph API patterns
            try:
                repo_graph.add_node(**node_attrs)
                type_to_id[obj_type] = node_id
                node_id += 1
            except:
                # If that doesn't work, try other patterns
                try:
                    # Maybe it needs explicit ID
                    repo_graph.add_node(node_id, **node_attrs)
                    type_to_id[obj_type] = node_id
                    node_id += 1
                except:
                    print(f"    ⚠️ Could not add node for {obj_type}")
                    continue
        
        # Add edges for methods that transform between types
        edge_id = 0
        for method in self.method_graph:
            if method.success and method.target_type != "Unknown" and method.target_type != "NoneType":
                source_id = type_to_id.get(method.source_type)
                # Map target type to our discovered types
                target_type = method.target_type
                
                # Handle return type mapping
                type_mapping = {
                    'bool': 'bool',
                    'int': 'int', 
                    'str': 'str',
                    'list': 'list',
                    'dict': 'dict',
                    'NodesTable': 'NodesTable',
                    'EdgesTable': 'EdgesTable', 
                    'GraphMatrix': 'GraphMatrix',
                    'GraphArray': 'GraphArray',
                    'BaseTable': 'BaseTable'
                }
                
                mapped_target = type_mapping.get(target_type, target_type)
                target_id = type_to_id.get(mapped_target)
                
                if source_id is not None and target_id is not None and source_id != target_id:
                    edge_attrs = {
                        'method_name': method.method_name,
                        'signature': method.signature,
                        'execution_time': method.execution_time,
                        'transform_type': f"{method.source_type} -> {method.target_type}",
                        'edge_type': 'method_transform'
                    }
                    
                    try:
                        repo_graph.add_edge(source_id, target_id, **edge_attrs)
                    except:
                        try:
                            repo_graph.add_edge(source_id, target_id, edge_id, **edge_attrs)
                        except:
                            print(f"    ⚠️ Could not add edge for {method.method_name}")
                    
                    edge_id += 1
        
        # Export using save_bundle and other standard methods
        try:
            # Method 1: Use save_bundle (Groggy's native serialization)
            bundle_path = "repo_structure_bundle"
            try:
                repo_graph.save_bundle(bundle_path)
                print(f"  ✅ Bundle export: {bundle_path}/")
            except Exception as e:
                print(f"    ⚠️ Bundle save failed: {e}")
            
            # Method 2: Export as CSV files (nodes and edges separately)
            try:
                nodes_table = repo_graph.nodes()
                edges_table = repo_graph.edges()
                
                # Save nodes as CSV
                if hasattr(nodes_table, 'to_pandas'):
                    nodes_df = nodes_table.to_pandas()
                    nodes_df.to_csv("repo_nodes.csv", index=False)
                    print(f"  ✅ Nodes CSV: repo_nodes.csv")
                
                # Save edges as CSV  
                if hasattr(edges_table, 'to_pandas'):
                    edges_df = edges_table.to_pandas()
                    edges_df.to_csv("repo_edges.csv", index=False)
                    print(f"  ✅ Edges CSV: repo_edges.csv")
                    
            except Exception as e:
                print(f"    ⚠️ CSV export failed: {e}")
            
            # Method 3: Export as Parquet (more efficient for large graphs)
            try:
                nodes_table = repo_graph.nodes()
                edges_table = repo_graph.edges()
                
                if hasattr(nodes_table, 'to_pandas'):
                    nodes_df = nodes_table.to_pandas()
                    nodes_df.to_parquet("repo_nodes.parquet")
                    print(f"  ✅ Nodes Parquet: repo_nodes.parquet")
                
                if hasattr(edges_table, 'to_pandas'):
                    edges_df = edges_table.to_pandas()
                    edges_df.to_parquet("repo_edges.parquet")
                    print(f"  ✅ Edges Parquet: repo_edges.parquet")
                    
            except Exception as e:
                print(f"    ⚠️ Parquet export failed: {e}")
            
            # Method 4: Export to NetworkX format (widely compatible)
            try:
                nx_graph = repo_graph.to_networkx(include_attributes=True)
                
                # Save as GraphML (XML-based, preserves attributes)
                import networkx as nx
                nx.write_graphml(nx_graph, "repo_structure.graphml")
                print(f"  ✅ GraphML export: repo_structure.graphml")
                
                # Save as edge list (simple text format)
                nx.write_edgelist(nx_graph, "repo_structure.edgelist")
                print(f"  ✅ Edge list: repo_structure.edgelist")
                
            except Exception as e:
                print(f"    ⚠️ NetworkX export failed: {e}")
            
            # Method 5: Custom JSON export with full metadata
            try:
                # Extract detailed node and edge data
                nodes_data = []
                for obj_type, node_id in type_to_id.items():
                    type_methods = [e for e in self.method_graph if e.source_type == obj_type]
                    successful_methods = sum(1 for m in type_methods if m.success)
                    total_methods = len(type_methods)
                    
                    nodes_data.append({
                        'id': node_id,
                        'object_type': obj_type,
                        'total_methods': total_methods,
                        'successful_methods': successful_methods,
                        'success_rate': (successful_methods / total_methods * 100) if total_methods > 0 else 0,
                        'working_methods': [m.method_name for m in type_methods if m.success],
                        'failing_methods': [m.method_name for m in type_methods if not m.success]
                    })
                
                edges_data = []
                for method in self.method_graph:
                    if method.success and method.target_type not in ["Unknown", "NoneType"]:
                        source_id = type_to_id.get(method.source_type)
                        target_type = method.target_type
                        # Map basic types to themselves for visualization
                        if target_type in ['bool', 'int', 'str', 'list', 'dict']:
                            continue  # Skip primitive return types for graph visualization
                        
                        target_id = type_to_id.get(target_type)
                        if source_id is not None and target_id is not None and source_id != target_id:
                            edges_data.append({
                                'source': source_id,
                                'target': target_id,
                                'method_name': method.method_name,
                                'signature': method.signature,
                                'execution_time': method.execution_time,
                                'transform': f"{method.source_type} -> {method.target_type}"
                            })
                
                graph_export = {
                    'nodes': nodes_data,
                    'edges': edges_data,
                    'metadata': {
                        'total_object_types': len(type_to_id),
                        'total_methods_analyzed': len(self.method_graph),
                        'successful_methods': sum(1 for m in self.method_graph if m.success),
                        'method_transforms': len(edges_data),
                        'timestamp': datetime.now().isoformat()
                    }
                }
                
                with open("repo_graph_complete.json", 'w') as f:
                    json.dump(graph_export, f, indent=2)
                print(f"  ✅ Complete graph JSON: repo_graph_complete.json")
                
            except Exception as e:
                print(f"    ⚠️ Custom JSON export failed: {e}")
            
            # Always save the basic metadata as fallback
            with open("repo_graph_metadata.json", 'w') as f:
                json.dump({
                    'type_to_id_mapping': type_to_id,
                    'total_nodes': len(type_to_id),
                    'total_edges': edge_id,
                    'discovered_object_types': list(type_to_id.keys())
                }, f, indent=2)
            print(f"  ✅ Graph metadata: repo_graph_metadata.json")
                
        except Exception as e:
            print(f"    ❌ All export methods failed: {e}")
    
    def _export_test_summary(self, timestamp: str):
        """Export a human-readable test summary"""
        summary_path = f"test_summary_{timestamp}.md"
        
        with open(summary_path, 'w') as f:
            f.write(f"# Repository Analysis Summary\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            f.write("## Object Types\n\n")
            for obj_type in self.example_objects.keys():
                methods = [e for e in self.method_graph if e.source_type == obj_type]
                successful = sum(1 for m in methods if m.success)
                total = len(methods)
                f.write(f"- **{obj_type}**: {successful}/{total} methods working ({successful/total*100:.1f}%)\n")
            
            f.write("\n## Failed Methods\n\n")
            failed_methods = [e for e in self.method_graph if not e.success]
            for method in failed_methods:
                f.write(f"- `{method.source_type}.{method.method_name}()`: {method.error}\n")
        
        print(f"  ✅ Test summary: {summary_path}")

def main():
    analyzer = DynamicRepoAnalyzer()
    
    try:
        analyzer.analyze_repository()
        results = analyzer.export_results()
        
        print(f"\n🎉 Analysis complete! Check the exported files for detailed results.")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())