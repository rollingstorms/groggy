#!/usr/bin/env python3

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')
import groggy as gr
import inspect
from datetime import datetime
from collections import deque

def get_method_signature(obj, method_name):
    """Get method signature information"""
    try:
        method = getattr(obj, method_name)
        if callable(method):
            try:
                sig = inspect.signature(method)
                return str(sig)
            except (ValueError, TypeError):
                # Some methods don't have inspectable signatures (builtin methods)
                return "(...)"
        else:
            # It's a property
            return "property"
    except Exception:
        return "unknown"

def get_object_id(obj):
    """Get a unique identifier for an object for memoization"""
    try:
        return f"{type(obj).__name__}_{id(obj)}"
    except:
        return f"unknown_{id(obj)}"

def safe_call_method(obj, method_name, method_func):
    """Safely call a method and return its result, handling various argument patterns"""
    try:
        # Get method signature to understand arguments
        try:
            sig = inspect.signature(method_func)
            params = list(sig.parameters.keys())
        except:
            params = []
        
        # Try calling with no arguments first
        if len(params) == 0 or all(p.default != inspect.Parameter.empty for p in sig.parameters.values()):
            return method_func()
        
        # Handle common method patterns that need arguments
        if method_name == 'subgraph' and hasattr(obj, 'node_ids'):
            node_ids = obj.node_ids()
            if len(node_ids) >= 2:
                return method_func(node_ids[:2])
        elif method_name == 'neighborhood' and hasattr(obj, 'node_ids'):
            node_ids = obj.node_ids()
            if len(node_ids) >= 1:
                return method_func([node_ids[0]], radius=1)
        elif method_name in ['__getitem__', 'get'] and hasattr(obj, '__len__'):
            if len(obj) > 0:
                return method_func(0)
        elif method_name in ['head', 'tail']:
            return method_func(3)
        elif method_name == 'sort_by' and hasattr(obj, 'column_names'):
            cols = obj.column_names()
            if cols:
                return method_func(cols[0])
        elif method_name == 'select' and hasattr(obj, 'column_names'):
            cols = obj.column_names()
            if cols:
                return method_func([cols[0]])
        elif method_name in ['filter_nodes', 'filter_edges'] and hasattr(gr, 'NodeFilter'):
            return None  # Skip filter methods for now
        
        # If we can't figure out the arguments, skip this method
        return None
        
    except Exception:
        return None

def is_groggy_object(obj):
    """Check if an object is a Groggy-specific type (not pandas/numpy/stdlib)"""
    obj_type = type(obj).__name__
    module_name = getattr(type(obj), '__module__', '')
    
    # Groggy objects should have modules starting with these patterns
    groggy_modules = [
        'builtins',  # Core Groggy objects show up as builtins
        'groggy',
        'python_groggy', 
        '_groggy',
    ]
    
    # Check if it's a known Groggy type
    groggy_types = [
        'Graph', 'Subgraph', 'NeighborhoodResult', 
        'NodesTable', 'EdgesTable', 'GraphTable', 'BaseTable',
        'GraphArray', 'GraphMatrix', 
        'NodeFilter', 'EdgeFilter', 'AttributeFilter',
        'GraphView'
    ]
    
    if obj_type in groggy_types:
        return True
    
    # Check module
    if any(groggy_mod in module_name for groggy_mod in groggy_modules):
        return True
    
    # Special case: dict objects that are actually GraphMatrix
    if obj_type == 'dict' and hasattr(obj, 'keys') and len(obj) > 0:
        # Check if keys look like matrix indices
        try:
            keys = list(obj.keys())
            if all(isinstance(k, tuple) and len(k) == 2 for k in keys[:5]):
                return True  # Looks like a GraphMatrix
        except:
            pass
    
    return False

def discover_all_available_methods_bfs(starting_objects, max_depth=2):
    """BFS discovery of all methods and objects reachable from starting objects (Groggy objects only)"""
    print("🔍 **Starting BFS Method Discovery (Groggy objects only)...**\n")
    
    # Memoization structures
    discovered_objects = {}  # object_id -> (obj, name, methods, depth)
    method_results_cache = {}  # (object_id, method_name) -> result_object
    type_method_cache = {}  # type_name -> set of method signatures (to avoid duplicates)
    queue = deque()
    
    # Initialize queue with starting objects
    for obj, name in starting_objects:
        obj_id = get_object_id(obj)
        queue.append((obj, name, 0))  # (object, name, depth)
        print(f"🌱 Starting with {name}: {type(obj).__name__}")
    
    while queue:
        current_obj, current_name, depth = queue.popleft()
        
        if depth > max_depth:
            continue
            
        obj_id = get_object_id(current_obj)
        
        # Skip if already processed
        if obj_id in discovered_objects:
            continue
        
        # Skip non-Groggy objects at depth > 0
        if depth > 0 and not is_groggy_object(current_obj):
            continue
            
        obj_type = type(current_obj).__name__
        print(f"{'  ' * depth}🔄 Processing {current_name} (depth {depth}): {obj_type}")
        
        # Check if we've already seen this type
        if obj_type in type_method_cache and depth > 0:
            print(f"{'  ' * depth}⚡ Skipping {obj_type} - already analyzed this type")
            continue
        
        # Discover methods for this object
        methods = discover_all_available_methods(current_obj, current_name, depth=depth)
        discovered_objects[obj_id] = (current_obj, current_name, methods, depth)
        
        # Cache the methods for this type
        if obj_type not in type_method_cache:
            type_method_cache[obj_type] = set(methods['total'].keys())
        
        # Try calling each method to discover new objects
        if depth < max_depth:
            for method_name in methods['total'].keys():
                try:
                    method_func = getattr(current_obj, method_name)
                    if not callable(method_func):
                        continue
                        
                    cache_key = (obj_id, method_name)
                    if cache_key in method_results_cache:
                        result = method_results_cache[cache_key]
                    else:
                        result = safe_call_method(current_obj, method_name, method_func)
                        method_results_cache[cache_key] = result
                    
                    if result is not None and hasattr(result, '__class__'):
                        result_type = type(result).__name__
                        
                        # Skip primitive types and common Python objects
                        if result_type in ['int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set', 'NoneType']:
                            continue
                        
                        # Skip standard library and third-party objects
                        if result_type.startswith(('pandas.', 'numpy.', 'scipy.', 'networkx.', 'matplotlib.')):
                            continue
                        
                        # Skip if this type was already processed (avoid pandas/numpy spam)
                        if result_type in type_method_cache and not is_groggy_object(result):
                            continue
                            
                        if result is current_obj:  # Don't recurse into self
                            continue
                            
                        result_obj_id = get_object_id(result)
                        if result_obj_id not in discovered_objects:
                            result_name = f"{current_name}.{method_name}()"
                            queue.append((result, result_name, depth + 1))
                            print(f"{'  ' * (depth+1)}🔗 Found: {result_name} -> {result_type}")
                            
                except Exception:
                    continue
    
    print(f"\n✅ **BFS Discovery Complete**: Found {len(discovered_objects)} unique Groggy objects\n")
    return discovered_objects

def discover_all_available_methods(obj, obj_name, depth=0):
    """Discover ALL methods available on an object through any mechanism"""
    print(f"\n## {obj_name} Method Signatures\n")
    
    # Get methods visible in dir()
    dir_methods = {}
    for name in dir(obj):
        if not name.startswith('_') or name in ['__str__', '__repr__', '__len__', '__getitem__', '__iter__']:
            signature = get_method_signature(obj, name)
            dir_methods[name] = signature
    
    print(f"**Direct Methods ({len(dir_methods)})**:\n")
    
    # Print all direct methods with signatures
    for method_name in sorted(dir_methods.keys()):
        signature = dir_methods[method_name]
        if signature == "property":
            print(f"- `{method_name}` → property")
        else:
            print(f"- `{method_name}{signature}`")
    
    # Strategy 1: Test delegation targets
    hidden_methods = {}
    
    # Check if this object has methods that delegate to other objects
    delegation_candidates = []
    
    # Look for methods that return other objects that might be delegation targets
    for attr_name in dir_methods.keys():
        try:
            attr = getattr(obj, attr_name)
            if callable(attr):
                # Try calling methods that might return delegatable objects
                delegation_method_names = [
                    'base_table', 'table', 'into_base_table',  # Table delegations
                    'subgraph', 'view',  # Graph delegations
                    'neighborhood',  # Neighborhood delegations
                ]
                
                if attr_name in delegation_method_names:
                    try:
                        # Special handling for methods that need arguments
                        if attr_name == 'subgraph':
                            # Graph.subgraph needs node list - try with all nodes
                            if hasattr(obj, 'node_ids') and len(obj.node_ids()) > 0:
                                result = attr(obj.node_ids()[:2])  # Use first 2 nodes
                            else:
                                continue
                        elif attr_name == 'neighborhood':
                            # Neighborhood needs center nodes
                            if hasattr(obj, 'node_ids') and len(obj.node_ids()) > 0:
                                result = attr([obj.node_ids()[0]], radius=1)
                            else:
                                continue
                        else:
                            # No-argument methods
                            result = attr()
                            
                        if hasattr(result, '__class__') and result != obj:
                            delegation_candidates.append((attr_name, result))
                            print(f"\n**Delegation via {attr_name}() → {type(result).__name__}**")
                    except Exception as e:
                        print(f"  ⚠️  Could not test {attr_name}(): {e}")
        except:
            continue
    
    # Strategy 2: For each delegation candidate, test if their methods work on the original object
    for delegate_name, delegate_obj in delegation_candidates:
        delegate_methods = {}
        for method_name in dir(delegate_obj):
            if not method_name.startswith('_') or method_name in ['__str__', '__repr__', '__len__', '__getitem__', '__iter__']:
                if method_name not in dir_methods:  # Not directly visible
                    try:
                        # Test if the method actually works on the original object
                        method = getattr(obj, method_name, None)
                        if method is not None:
                            signature = get_method_signature(obj, method_name)
                            delegate_methods[method_name] = signature
                    except:
                        continue
        
        if delegate_methods:
            print(f"\n**Hidden Methods via {delegate_name} ({len(delegate_methods)})**:")
            for method_name in sorted(delegate_methods.keys()):
                signature = delegate_methods[method_name]
                if signature == "property":
                    print(f"- `{method_name}` → property")
                else:
                    print(f"- `{method_name}{signature}`")
            hidden_methods.update(delegate_methods)
    
    # Total summary
    total_methods = {**dir_methods, **hidden_methods}
    
    print(f"\n**Summary**: {len(total_methods)} total methods ({len(dir_methods)} direct + {len(hidden_methods)} delegated)")
    
    return {
        'direct': dir_methods,
        'hidden': hidden_methods,
        'total': total_methods
    }

def discover_all_groggy_objects():
    """Discover all object types available in the groggy module"""
    print("🔍 **Discovering all Groggy objects...**\n")
    
    # Create comprehensive test data
    g = gr.Graph()
    test_nodes = g.add_nodes([
        {'name': 'Alice', 'age': 25, 'salary': 75000, 'active': True, 'team': 'Engineering'},
        {'name': 'Bob', 'age': 30, 'salary': 85000, 'active': True, 'team': 'Sales'},
        {'name': 'Charlie', 'age': 35, 'salary': 95000, 'active': False, 'team': 'Marketing'},
        {'name': 'Diana', 'age': 28, 'salary': 80000, 'active': True, 'team': 'Engineering'},
    ])
    
    test_edges = g.add_edges([
        (test_nodes[0], test_nodes[1], {'weight': 1.5, 'type': 'collaboration'}),
        (test_nodes[1], test_nodes[2], {'weight': 2.0, 'type': 'reports_to'}),
        (test_nodes[2], test_nodes[3], {'weight': 0.8, 'type': 'peer'}),
    ])
    
    objects_to_test = []
    
    # Core graph object
    objects_to_test.append((g, "Graph"))
    
    # Table objects
    try:
        nodes_table = g.nodes.table()
        objects_to_test.append((nodes_table, "NodesTable"))
        print(f"✓ NodesTable: {type(nodes_table).__name__}")
    except Exception as e:
        print(f"✗ NodesTable failed: {e}")
    
    try:
        edges_table = g.edges.table()
        objects_to_test.append((edges_table, "EdgesTable"))
        print(f"✓ EdgesTable: {type(edges_table).__name__}")
    except Exception as e:
        print(f"✗ EdgesTable failed: {e}")
    
    try:
        graph_table = g.table()
        objects_to_test.append((graph_table, "GraphTable"))
        print(f"✓ GraphTable: {type(graph_table).__name__}")
    except Exception as e:
        print(f"✗ GraphTable failed: {e}")
    
    # Array objects
    try:
        nodes_table = g.nodes.table()
        node_ids_array = nodes_table["node_id"]
        objects_to_test.append((node_ids_array, "GraphArray (node_id)"))
        print(f"✓ GraphArray: {type(node_ids_array).__name__}")
        
        # Try different column types
        if "age" in nodes_table.column_names():
            age_array = nodes_table["age"]
            objects_to_test.append((age_array, "GraphArray (age)"))
            print(f"✓ GraphArray (age): {type(age_array).__name__}")
            
    except Exception as e:
        print(f"✗ GraphArray failed: {e}")
    
    # Matrix objects
    try:
        adj_matrix = g.adjacency_matrix()
        objects_to_test.append((adj_matrix, "GraphMatrix (adjacency)"))
        print(f"✓ GraphMatrix: {type(adj_matrix).__name__}")
    except Exception as e:
        print(f"✗ GraphMatrix failed: {e}")
    
    try:
        laplacian_matrix = g.laplacian_matrix()
        objects_to_test.append((laplacian_matrix, "GraphMatrix (laplacian)"))
        print(f"✓ Laplacian Matrix: {type(laplacian_matrix).__name__}")
    except Exception as e:
        print(f"✗ Laplacian Matrix failed: {e}")
    
    # Subgraph objects - this is crucial!
    try:
        subgraph = g.subgraph(test_nodes[:2])
        objects_to_test.append((subgraph, "Subgraph"))
        print(f"✓ Subgraph: {type(subgraph).__name__}")
    except Exception as e:
        print(f"✗ Subgraph failed: {e}")
    
    # Neighborhood objects
    try:
        neighborhood = g.neighborhood([test_nodes[0]], radius=2)
        objects_to_test.append((neighborhood, "Neighborhood"))
        print(f"✓ Neighborhood: {type(neighborhood).__name__}")
    except Exception as e:
        print(f"✗ Neighborhood failed: {e}")
    
    # View objects
    try:
        view = g.view()
        objects_to_test.append((view, "GraphView"))
        print(f"✓ GraphView: {type(view).__name__}")
    except Exception as e:
        print(f"✗ GraphView failed: {e}")
    
    # Filter objects
    try:
        # Try to create filter objects
        node_filter = gr.NodeFilter("age > 25")
        objects_to_test.append((node_filter, "NodeFilter"))
        print(f"✓ NodeFilter: {type(node_filter).__name__}")
    except Exception as e:
        print(f"✗ NodeFilter failed: {e}")
    
    try:
        edge_filter = gr.EdgeFilter("weight > 1.0")
        objects_to_test.append((edge_filter, "EdgeFilter"))
        print(f"✓ EdgeFilter: {type(edge_filter).__name__}")
    except Exception as e:
        print(f"✗ EdgeFilter failed: {e}")
    
    # Check what else is available in the groggy module
    print(f"\n🔍 **Checking groggy module contents...**")
    for name in dir(gr):
        if not name.startswith('_'):
            try:
                obj = getattr(gr, name)
                if callable(obj) and hasattr(obj, '__class__'):
                    # Try to create instances of callable classes
                    if name not in ['NodeFilter', 'EdgeFilter', 'Graph']:  # Already tested above
                        try:
                            if name == 'AttributeFilter':
                                instance = obj("name", "Alice")
                                objects_to_test.append((instance, name))
                                print(f"✓ {name}: {type(instance).__name__}")
                        except:
                            print(f"✗ {name}: Could not instantiate")
                elif not callable(obj):
                    print(f"→ {name}: {type(obj).__name__} (non-callable)")
            except:
                continue
    
    print(f"\n**Total objects discovered: {len(objects_to_test)}**\n")
    return objects_to_test

def main():
    """Comprehensively discover all available methods using BFS"""
    print("# Complete Groggy Python API Reference (BFS Discovery)")
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nBreadth-first search discovery of ALL Groggy objects and their methods.\n")
    
    # Get initial seed objects
    seed_objects = discover_all_groggy_objects()
    
    # Run BFS discovery
    discovered_objects = discover_all_available_methods_bfs(seed_objects, max_depth=2)
    
    # Generate report organized by discovery depth
    print("# Complete API Reference\n")
    
    # Group by depth
    by_depth = {}
    for obj_id, (obj, name, methods, depth) in discovered_objects.items():
        if depth not in by_depth:
            by_depth[depth] = []
        by_depth[depth].append((obj, name, methods, depth))
    
    total_objects = 0
    total_methods = 0
    
    # Report each depth level
    for depth in sorted(by_depth.keys()):
        objects_at_depth = by_depth[depth]
        total_objects += len(objects_at_depth)
        
        print(f"## Depth {depth} Objects ({len(objects_at_depth)} objects)\n")
        
        for obj, name, methods, _ in objects_at_depth:
            method_count = len(methods['total'])
            total_methods += method_count
            
            print(f"### {name}")
            print(f"**Type**: `{type(obj).__name__}`")
            print(f"**Methods**: {method_count} ({len(methods['direct'])} direct + {len(methods['hidden'])} delegated)")
            
            # Show top 10 most interesting methods
            all_methods = list(methods['total'].keys())
            interesting_methods = [m for m in all_methods if not m.startswith('__')][:10]
            
            print(f"**Key Methods**: {', '.join(f'`{m}`' for m in interesting_methods)}")
            if len(all_methods) > 10:
                print(f"... and {len(all_methods) - 10} more")
            print()
    
    # Final summary
    print(f"# BFS Discovery Summary\n")
    print(f"**Total Objects Discovered**: {total_objects}")
    print(f"**Total Methods Discovered**: {total_methods}")
    print(f"**Average Methods per Object**: {total_methods/total_objects:.1f}")
    
    print(f"\n**Discovery by Depth**:")
    for depth in sorted(by_depth.keys()):
        objects_at_depth = by_depth[depth]
        method_sum = sum(len(methods['total']) for _, _, methods, _ in objects_at_depth)
        print(f"- **Depth {depth}**: {len(objects_at_depth)} objects, {method_sum} methods")
    
    print(f"\n---\n*Complete API surface discovered through BFS traversal*")

if __name__ == "__main__":
    main()