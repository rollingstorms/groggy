#!/usr/bin/env python3
"""
Groggy Architecture Graph Generator
Creates a groggy graph of the main objects and their delegation relationships
"""

import groggy as gr
import json

def get_main_object_types():
    """Get instances of all main object types in the groggy architecture"""
    
    g = gr.karate_club()
    
    main_objects = {}
    
    try:
        # Core types
        main_objects['Graph'] = g
        main_objects['BaseArray'] = g.nodes.ids()  # GraphArray extends BaseArray
        main_objects['BaseTable'] = g.table()  # GraphTable extends BaseTable  
        main_objects['Matrix'] = g.dense_adjacency_matrix()  # GraphMatrix
        main_objects['Subgraph'] = g.nodes.all()
        
        # Specialized arrays
        main_objects['NodesArray'] = g.nodes.ids()  # Represents node arrays
        main_objects['EdgesArray'] = g.edge_ids  # Represents edge arrays
        main_objects['SubgraphArray'] = g.nodes.all().connected_components().sample(1)  # SubgraphArray
        
        # Table types
        main_objects['NodesTable'] = g.nodes.all().table()
        main_objects['EdgesTable'] = g.nodes.all().edges_table()
        
        # Component types
        main_objects['ComponentsArray'] = g.nodes.all().connected_components()
        main_objects['NeighborhoodResult'] = g.nodes.all().neighborhood([0], 1)
        
        print(f"✓ Created {len(main_objects)} main object types")
        
    except Exception as e:
        print(f"Error creating main objects: {e}")
    
    return main_objects

def collect_all_methods(main_objects):
    """Collect all methods for each main object type"""
    
    type_methods = {}
    
    for type_name, instance in main_objects.items():
        if instance is not None:
            # Get all public methods (not starting with _)
            methods = []
            for attr in dir(instance):
                if not attr.startswith('_'):
                    attr_obj = getattr(instance, attr)
                    if callable(attr_obj):
                        methods.append(attr)
                    else:
                        # Also include properties/attributes
                        methods.append(f"{attr} (property)")
            
            type_methods[type_name] = sorted(methods)
            print(f"{type_name}: {len(methods)} methods/properties")
        else:
            type_methods[type_name] = []
            print(f"{type_name}: Could not create instance")
    
    return type_methods



def analyze_method_return_types(main_objects, type_methods):
    """Comprehensively analyze each method to determine its return type"""
    
    print("\n🔍 COMPREHENSIVE METHOD RETURN TYPE ANALYSIS")
    print("-" * 50)
    
    method_mappings = []
    
    for type_name, instance in main_objects.items():
        if instance is None:
            continue
            
        methods = type_methods.get(type_name, [])
        callable_methods = [m for m in methods if "(property)" not in m]
        property_methods = [m for m in methods if "(property)" in m]
        
        print(f"\n📍 Analyzing {type_name}:")
        print(f"   {len(callable_methods)} callable methods + {len(property_methods)} properties")
        
        successful_tests = 0
        
        # Handle properties first (easier to test)
        for prop_name in property_methods:
            try:
                clean_prop_name = prop_name.replace(" (property)", "")
                prop_value = getattr(instance, clean_prop_name)
                if prop_value is not None:
                    result_type = type(prop_value).__name__
                    if 'groggy' in str(type(prop_value)) or 'builtins' in str(type(prop_value)):
                        result_type = str(type(prop_value)).split('.')[-1].replace("'>", "").replace("'", "")
                    
                    target_type = map_result_to_main_type(result_type)
                    if target_type:
                        method_mappings.append((type_name, prop_name, target_type, result_type))
                        print(f"   {clean_prop_name} (prop) → {target_type}")
                        successful_tests += 1
            except:
                continue
        
        # Handle callable methods with comprehensive parameter testing
        for method_name in callable_methods:
            try:
                method = getattr(instance, method_name)
                if not callable(method):
                    continue
                
                result = None
                result_type = None
                successful_strategy = None
                
                # Comprehensive parameter testing strategies
                strategies = [
                    # Zero parameters
                    (lambda m: m(), "no_params"),
                    
                    # Single parameters - integers
                    (lambda m: m(0), "int_0"),
                    (lambda m: m(1), "int_1"), 
                    (lambda m: m(2), "int_2"),
                    (lambda m: m(3), "int_3"),
                    (lambda m: m(5), "int_5"),
                    (lambda m: m(10), "int_10"),
                    
                    # Lists
                    (lambda m: m([]), "empty_list"),
                    (lambda m: m([0]), "list_0"),
                    (lambda m: m([0, 1]), "list_pair"),
                    (lambda m: m([0, 1, 2]), "list_triple"),
                    
                    # List with int (neighborhood pattern)
                    (lambda m: m([0], 1), "list_int_1"),
                    (lambda m: m([0], 2), "list_int_2"),
                    (lambda m: m([0, 1], 1), "list_pair_int_1"),
                    
                    # Floats/percentiles
                    (lambda m: m(0.5), "float_half"),
                    (lambda m: m(25.0), "float_25"),
                    (lambda m: m(50.0), "float_50"),
                    (lambda m: m(75.0), "float_75"),
                    (lambda m: m(90.0), "float_90"),
                    
                    # Strings
                    (lambda m: m(""), "empty_string"),
                    (lambda m: m("id"), "string_id"),
                    (lambda m: m("name"), "string_name"),
                    (lambda m: m("degree"), "string_degree"),
                    (lambda m: m("weight"), "string_weight"),
                    (lambda m: m("degree > 0"), "filter_degree"),
                    (lambda m: m("id == 0"), "filter_id"),
                    
                    # Booleans
                    (lambda m: m(True), "bool_true"),
                    (lambda m: m(False), "bool_false"),
                    
                    # Multiple parameters
                    (lambda m: m(0, 1), "two_ints"),
                    (lambda m: m(0, 1, 2), "three_ints"),
                    (lambda m: m("id", "asc"), "string_string"),
                ]
                
                # Test each strategy
                for strategy_func, strategy_name in strategies:
                    try:
                        result = strategy_func(method)
                        successful_strategy = strategy_name
                        break
                    except Exception:
                        continue
                
                # Process result if we got one
                if result is not None:
                    result_type = type(result).__name__ 
                    if 'groggy' in str(type(result)) or 'builtins' in str(type(result)):
                        result_type = str(type(result)).split('.')[-1].replace("'>", "").replace("'", "")
                    
                    target_type = map_result_to_main_type(result_type)
                    if target_type:
                        method_mappings.append((type_name, method_name, target_type, result_type))
                        print(f"   {method_name}({successful_strategy}) → {target_type}")
                        successful_tests += 1
                
            except Exception:
                continue
        
        print(f"   ✓ Successfully tested {successful_tests}/{len(methods)} total methods")
    
    print(f"\n🎯 TOTAL METHOD MAPPINGS FOUND: {len(method_mappings)}")
    print(f"   This should be much closer to the ~160+ total methods!")
    return method_mappings

def map_result_to_main_type(result_type):
    """Map specific result types to our main object categories"""
    mapping = {
        'Subgraph': 'Subgraph',
        'Graph': 'Graph', 
        'GraphArray': 'BaseArray',
        'GraphMatrix': 'Matrix',
        'ComponentsArray': 'ComponentsArray',
        'NeighborhoodResult': 'NeighborhoodResult',
        'NodesTable': 'NodesTable',
        'EdgesTable': 'EdgesTable',
        'GraphTable': 'BaseTable',
        'BaseTable': 'BaseTable',
        'SubgraphArray': 'SubgraphArray',
        'ndarray': 'BaseArray',  # numpy arrays map to BaseArray
        'list': 'BaseArray',     # lists map to BaseArray
        'dict': 'BaseArray',     # dicts map to BaseArray  
        'int': None,            # primitives don't need edges
        'float': None,
        'str': None,
        'bool': None,
        'tuple': None,
    }
    return mapping.get(result_type)

def create_groggy_architecture_graph(main_objects, type_methods, method_mappings):
    """Create the groggy graph where edges are methods connecting type nodes"""
    
    print("\n🏗️  Creating Method-Edge Architecture Graph")
    print("-" * 50)
    
    # Create new graph
    arch_graph = gr.Graph()
    
    # Add nodes for each main object type
    type_nodes = {}
    for type_name in main_objects.keys():
        node_id = arch_graph.add_node()
        arch_graph.set_node_attr(node_id, "object_type", type_name)
        arch_graph.set_node_attr(node_id, "method_count", len(type_methods.get(type_name, [])))
        
        # Add category based on type
        if type_name in ['Graph']:
            arch_graph.set_node_attr(node_id, "category", "Core")
        elif type_name in ['BaseArray', 'BaseTable', 'Matrix', 'Subgraph']:
            arch_graph.set_node_attr(node_id, "category", "Base")
        elif 'Array' in type_name:
            arch_graph.set_node_attr(node_id, "category", "Array")
        elif 'Table' in type_name:
            arch_graph.set_node_attr(node_id, "category", "Table")
        else:
            arch_graph.set_node_attr(node_id, "category", "Specialized")
        
        type_nodes[type_name] = node_id
        print(f"   Added {type_name} (node {node_id}): {len(type_methods.get(type_name, []))} methods")
    
    # Add edges for each method (source type → target type)
    edge_count = 0
    for source_type, method_name, target_type, original_return_type in method_mappings:
        if source_type in type_nodes and target_type in type_nodes:
            source_node = type_nodes[source_type]
            target_node = type_nodes[target_type]
            
            # Each method is an edge
            edge_id = arch_graph.add_edge(source_node, target_node)
            arch_graph.set_edge_attr(edge_id, "method_name", method_name)
            arch_graph.set_edge_attr(edge_id, "source_type", source_type)
            arch_graph.set_edge_attr(edge_id, "target_type", target_type)
            arch_graph.set_edge_attr(edge_id, "return_type", original_return_type)
            arch_graph.set_edge_attr(edge_id, "delegation_pattern", f"{source_type}.{method_name}() → {target_type}")
            edge_count += 1
    
    print(f"   Added {edge_count} method edges")
    print(f"   Graph: {arch_graph.node_count()} nodes, {arch_graph.edge_count()} edges")
    
    return arch_graph, type_nodes

def export_graph_data(arch_graph, type_nodes, type_methods):
    """Export the graph to nodes and edges files using groggy's table.to_csv() method"""
    
    print("\n📄 Exporting Graph Data")
    print("-" * 30)
    
    # Create nodes table data for CSV export
    nodes_list = []
    for type_name, node_id in type_nodes.items():
        methods_list = type_methods.get(type_name, [])
        sample_methods = '; '.join(methods_list[:5]) if methods_list else ""
        all_methods = '|'.join(methods_list)  # All methods separated by |
        
        nodes_list.append([
            int(node_id),
            arch_graph.get_node_attr(node_id, "object_type"),
            arch_graph.get_node_attr(node_id, "category"), 
            arch_graph.get_node_attr(node_id, "method_count"),
            sample_methods,
            all_methods
        ])
    
    # Create edges table data (each edge represents a method)
    edges_list = []
    for edge_id in arch_graph.edge_ids.to_list():
        source, target = arch_graph.edge_endpoints(edge_id)
        edges_list.append([
            int(edge_id),
            int(source),
            int(target),
            arch_graph.get_edge_attr(edge_id, "method_name"),
            arch_graph.get_edge_attr(edge_id, "source_type"),
            arch_graph.get_edge_attr(edge_id, "target_type"),
            arch_graph.get_edge_attr(edge_id, "return_type")
        ])
    
    try:
        # Convert to groggy table format (dict of lists)
        nodes_dict = {
            'id': [row[0] for row in nodes_list],
            'object_type': [row[1] for row in nodes_list],  
            'category': [row[2] for row in nodes_list],
            'method_count': [row[3] for row in nodes_list],
            'sample_methods': [row[4] for row in nodes_list],
            'all_methods': [row[5] for row in nodes_list]
        }
        
        edges_dict = {
            'edge_id': [row[0] for row in edges_list],
            'source_node': [row[1] for row in edges_list],
            'target_node': [row[2] for row in edges_list],
            'method_name': [row[3] for row in edges_list],
            'source_type': [row[4] for row in edges_list],
            'target_type': [row[5] for row in edges_list],
            'return_type': [row[6] for row in edges_list]
        }
        
        # Create groggy tables and use native to_csv method
        try:
            nodes_table = gr.table(nodes_dict)
            nodes_table.to_csv('groggy_architecture_nodes.csv')
            print(f"✓ Exported {len(nodes_list)} nodes using groggy table.to_csv(): groggy_architecture_nodes.csv")
        except Exception as e:
            print(f"   ⚠️ groggy nodes table.to_csv() failed: {e}")
            # Fallback to pandas
            import pandas as pd
            nodes_df = pd.DataFrame(nodes_list, columns=[
                'id', 'object_type', 'category', 'method_count', 'sample_methods', 'all_methods'
            ])
            nodes_df.to_csv('groggy_architecture_nodes.csv', index=False)
            print(f"   ✓ Fallback - exported nodes using pandas: groggy_architecture_nodes.csv")
        
        try:
            edges_table = gr.table(edges_dict)
            edges_table.to_csv('groggy_architecture_edges.csv')
            print(f"✓ Exported {len(edges_list)} edges using groggy table.to_csv(): groggy_architecture_edges.csv")
        except Exception as e:
            print(f"   ⚠️ groggy edges table.to_csv() failed: {e}")
            # Fallback to pandas
            import pandas as pd
            edges_df = pd.DataFrame(edges_list, columns=[
                'edge_id', 'source_node', 'target_node', 'method_name', 'source_type', 'target_type', 'return_type'
            ])
            edges_df.to_csv('groggy_architecture_edges.csv', index=False)
            print(f"   ✓ Fallback - exported edges using pandas: groggy_architecture_edges.csv")
            
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
        print("   Creating simple CSV manually...")
        
        # Manual CSV fallback
        with open('groggy_architecture_nodes.csv', 'w') as f:
            f.write("id,object_type,category,method_count,sample_methods\n")
            for row in nodes_list:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},\"{row[4]}\"\n")
        
        with open('groggy_architecture_edges.csv', 'w') as f:
            f.write("id,source,target,method_name,delegation_type\n")
            for row in edges_list:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")
        
        print(f"   ✓ Manual CSV export completed")
    
    # Also create JSON for completeness
    nodes_data = [
        {
            'id': int(node_id),
            'object_type': arch_graph.get_node_attr(node_id, "object_type"),
            'category': arch_graph.get_node_attr(node_id, "category"),
            'method_count': arch_graph.get_node_attr(node_id, "method_count"),
            'methods': type_methods.get(type_name, [])
        }
        for type_name, node_id in type_nodes.items()
    ]
    
    with open('groggy_architecture_nodes.json', 'w') as f:
        json.dump(nodes_data, f, indent=2)
    print(f"✓ Also exported JSON: groggy_architecture_nodes.json")

def analyze_architecture_patterns(arch_graph, type_nodes):
    """Analyze the architectural patterns in the delegation graph"""
    
    print("\n📊 Architecture Pattern Analysis")
    print("-" * 40)
    
    # Group nodes by category
    categories = {}
    for type_name, node_id in type_nodes.items():
        category = arch_graph.get_node_attr(node_id, "category")
        if category not in categories:
            categories[category] = []
        categories[category].append(type_name)
    
    print("Objects by category:")
    for category, objects in categories.items():
        print(f"   {category}: {', '.join(objects)}")
    
    # Analyze delegation patterns
    print("\nDelegation patterns:")
    for type_name, node_id in type_nodes.items():
        neighbors = arch_graph.neighbors(node_id)
        if neighbors:
            neighbor_names = []
            for neighbor_id in neighbors:
                neighbor_name = arch_graph.get_node_attr(neighbor_id, "object_type")
                neighbor_names.append(neighbor_name)
            print(f"   {type_name} → {', '.join(neighbor_names)}")
    
    # Find most connected objects
    connection_counts = {}
    for type_name, node_id in type_nodes.items():
        in_degree = len([n for n in type_nodes.values() if node_id in arch_graph.neighbors(n)])
        out_degree = len(arch_graph.neighbors(node_id))
        connection_counts[type_name] = (in_degree, out_degree, in_degree + out_degree)
    
    print("\nMost connected objects (in_degree, out_degree, total):")
    sorted_connections = sorted(connection_counts.items(), key=lambda x: x[1][2], reverse=True)
    for type_name, (in_deg, out_deg, total) in sorted_connections[:5]:
        print(f"   {type_name}: ({in_deg}, {out_deg}, {total})")

def main():
    """Main function to generate the groggy architecture graph"""
    
    print("🚀 GROGGY ARCHITECTURE GRAPH GENERATOR")
    print("=" * 60)
    
    # Step 1: Get main object types
    print("\n📍 Step 1: Collecting Main Object Types")
    main_objects = get_main_object_types()
    
    # Step 2: Collect all methods
    print("\n📍 Step 2: Collecting Methods for Each Type")
    type_methods = collect_all_methods(main_objects)
    
    # Step 3: Analyze method return types
    print("\n📍 Step 3: Analyzing Method Return Types")
    method_mappings = analyze_method_return_types(main_objects, type_methods)
    
    # Step 4: Create the graph where each method is an edge
    print("\n📍 Step 4: Creating Method-Edge Architecture Graph")
    arch_graph, type_nodes = create_groggy_architecture_graph(main_objects, type_methods, method_mappings)
    
    # Step 5: Export graph data
    print("\n📍 Step 5: Exporting Graph Data")
    export_graph_data(arch_graph, type_nodes, type_methods)
    
    # Step 6: Analyze patterns
    print("\n📍 Step 6: Analyzing Architecture Patterns")
    analyze_architecture_patterns(arch_graph, type_nodes)
    
    print("\n✅ COMPLETE!")
    print(f"Generated architecture graph with {arch_graph.node_count()} nodes and {arch_graph.edge_count()} edges")
    print("Files created:")
    print("   - groggy_architecture_nodes.csv (using table.to_csv())")
    print("   - groggy_architecture_edges.csv (using table.to_csv())")
    print("   - groggy_architecture_nodes.json")
    
    return arch_graph

if __name__ == "__main__":
    arch_graph = main()
