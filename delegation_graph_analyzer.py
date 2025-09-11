#!/usr/bin/env python3
"""
Method Delegation Graph Builder
Creates a groggy graph where nodes are types and edges are methods showing input/output relationships
"""

import groggy as gr

def analyze_method_signatures():
    """Analyze all methods and their return types to build delegation graph"""
    
    print("🔍 ANALYZING METHOD DELEGATION RELATIONSHIPS")
    print("=" * 60)
    
    g = gr.karate_club()
    
    # Create instances of all main types
    type_instances = {}
    
    try:
        type_instances['Graph'] = g
        type_instances['Subgraph'] = g.nodes.all()  
        type_instances['GraphArray'] = g.nodes.ids()
        type_instances['GraphMatrix'] = g.dense_adjacency_matrix()
        type_instances['ComponentsArray'] = g.nodes.all().connected_components()
        type_instances['NodesTable'] = g.table()
        type_instances['EdgesTable'] = g.nodes.all().edges_table()
        type_instances['NeighborhoodResult'] = g.nodes.all().neighborhood([0], 1)
    except Exception as e:
        print(f"Error creating instances: {e}")
        return
    
    # Collect all methods for each type
    type_methods = {}
    for type_name, instance in type_instances.items():
        methods = [m for m in dir(instance) if not m.startswith('_')]
        type_methods[type_name] = methods
        print(f"{type_name}: {len(methods)} methods")
    
    print("\n🧪 TESTING METHOD RETURN TYPES")
    print("-" * 40)
    
    # Test methods to find return types (delegation relationships)
    delegation_edges = []
    
    for source_type, instance in type_instances.items():
        print(f"\n📍 Testing {source_type} methods:")
        
        # Test each method to see what type it returns
        for method_name in type_methods[source_type]:
            try:
                method = getattr(instance, method_name)
                
                # Skip if it's a property, not a method
                if not callable(method):
                    continue
                
                # Try to call method with common parameters
                result = None
                result_type = None
                
                # Try different parameter combinations for common methods
                if method_name in ['bfs', 'dfs']:
                    result = method(0)
                elif method_name in ['sample']:
                    result = method(10)
                elif method_name in ['head']:
                    result = method(5)
                elif method_name in ['neighborhood']:
                    result = method([0], 1)
                elif method_name in ['percentile']:
                    result = method(50)
                elif method_name in ['filter_nodes', 'filter_edges']:
                    # Skip filter methods for now - need proper query syntax
                    continue
                elif method_name in ['get_cell', 'get_row', 'get_column']:
                    result = method(0)
                elif method_name in ['mean_axis', 'sum_axis', 'std_axis']:
                    result = method(0)
                else:
                    # Try calling with no parameters
                    result = method()
                
                if result is not None:
                    result_type = type(result).__name__
                    if 'groggy' in str(type(result)):
                        result_type = str(type(result)).split('.')[-1].replace("'>", "")
                    elif 'builtins' in str(type(result)):
                        result_type = str(type(result)).split('.')[-1].replace("'>", "")
                    
                    # Add edge to delegation graph
                    delegation_edges.append((source_type, method_name, result_type))
                    print(f"   {method_name}() → {result_type}")
                
            except Exception as e:
                # Method requires parameters we don't have or has other issues
                continue
    
    return delegation_edges, type_methods

def create_delegation_graph(delegation_edges, type_methods):
    """Create a groggy graph from the delegation relationships"""
    
    print(f"\n🏗️  BUILDING DELEGATION GRAPH")
    print("-" * 40)
    
    # Create new graph for delegation relationships
    delegation_graph = gr.Graph()
    
    # Add nodes for each type
    type_nodes = {}
    for i, type_name in enumerate(['Graph', 'Subgraph', 'GraphArray', 'GraphMatrix', 
                                   'ComponentsArray', 'NodesTable', 'EdgesTable', 'NeighborhoodResult']):
        node_id = delegation_graph.add_node()
        delegation_graph.set_node_attr(node_id, "type_name", type_name)
        delegation_graph.set_node_attr(node_id, "method_count", len(type_methods.get(type_name, [])))
        type_nodes[type_name] = node_id
        print(f"   Added node {node_id}: {type_name} ({len(type_methods.get(type_name, []))} methods)")
    
    # Add edges for method delegation relationships
    edge_count = 0
    for source_type, method_name, target_type in delegation_edges:
        if source_type in type_nodes and target_type in type_nodes:
            source_node = type_nodes[source_type]
            target_node = type_nodes[target_type]
            
            edge_id = delegation_graph.add_edge(source_node, target_node)
            delegation_graph.set_edge_attr(edge_id, "method_name", method_name)
            delegation_graph.set_edge_attr(edge_id, "delegation_type", f"{source_type}→{target_type}")
            edge_count += 1
    
    print(f"   Added {edge_count} delegation edges")
    
    return delegation_graph, type_nodes

def analyze_delegation_patterns(delegation_graph, type_nodes):
    """Analyze the delegation patterns in the graph"""
    
    print(f"\n📊 DELEGATION PATTERN ANALYSIS")
    print("-" * 40)
    
    # Analyze each type's delegation patterns
    for type_name, node_id in type_nodes.items():
        
        # Outgoing delegations (methods that return other types)
        out_neighbors = delegation_graph.neighbors(node_id)
        out_edges = []
        for neighbor in out_neighbors:
            edge_id = delegation_graph.edges_between(node_id, neighbor)[0] if delegation_graph.edges_between(node_id, neighbor) else None
            if edge_id:
                method_name = delegation_graph.get_edge_attr(edge_id, "method_name")
                target_type = delegation_graph.get_node_attr(neighbor, "type_name") 
                out_edges.append((method_name, target_type))
        
        # Incoming delegations (other types that can create this type)
        in_edges = []
        for other_name, other_node in type_nodes.items():
            if other_node != node_id:
                edges = delegation_graph.edges_between(other_node, node_id)
                for edge_id in edges:
                    method_name = delegation_graph.get_edge_attr(edge_id, "method_name")
                    in_edges.append((other_name, method_name))
        
        print(f"\n📍 {type_name}")
        print(f"   Outgoing: {len(out_edges)} delegations")
        for method, target in out_edges[:5]:  # Show first 5
            print(f"     {method}() → {target}")
        if len(out_edges) > 5:
            print(f"     ... and {len(out_edges) - 5} more")
            
        print(f"   Incoming: {len(in_edges)} delegations") 
        for source, method in in_edges[:3]:  # Show first 3
            print(f"     {source}.{method}() → {type_name}")
        if len(in_edges) > 3:
            print(f"     ... and {len(in_edges) - 3} more")
    
    return delegation_graph

def find_delegation_bugs(delegation_edges, type_methods):
    """Look for potential bugs in delegation patterns"""
    
    print(f"\n🐛 BUG DETECTION")
    print("-" * 40)
    
    # Group methods by return type to find inconsistencies
    methods_by_return_type = {}
    for source_type, method_name, target_type in delegation_edges:
        if target_type not in methods_by_return_type:
            methods_by_return_type[target_type] = []
        methods_by_return_type[target_type].append((source_type, method_name))
    
    print("Methods that return each type:")
    for return_type, methods in methods_by_return_type.items():
        print(f"\n{return_type} returned by:")
        for source_type, method_name in methods:
            print(f"   {source_type}.{method_name}()")
    
    # Look for missing delegation patterns
    print(f"\n🔍 POTENTIAL MISSING DELEGATIONS:")
    
    # Check if all types can be created from Graph
    graph_methods = [method for source, method, target in delegation_edges if source == 'Graph']
    graph_targets = [target for source, method, target in delegation_edges if source == 'Graph'] 
    
    all_types = set(['Subgraph', 'GraphArray', 'GraphMatrix', 'ComponentsArray', 'NodesTable', 'EdgesTable'])
    missing_from_graph = all_types - set(graph_targets)
    
    if missing_from_graph:
        print(f"   Types not directly creatable from Graph: {missing_from_graph}")
    else:
        print(f"   ✓ All major types can be created from Graph")

def main():
    """Main analysis function"""
    
    print("🚀 GROGGY DELEGATION GRAPH ANALYSIS")
    print("=" * 60)
    
    # Step 1: Analyze method signatures and return types
    delegation_edges, type_methods = analyze_method_signatures()
    
    # Step 2: Create delegation graph
    delegation_graph, type_nodes = create_delegation_graph(delegation_edges, type_methods)
    
    # Step 3: Analyze delegation patterns
    analyzed_graph = analyze_delegation_patterns(delegation_graph, type_nodes)
    
    # Step 4: Find potential bugs
    find_delegation_bugs(delegation_edges, type_methods)
    
    print(f"\n📈 SUMMARY")
    print("-" * 40)
    print(f"Total delegation edges found: {len(delegation_edges)}")
    print(f"Types analyzed: {len(type_nodes)}")
    print(f"Graph nodes: {delegation_graph.node_count()}")
    print(f"Graph edges: {delegation_graph.edge_count()}")
    
    return delegation_graph

if __name__ == "__main__":
    delegation_graph = main()
