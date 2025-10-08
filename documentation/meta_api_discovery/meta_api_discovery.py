#!/usr/bin/env python3
"""
Meta API Discovery and Testing System for Groggy

This system discovers every method available on all main Groggy objects,
builds a graph representation of the API structure, generates dynamic tests,
and provides this meta-graph as a canonical example.

Core Objects: Graph, Nodes, Edges, Subgraph, Table, Array, Matrix
Array Types: NodesArray, EdgesArray, SubgraphArray, TableArray, MatrixArray

Author: Meta API Discovery System
"""

import inspect
import sys
import traceback
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import json


class APIMethodDiscovery:
    """Core discovery engine for Groggy API methods"""
    
    def __init__(self):
        self.discovered_methods = defaultdict(list)
        self.method_signatures = {}
        self.method_return_types = {}
        self.delegation_chains = defaultdict(list)
        self.object_instances = {}
        self.discovery_errors = []
        
    def setup_groggy_environment(self):
        """Setup Groggy and create instances for discovery"""
        try:
            # Build groggy first
            print("Building Groggy...")
            import subprocess
            result = subprocess.run(
                ["maturin", "develop"], 
                cwd="/Users/michaelroth/Documents/Code/groggy/python-groggy",
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                print(f"Build failed: {result.stderr}")
                return False
            
            print("Importing Groggy...")
            sys.path.append("/Users/michaelroth/Documents/Code/groggy/")
            import groggy
            
            # Create sample data using the correct API patterns from PHASE_1_2_USAGE_EXAMPLE.py
            # Create tables first, then graph
            users_data = {
                'node_id': [1, 2, 3],
                'name': ['Alice', 'Bob', 'Charlie'],
                'type': ['user', 'user', 'user']
            }
            
            edges_data = {
                'edge_id': [1, 2],
                'source': [1, 2],
                'target': [2, 3],
                'relationship': ['friend', 'colleague'],
                'weight': [1.0, 2.0]
            }
            
            # Create BaseTable, NodesTable, EdgesTable instances
            base_table = groggy.BaseTable.from_dict(users_data)
            nodes_table = groggy.NodesTable.from_dict(users_data)
            edges_table = groggy.EdgesTable.from_dict(edges_data)
            graph_table = groggy.GraphTable(nodes_table, edges_table)
            
            # Create arrays
            base_array = groggy.BaseArray([1, 2, 3, 4, 5])
            
            # Store object instances for discovery
            self.object_instances = {
                'BaseTable': base_table,
                'NodesTable': nodes_table,
                'EdgesTable': edges_table,
                'GraphTable': graph_table,
                'BaseArray': base_array,
            }
            
            # Try to create Graph from GraphTable
            try:
                graph = graph_table.to_graph()
                self.object_instances['Graph'] = graph
                print(f"✓ Created Graph with {graph.node_count()} nodes, {graph.edge_count()} edges")
                
                # Try to get accessors
                try:
                    self.object_instances['Nodes'] = graph.nodes
                    print("✓ Got Nodes accessor")
                except Exception as e:
                    print(f"Could not get Nodes accessor: {e}")
                    
                try:
                    self.object_instances['Edges'] = graph.edges
                    print("✓ Got Edges accessor")
                except Exception as e:
                    print(f"Could not get Edges accessor: {e}")
                    
            except Exception as e:
                print(f"Could not create Graph: {e}")
                
            # Try to create other objects
            try:
                components = graph.connected_components()
                self.object_instances['ComponentsArray'] = components
                print("✓ Created ComponentsArray")
            except Exception as e:
                print(f"Could not create ComponentsArray: {e}")
                
            try:
                # Create Subgraph using g.view() and g.nodes.all()  
                subgraph = graph.view()
                self.object_instances['Subgraph'] = subgraph
                print("✓ Created Subgraph via graph.view()")
            except Exception as e:
                print(f"Could not create Subgraph: {e}")
                
            try:
                # Create SubgraphArray (from connected components)
                subgraph_array = components.iter() if 'ComponentsArray' in self.object_instances else None
                if subgraph_array:
                    self.object_instances['SubgraphArray'] = subgraph_array
                    print("✓ Created SubgraphArray")
            except Exception as e:
                print(f"Could not create SubgraphArray: {e}")
                
            try:
                # Create GraphArray
                graph_array = graph.nodes.ids()
                self.object_instances['GraphArray'] = graph_array
                print("✓ Created GraphArray")
            except Exception as e:
                print(f"Could not create GraphArray: {e}")
                
            try:
                matrix = graph.adjacency_matrix()
                self.object_instances['Matrix'] = matrix
                print("✓ Created Matrix")
            except Exception as e:
                print(f"Could not create Matrix: {e}")
                
            try:
                # Create BaseTable from nodes
                base_table = nodes_table.base_table()
                self.object_instances['BaseTable'] = base_table
                print("✓ Created BaseTable")
            except Exception as e:
                print(f"Could not create BaseTable: {e}")
                
            print(f"Successfully created {len(self.object_instances)} object instances")
            return True
            
        except Exception as e:
            print(f"Failed to setup Groggy environment: {e}")
            traceback.print_exc()
            return False
    
    def discover_methods(self, obj_name: str, obj_instance: Any) -> List[Dict]:
        """Discover all methods on an object instance"""
        methods = []
        obj_type = type(obj_instance).__name__
        
        print(f"\nDiscovering methods for {obj_name} (type: {obj_type})")
        
        # Get all attributes safely
        try:
            all_attrs = dir(obj_instance)
        except Exception as e:
            print(f"  Error getting attributes with dir(): {e}")
            # Fallback to class attributes
            try:
                all_attrs = [attr for attr in dir(type(obj_instance)) if not attr.startswith('__')]
            except Exception as e2:
                print(f"  Fallback also failed: {e2}")
                self.discovery_errors.append({
                    'object': obj_name,
                    'error_type': 'dir_failed',
                    'error': str(e)
                })
                return methods
        method_attrs = []
        
        for attr_name in all_attrs:
            # Skip private attributes
            if attr_name.startswith('_'):
                continue
                
            try:
                attr = getattr(obj_instance, attr_name)
                
                # Check if it's callable
                if callable(attr):
                    method_attrs.append(attr_name)
                    
                    # Get method signature
                    try:
                        sig = inspect.signature(attr)
                        signature_str = str(sig)
                    except Exception:
                        signature_str = "signature_unavailable"
                    
                    # Try to determine return type through inspection
                    return_type = "unknown"
                    try:
                        if hasattr(attr, '__annotations__'):
                            annotations = getattr(attr, '__annotations__')
                            if 'return' in annotations:
                                return_type = str(annotations['return'])
                    except Exception:
                        pass
                    
                    method_info = {
                        'name': attr_name,
                        'signature': signature_str,
                        'return_type': return_type,
                        'source_object': obj_name,
                        'source_type': obj_type,
                        'is_property': isinstance(getattr(type(obj_instance), attr_name, None), property),
                        'doc': getattr(attr, '__doc__', None)
                    }
                    
                    methods.append(method_info)
                    
            except Exception as e:
                self.discovery_errors.append({
                    'object': obj_name,
                    'attribute': attr_name,
                    'error': str(e)
                })
        
        print(f"  Found {len(method_attrs)} callable methods: {method_attrs[:10]}{'...' if len(method_attrs) > 10 else ''}")
        return methods
    
    def analyze_delegation_patterns(self):
        """Analyze delegation patterns between objects"""
        print("\nAnalyzing delegation patterns...")
        
        for obj_name, obj_instance in self.object_instances.items():
            obj_methods = self.discovered_methods[obj_name]
            
            for method_info in obj_methods:
                method_name = method_info['name']
                
                # Check if this method exists on other objects
                for other_obj_name, other_methods in self.discovered_methods.items():
                    if other_obj_name == obj_name:
                        continue
                        
                    for other_method in other_methods:
                        if other_method['name'] == method_name:
                            self.delegation_chains[method_name].append({
                                'from': obj_name,
                                'to': other_obj_name,
                                'method': method_name
                            })
    
    def test_method_calls(self):
        """Test method calls to understand return types"""
        print("\nTesting method calls to understand return types...")
        
        for obj_name, obj_instance in self.object_instances.items():
            methods = self.discovered_methods[obj_name]
            
            for method_info in methods:
                method_name = method_info['name']
                
                # Skip certain methods that might be destructive
                skip_methods = {
                    'clear', 'reset', 'close', 'destroy', 'delete', 'remove',
                    'pop', 'shutdown', 'exit', 'quit'
                }
                
                if method_name in skip_methods:
                    continue
                
                try:
                    method = getattr(obj_instance, method_name)
                    
                    # Try calling methods with no parameters
                    sig = inspect.signature(method)
                    params = sig.parameters
                    
                    # Only test methods that can be called with no arguments
                    required_params = [p for p in params.values() 
                                     if p.default == inspect.Parameter.empty and p.name != 'self']
                    
                    if len(required_params) == 0:
                        try:
                            result = method()
                            actual_return_type = type(result).__name__
                            method_info['actual_return_type'] = actual_return_type
                            method_info['test_successful'] = True
                            
                            # If result is a Groggy object, note it for graph construction
                            if hasattr(result, '__module__') and 'groggy' in str(result.__module__):
                                method_info['returns_groggy_object'] = True
                                method_info['returned_object_type'] = actual_return_type
                            
                        except Exception as e:
                            method_info['test_error'] = str(e)
                            method_info['test_successful'] = False
                    else:
                        method_info['requires_parameters'] = [p.name for p in required_params]
                        
                except Exception as e:
                    method_info['discovery_error'] = str(e)
    
    def run_full_discovery(self) -> Dict:
        """Run the complete discovery process"""
        print("=" * 60)
        print("STARTING META API DISCOVERY FOR GROGGY")
        print("=" * 60)
        
        # Setup environment
        if not self.setup_groggy_environment():
            return {"error": "Failed to setup Groggy environment"}
        
        # Discover methods for each object
        for obj_name, obj_instance in self.object_instances.items():
            methods = self.discover_methods(obj_name, obj_instance)
            self.discovered_methods[obj_name] = methods
        
        # Test method calls
        self.test_method_calls()
        
        # Analyze delegation patterns
        self.analyze_delegation_patterns()
        
        # Create summary
        summary = self.create_discovery_summary()
        
        print("\n" + "=" * 60)
        print("DISCOVERY COMPLETE")
        print("=" * 60)
        
        return summary
    
    def build_api_meta_graph(self) -> Dict:
        """Build a Groggy graph where objects are nodes and methods are edges between objects"""
        print("\n" + "=" * 60)
        print("BUILDING META-GRAPH OF THE API")
        print("=" * 60)
        
        try:
            import groggy
            
            # Create nodes for each discovered object type (only objects with methods)
            meta_nodes_data = {
                'node_id': [],
                'object_name': [],
                'object_type': [], 
                'method_count': [],
                'module': []
            }
            
            node_id = 1
            object_name_to_node_id = {}
            
            # First pass: create nodes for all discovered objects
            for obj_name, methods in self.discovered_methods.items():
                obj_type = type(self.object_instances[obj_name]).__name__
                
                meta_nodes_data['node_id'].append(node_id)
                meta_nodes_data['object_name'].append(obj_name)
                meta_nodes_data['object_type'].append(obj_type)
                meta_nodes_data['method_count'].append(len(methods))
                
                # Try to get module info
                try:
                    module_name = type(self.object_instances[obj_name]).__module__
                    meta_nodes_data['module'].append(module_name)
                except:
                    meta_nodes_data['module'].append('unknown')
                
                object_name_to_node_id[obj_name] = node_id
                node_id += 1
            
            # Second pass: add nodes for return types that match discovered object types
            return_type_to_object_name = {}
            for obj_name, methods in self.discovered_methods.items():
                obj_type = type(self.object_instances[obj_name]).__name__
                return_type_to_object_name[obj_type] = obj_name
            
            # Create edges only when methods connect objects we actually discovered
            meta_edges_data = {
                'edge_id': [],
                'source': [],
                'target': [],
                'method_name': [],
                'method_signature': [],
                'return_type': []
            }
            
            edge_id = 1
            
            for obj_name, methods in self.discovered_methods.items():
                source_node_id = object_name_to_node_id[obj_name]
                
                for method_info in methods:
                    method_name = method_info['name']
                    return_type = method_info.get('actual_return_type', method_info.get('return_type', 'unknown'))
                    
                    # Only create edges to objects we actually have nodes for
                    target_obj_name = return_type_to_object_name.get(return_type)
                    if target_obj_name and target_obj_name in object_name_to_node_id:
                        target_node_id = object_name_to_node_id[target_obj_name]
                        
                        # Add edge for this method connecting two objects
                        meta_edges_data['edge_id'].append(edge_id)
                        meta_edges_data['source'].append(source_node_id)
                        meta_edges_data['target'].append(target_node_id)
                        meta_edges_data['method_name'].append(method_name)
                        # Clean signature for CSV compatibility
                        signature = method_info['signature'].replace('"', "'").replace('\n', ' ')
                        meta_edges_data['method_signature'].append(signature)
                        meta_edges_data['return_type'].append(return_type)
                        
                        edge_id += 1
            
            # Create the meta-graph
            meta_nodes_table = groggy.NodesTable.from_dict(meta_nodes_data)
            meta_edges_table = groggy.EdgesTable.from_dict(meta_edges_data)
            meta_graph_table = groggy.GraphTable(meta_nodes_table, meta_edges_table)
            meta_graph = meta_graph_table.to_graph()
            
            total_methods = sum(len(methods) for methods in self.discovered_methods.values())
            
            print(f"✓ Created API Meta-Graph:")
            print(f"  - Nodes: {meta_graph.node_count()} (objects with methods)")
            print(f"  - Edges: {meta_graph.edge_count()} (methods connecting objects)")
            print(f"  - Total methods discovered: {total_methods}")
            print(f"  - Methods with object return types: {meta_graph.edge_count()}")
            
            # Save the meta-graph
            meta_graph_table.save_bundle("./groggy_api_meta_graph")
            print("✓ Saved API meta-graph to: ./groggy_api_meta_graph")
            
            return {
                'meta_graph': meta_graph,
                'meta_graph_table': meta_graph_table,
                'object_name_to_node_id': object_name_to_node_id,
                'return_type_to_object_name': return_type_to_object_name,
                'nodes_data': meta_nodes_data,
                'edges_data': meta_edges_data
            }
            
        except Exception as e:
            print(f"❌ Failed to build meta-graph: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def create_discovery_summary(self) -> Dict:
        """Create a comprehensive summary of the discovery"""
        total_methods = sum(len(methods) for methods in self.discovered_methods.values())
        
        # Build the meta-graph
        meta_graph_info = self.build_api_meta_graph()
        
        summary = {
            'discovery_metadata': {
                'total_objects': len(self.object_instances),
                'total_methods': total_methods,
                'total_errors': len(self.discovery_errors),
                'objects_discovered': list(self.object_instances.keys())
            },
            'objects': {},
            'delegation_patterns': dict(self.delegation_chains),
            'errors': self.discovery_errors,
            'meta_graph': meta_graph_info
        }
        
        # Add detailed method info for each object
        for obj_name, methods in self.discovered_methods.items():
            summary['objects'][obj_name] = {
                'type': type(self.object_instances[obj_name]).__name__,
                'method_count': len(methods),
                'methods': methods
            }
        
        # Print summary
        print(f"\nDISCOVERY SUMMARY:")
        print(f"  Objects: {len(self.object_instances)}")
        print(f"  Total Methods: {total_methods}")
        print(f"  Errors: {len(self.discovery_errors)}")
        
        for obj_name, obj_data in summary['objects'].items():
            print(f"  {obj_name}: {obj_data['method_count']} methods")
        
        return summary


def main():
    """Main entry point for API discovery"""
    discovery = APIMethodDiscovery()
    summary = discovery.run_full_discovery()
    
    # Save results
    output_file = "api_discovery_results.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return summary


if __name__ == "__main__":
    main()