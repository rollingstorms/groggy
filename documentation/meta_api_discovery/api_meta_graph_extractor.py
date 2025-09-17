#!/usr/bin/env python3
"""
API Meta-Graph Extractor

A streamlined script that combines API discovery, type inference, and graph creation
to extract the essential meta-graph structure:
- Nodes: Object types and return types (with all discovered info)
- Edges: Methods (source=object_type, target=return_type, with method info)

This replaces the complex 3-step process with a single focused extraction.
"""

import groggy
import inspect
import re
import json
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict


class APIMetaGraphExtractor:
    """Extract API structure directly into a meta-graph"""
    
    def __init__(self):
        self.graph = groggy.Graph()
        self.discovered_types = set()
        self.method_relationships = []
        self.type_info = {}
        self.method_info = {}
        
        # Type inference patterns
        self.type_patterns = {
            r'\b(?:dictionary|dict|Dict)\b': 'dict',
            r'\b(?:list|List|array|Array)\b': 'list',
            r'\b(?:string|str|String)\b': 'str',
            r'\b(?:integer|int|Integer)\b': 'int',
            r'\b(?:number|float|Float|numeric)\b': 'float',
            r'\b(?:boolean|bool|Boolean)\b': 'bool',
            r'\bBaseTable\b': 'BaseTable',
            r'\bNodesTable\b': 'NodesTable',
            r'\bEdgesTable\b': 'EdgesTable',
            r'\bGraphTable\b': 'GraphTable',
            r'\bGraph\b': 'Graph',
            r'\bSubgraph\b': 'Subgraph',
            r'\bBaseArray\b': 'BaseArray',
            r'\bGraphArray\b': 'GraphArray',
            r'\bMatrix\b': 'Matrix',
            r'\bComponentsArray\b': 'ComponentsArray',
        }
    
    def extract_return_type_from_signature(self, method) -> str:
        """Extract return type from method signature"""
        try:
            sig = inspect.signature(method)
            if sig.return_annotation and sig.return_annotation != inspect.Signature.empty:
                annotation = str(sig.return_annotation)
                # Clean up the annotation
                annotation = annotation.replace('<class \'', '').replace('\'>', '')
                annotation = annotation.replace('groggy.', '')  # Remove module prefix
                
                # Extract just the class name if it's a full path
                if '.' in annotation:
                    annotation = annotation.split('.')[-1]
                    
                # Add to discovered types
                self.discovered_types.add(annotation)
                return annotation
        except:
            pass
        return 'Unknown'
    
    def extract_return_type_from_docstring(self, docstring: str) -> str:
        """Extract return type from docstring"""
        if not docstring:
            return 'Unknown'
        
        # First, check for known type patterns in the docstring (most reliable)
        for type_pattern, canonical_type in self.type_patterns.items():
            # Only match whole words to avoid partial matches
            word_pattern = r'\b' + type_pattern.strip(r'\b') + r'\b'
            if re.search(word_pattern, docstring, re.IGNORECASE):
                self.discovered_types.add(canonical_type)
                return canonical_type
        
        # If no known types found, try specific return type patterns
        return_patterns = [
            r'Returns?:?\s*([A-Z][A-Za-z0-9_]*)',  # Capitalized types after "Returns:"
            r'-> ([A-Z][A-Za-z0-9_]*)',  # Arrow notation with capitalized types
            r'return[s]?\s+([A-Z][A-Za-z0-9_]*)',  # "return Type" with capitalized types
            r'PyResult<([A-Za-z_][A-Za-z0-9_]*)>',  # Rust-style PyResult
            r'Py([A-Z][A-Za-z0-9_]*)',  # Python wrapper types (PyGraph, PyTable, etc.)
        ]
        
        # Common words to reject (expanded list)
        reject_words = {
            'the', 'and', 'this', 'that', 'with', 'from', 'for', 'new', 'old',
            'ing', 'ion', 'tion', 'thon', 'report', 'empty', 'New', 'a', 's', 'L'
        }
        
        for pattern in return_patterns:
            match = re.search(pattern, docstring, re.IGNORECASE)
            if match:
                return_type = match.group(1)
                
                # Filter out obviously wrong types
                if (len(return_type) <= 2 or 
                    return_type.lower() in reject_words or
                    not return_type[0].isupper() or
                    not return_type.isalpha()):  # Only alphabetic characters
                    continue
                
                # Clean up common prefixes
                original_type = return_type
                return_type = return_type.replace('Py', '')  # Remove Py prefix
                
                # Skip if it becomes too short after cleanup or matches reject list
                if len(return_type) <= 2 or return_type.lower() in reject_words:
                    continue
                
                # Validate it looks like a proper class name
                if return_type[0].isupper() and len(return_type) >= 3 and return_type.isalnum():
                    self.discovered_types.add(return_type)
                    return return_type
        
        return 'Unknown'
    
    def extract_parameters_info(self, method) -> List[Dict]:
        """Extract parameter information from method"""
        params = []
        try:
            sig = inspect.signature(method)
            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'py']:  # Skip self and py parameters
                    continue
                
                param_info = {
                    'name': param_name,
                    'type': 'Any',
                    'default': None,
                    'required': param.default == inspect.Parameter.empty
                }
                
                # Get type annotation
                if param.annotation != inspect.Parameter.empty:
                    param_info['type'] = str(param.annotation).replace('<class \'', '').replace('\'>', '')
                
                # Get default value
                if param.default != inspect.Parameter.empty:
                    param_info['default'] = str(param.default)
                
                params.append(param_info)
        except:
            pass
        
        return params
    
    def discover_object_methods(self, obj, obj_name: str) -> List[Dict]:
        """Discover all methods for an object and their relationships"""
        methods = []
        
        for method_name in dir(obj):
            if method_name.startswith('_'):  # Skip private methods
                continue
            
            try:
                method = getattr(obj, method_name)
                if not callable(method):
                    continue
                
                # Get method info
                docstring = inspect.getdoc(method) or ""
                
                # Extract return type
                return_type = self.extract_return_type_from_signature(method)
                if return_type == 'Unknown':
                    return_type = self.extract_return_type_from_docstring(docstring)
                
                # Extract parameters
                parameters = self.extract_parameters_info(method)
                
                method_info = {
                    'object_type': obj_name,
                    'method_name': method_name,
                    'return_type': return_type,
                    'parameters': parameters,
                    'docstring': docstring,
                    'signature': str(inspect.signature(method)) if hasattr(inspect, 'signature') else 'Unknown'
                }
                
                methods.append(method_info)
                
                # Track discovered types
                self.discovered_types.add(obj_name)
                self.discovered_types.add(return_type)
                
            except Exception as e:
                continue
        
        return methods
    
    def discover_groggy_api(self) -> Dict:
        """Discover the complete Groggy API structure"""
        print("Discovering Groggy API structure...")
        
        # Core objects to analyze
        core_objects = {
            'Graph': groggy.Graph(),
        }
        
        # Try to discover additional objects
        try:
            # Create a graph with some data to get access to more object types
            g = groggy.Graph()
            nodes_data = [{'id': 'test1'}, {'id': 'test2'}, {'id': 'test3'}]
            g.add_nodes(nodes_data, uid_key='id')
            g.add_edges([('test1', 'test2'), ('test2', 'test3')], uid_key='id')
            
            # Try to get table objects
            try:
                table = g.table()
                core_objects['BaseTable'] = table
            except:
                pass
            
            try:
                nodes_table = g.nodes()
                core_objects['NodesTable'] = nodes_table
            except:
                pass
            
            try:
                edges_table = g.edges()
                core_objects['EdgesTable'] = edges_table
            except:
                pass
            
            try:
                matrix = g.to_matrix()
                core_objects['Matrix'] = matrix
            except:
                pass
            
            # Try to create Subgraph objects using various methods
            try:
                # Method 1: Try slicing nodes (if supported)
                subgraph = g.nodes[:2]  # First 2 nodes
                if subgraph is not None:
                    core_objects['Subgraph'] = subgraph
                    print(f"    Found Subgraph via slicing: {type(subgraph)}")
            except Exception as slice_e:
                print(f"    Slicing method failed: {slice_e}")
            
            # Method 2: Try filter methods if Subgraph not found yet
            if 'Subgraph' not in core_objects:
                try:
                    # Try filtering nodes
                    subgraph = g.filter_nodes(lambda node: True)  # Filter that accepts all
                    if subgraph is not None:
                        core_objects['Subgraph'] = subgraph
                        print(f"    Found Subgraph via filter_nodes: {type(subgraph)}")
                except Exception as filter_e:
                    print(f"    filter_nodes method failed: {filter_e}")
            
            # Method 3: Try connected_components if Subgraph not found yet
            if 'Subgraph' not in core_objects:
                try:
                    components = g.connected_components()
                    if components and len(components) > 0:
                        subgraph = components[0]  # First component
                        if subgraph is not None:
                            core_objects['Subgraph'] = subgraph
                            print(f"    Found Subgraph via connected_components: {type(subgraph)}")
                except Exception as comp_e:
                    print(f"    connected_components method failed: {comp_e}")
            
            # Method 4: Try subgraph method if exists
            if 'Subgraph' not in core_objects:
                try:
                    if hasattr(g, 'subgraph'):
                        subgraph = g.subgraph(nodes=['test1', 'test2'])
                        if subgraph is not None:
                            core_objects['Subgraph'] = subgraph
                            print(f"    Found Subgraph via subgraph method: {type(subgraph)}")
                except Exception as sub_e:
                    print(f"    subgraph method failed: {sub_e}")
            
            # If we still don't have a Subgraph, list available methods on the graph
            if 'Subgraph' not in core_objects:
                print(f"    Could not create Subgraph object. Available Graph methods:")
                graph_methods = [m for m in dir(g) if not m.startswith('_') and callable(getattr(g, m))]
                subgraph_related = [m for m in graph_methods if 'subgraph' in m.lower() or 'component' in m.lower() or 'filter' in m.lower()]
                print(f"      Potentially relevant methods: {subgraph_related}")
            
        except Exception as e:
            print(f"Note: Could not create test objects: {e}")
        
        # Discover methods for each object
        all_methods = []
        for obj_name, obj in core_objects.items():
            print(f"  Analyzing {obj_name}...")
            methods = self.discover_object_methods(obj, obj_name)
            all_methods.extend(methods)
            
            # Store object info
            self.type_info[obj_name] = {
                'type_name': obj_name,
                'class_name': obj.__class__.__name__,
                'module': obj.__class__.__module__,
                'methods_count': len(methods),
                'docstring': inspect.getdoc(obj.__class__) or ""
            }
        
        print(f"Discovered {len(all_methods)} methods across {len(core_objects)} object types")
        print(f"Found {len(self.discovered_types)} unique types")
        
        return {
            'methods': all_methods,
            'types': list(self.discovered_types),
            'type_info': self.type_info,
            'discovery_stats': {
                'total_methods': len(all_methods),
                'total_types': len(self.discovered_types),
                'core_objects': len(core_objects)
            }
        }
    
    def build_meta_graph(self, discovery_data: Dict) -> groggy.Graph:
        """Build the meta-graph from discovery data"""
        print("\nBuilding meta-graph...")
        
        # Prepare nodes data (all discovered types)
        nodes_data = []
        for type_name in discovery_data['types']:
            node_data = {
                'type_name': type_name,
                'category': 'core' if type_name in self.type_info else 'inferred',
                'methods_count': 0,  # Will be updated
                'description': ''
            }
            
            if type_name in self.type_info:
                node_data.update(self.type_info[type_name])
            
            nodes_data.append(node_data)
        
        # Add nodes to graph
        print(f"  Adding {len(nodes_data)} type nodes...")
        self.graph.add_nodes(nodes_data, uid_key='type_name')
        
        # Prepare edges data (method relationships)
        edges_data = []
        for method in discovery_data['methods']:
            # Extract parameter names for requires_parameters
            requires_parameters = [p['name'] for p in method['parameters'] if p['required']]
            
            # Build enhanced signature with type hints
            param_strs = []
            parameter_types = {}
            for param in method['parameters']:
                param_str = param['name']
                if param['type'] != 'Any':
                    param_str += f": {param['type']}"
                    parameter_types[param['name']] = param['type']
                if not param['required'] and param['default']:
                    param_str += f"={param['default']}"
                param_strs.append(param_str)
            
            enhanced_signature = f"({', '.join(param_strs)})"
            
            edge_data = {
                # Core method identification (matching your example format)
                'object_type': method['object_type'],      # Source type string
                'return_type': method['return_type'],      # Target type string
                'name': method['method_name'],             # Method name (matching your format)
                
                # Method signature details (matching your example)
                'signature': method['signature'],
                'enhanced_signature': enhanced_signature,
                'parameter_types': json.dumps(parameter_types),  # Convert dict to JSON string
                'requires_parameters': json.dumps(requires_parameters),  # Convert list to JSON string for Groggy
                
                # Source object info (matching your format)
                'source_object': method['object_type'],
                'source_type': method['object_type'],
                
                # Additional method metadata
                'is_property': False,  # We can enhance this later by checking if it's a property
                'doc': method['docstring'],
                
                # Our additional useful metadata
                'parameters_count': len(method['parameters']),
                'method_full_name': f"{method['object_type']}.{method['method_name']}",
                'relationship': f"{method['object_type']} -> {method['return_type']}",
                
                # CRITICAL: Preserve object_type and return_type as edge attributes
                # Since source/target parameters consume these fields for edge creation,
                # we need to duplicate them with different names to preserve as attributes
                'edge_object_type': method['object_type'],
                'edge_return_type': method['return_type']
            }
            edges_data.append(edge_data)
        
        # Add edges to graph using our new enhanced add_edges
        print(f"  Adding {len(edges_data)} method edges...")
        
        # CRITICAL FIX: Preserve object_type and return_type as edge attributes
        # Since source/target parameters consume these fields, we need to ensure they're preserved
        for edge_data in edges_data:
            # Make sure object_type and return_type are preserved as edge attributes
            if 'object_type' not in edge_data:
                edge_data['object_type'] = edge_data['edge_object_type']
            if 'return_type' not in edge_data:
                edge_data['return_type'] = edge_data['edge_return_type']
        
        self.graph.add_edges(edges_data, 
                           uid_key='type_name',
                           source='edge_object_type',  # Use our backup fields for source/target
                           target='edge_return_type')
        
        print(f"Meta-graph created: {self.graph.node_count()} nodes, {self.graph.edge_count()} edges")
        return self.graph
    
    def extract_complete_meta_graph(self) -> Tuple[groggy.Graph, Dict]:
        """Complete extraction process"""
        print("API Meta-Graph Extraction")
        print("=" * 50)
        
        # Step 1: Discover API
        discovery_data = self.discover_groggy_api()
        
        # Step 2: Build meta-graph
        meta_graph = self.build_meta_graph(discovery_data)
        
        # Step 3: Create summary
        summary = {
            'meta_graph_stats': {
                'nodes': meta_graph.node_count(),
                'edges': meta_graph.edge_count(),
                'types_discovered': len(discovery_data['types'])
            },
            'discovery_data': discovery_data,
            'type_info': self.type_info
        }
        
        return meta_graph, summary
    
    def save_results(self, meta_graph: groggy.Graph, summary: Dict, 
                    graph_file: str = "api_meta_graph.json",
                    summary_file: str = "api_meta_summary.json"):
        """Save the meta-graph and summary"""
        print(f"\nSaving results...")
        
        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Summary saved to: {summary_file}")
        
        # For the graph, we'll save the structure
        graph_data = {
            'nodes': [],
            'edges': [],
            'stats': summary['meta_graph_stats']
        }
        
        # Extract node data
        try:
            nodes_table = meta_graph.nodes()
            # We'd need to iterate through the table to extract data
            # For now, just save the stats
        except:
            pass
        
        with open(graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"  Graph structure saved to: {graph_file}")


def main():
    """Main execution"""
    extractor = APIMetaGraphExtractor()
    
    # Extract the complete meta-graph
    meta_graph, summary = extractor.extract_complete_meta_graph()
    
    # Print summary
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Meta-graph nodes: {summary['meta_graph_stats']['nodes']}")
    print(f"Meta-graph edges: {summary['meta_graph_stats']['edges']}")
    print(f"Types discovered: {summary['meta_graph_stats']['types_discovered']}")
    print(f"Methods analyzed: {summary['discovery_data']['discovery_stats']['total_methods']}")
    
    # Show some example types
    print(f"\nDiscovered types: {', '.join(list(summary['discovery_data']['types'])[:10])}...")
    
    # Save results
    extractor.save_results(meta_graph, summary)
    
    print("\n✅ API Meta-Graph extraction complete!")
    print("The graph represents the API structure where:")
    print("  • Nodes = Object types and return types")
    print("  • Edges = Methods (object_type -> return_type)")
    print("  • Edge attributes = Method signatures, parameters, docs")
    
    return meta_graph, summary


if __name__ == "__main__":
    meta_graph, summary = main()
