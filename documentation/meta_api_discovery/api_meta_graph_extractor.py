#!/usr/bin/env python3
"""
API Meta-Graph Extractor

A streamlined script that combines API discovery, type inference, and graph creation
to extract the essential meta-graph structure:
- Nodes: Object types and return types (with all discovered info)
- Edges: Methods (source=object_type, target=return_type, with method info)

This replaces the complex 3-step process with a single focused extraction.
"""
import sys
sys.path.append("/Users/michaelroth/Documents/Code/groggy/")
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
        
        # Type inference patterns (expanded with all discovered Groggy types)
        self.type_patterns = {
            r'\b(?:dictionary|dict|Dict)\b': 'dict',
            r'\b(?:list|List|array|Array)\b': 'list',
            r'\b(?:string|str|String)\b': 'str',
            r'\b(?:integer|int|Integer)\b': 'int',
            r'\b(?:number|float|Float|numeric)\b': 'float',
            r'\b(?:boolean|bool|Boolean)\b': 'bool',
            
            # Core Groggy objects
            r'\bGraph\b': 'Graph',
            r'\bSubgraph\b': 'Subgraph',
            
            # Table types
            r'\bBaseTable\b': 'BaseTable',
            r'\bNodesTable\b': 'NodesTable',
            r'\bEdgesTable\b': 'EdgesTable',
            r'\bGraphTable\b': 'GraphTable',
            
            # Array types
            r'\bBaseArray\b': 'BaseArray',
            r'\bNodesArray\b': 'NodesArray',
            r'\bEdgesArray\b': 'EdgesArray',
            r'\bSubgraphArray\b': 'SubgraphArray',
            r'\bTableArray\b': 'TableArray',
            r'\bComponentsArray\b': 'ComponentsArray',
            r'\bMetaNodeArray\b': 'MetaNodeArray',
            r'\bStatsArray\b': 'StatsArray',
            r'\bNumArray\b': 'NumArray',
            
            # Matrix types
            r'\bMatrix\b': 'Matrix',
            r'\bGraphMatrix\b': 'GraphMatrix',
            
            # Accessor types
            r'\bNodesAccessor\b': 'NodesAccessor',
            r'\bEdgesAccessor\b': 'EdgesAccessor',
            
            # Specialized types
            r'\bMetaNode\b': 'MetaNode',
            r'\bDisplayConfig\b': 'DisplayConfig',
            r'\bTableFormatter\b': 'TableFormatter',
            r'\bAggregationResult\b': 'AggregationResult',
            r'\bGroupedAggregationResult\b': 'GroupedAggregationResult',
            r'\bHistoricalView\b': 'HistoricalView',
            r'\bHistoryStatistics\b': 'HistoryStatistics',
            r'\bBranchInfo\b': 'BranchInfo',
            r'\bCommit\b': 'Commit',
            
            # Generic fallbacks
            r'\bGraphArray\b': 'GraphArray',
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
        
        try:
            # Get methods safely, handling special objects that might have issues with dir()
            if hasattr(obj, '__class__'):
                # For complex objects, get methods from the class instead of the instance
                method_names = [name for name in dir(obj.__class__) if not name.startswith('_')]
            else:
                method_names = [name for name in dir(obj) if not name.startswith('_')]
        except Exception as e:
            print(f"    Warning: Could not get methods for {obj_name}: {e}")
            return methods
        
        for method_name in method_names:
            if method_name.startswith('_'):  # Skip private methods
                continue
            
            try:
                # Get method safely
                if hasattr(obj, method_name):
                    method = getattr(obj, method_name)
                else:
                    # Try getting from class if not available on instance
                    method = getattr(obj.__class__, method_name)
                
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
                # Skip methods that can't be analyzed
                print(f"    Warning: Could not analyze {obj_name}.{method_name}: {e}")
                continue
        
        return methods
    
    def discover_groggy_api(self) -> Dict:
        """Discover the complete Groggy API structure"""
        print("Discovering Groggy API structure...")
        
        # Core objects to analyze
        core_objects = {
            'Graph': groggy.Graph(),
        }
        
        # Discover objects available directly from the groggy module
        print("  Checking groggy module for available types...")
        groggy_module_objects = {}
        for attr_name in dir(groggy):
            if (not attr_name.startswith('_') and 
                not attr_name.islower() and 
                attr_name not in ['Graph']):  # Skip Graph as we already have it
                try:
                    attr_obj = getattr(groggy, attr_name)
                    if inspect.isclass(attr_obj):
                        # Try to create an instance (for classes that can be instantiated)
                        try:
                            # Some classes might need parameters, try empty constructor first
                            instance = attr_obj()
                            groggy_module_objects[attr_name] = instance
                            print(f"    Created {attr_name} instance")
                        except:
                            # If we can't instantiate, at least record the class
                            print(f"    Found {attr_name} class (could not instantiate)")
                            pass
                    elif callable(attr_obj):
                        # It's a function, skip for now
                        pass
                    else:
                        # It's some other object, try to use it
                        groggy_module_objects[attr_name] = attr_obj
                        print(f"    Added {attr_name} object")
                except Exception as e:
                    print(f"    Could not access {attr_name}: {e}")
        
        # Add successfully created module objects to core_objects
        core_objects.update(groggy_module_objects)
        
        # Try to discover additional objects through API usage
        try:
            # Create a graph with some data to get access to more object types
            g = groggy.Graph()
            
            # Add diverse data to enable various object creations
            for i in range(5):
                g.add_node(i, name=f'node_{i}', age=20+i*5, category='A' if i%2==0 else 'B', value=i*10)
            
            for i in range(4):
                g.add_edge(i, i+1, weight=1.0+i*0.5, edge_type='connection')
            
            print("  Creating objects through Graph API usage...")
            
            # === ACCESSOR OBJECTS ===
            try:
                nodes_accessor = g.nodes
                if 'NodesAccessor' not in core_objects:
                    core_objects['NodesAccessor'] = nodes_accessor
                    print(f"    Found NodesAccessor: {type(nodes_accessor)}")
            except: pass
            
            try:
                edges_accessor = g.edges
                if 'EdgesAccessor' not in core_objects:
                    core_objects['EdgesAccessor'] = edges_accessor
                    print(f"    Found EdgesAccessor: {type(edges_accessor)}")
            except: pass
            
            # === TABLE OBJECTS ===
            try:
                graph_table = g.table()
                if 'GraphTable' not in core_objects:
                    core_objects['GraphTable'] = graph_table
                    print(f"    Found GraphTable: {type(graph_table)}")
            except: pass
            
            try:
                nodes_table = g.nodes.table()
                if 'NodesTable' not in core_objects:
                    core_objects['NodesTable'] = nodes_table
                    print(f"    Found NodesTable: {type(nodes_table)}")
            except: pass
            
            try:
                edges_table = g.edges.table()
                if 'EdgesTable' not in core_objects:
                    core_objects['EdgesTable'] = edges_table
                    print(f"    Found EdgesTable: {type(edges_table)}")
            except: pass
            
            # === ARRAY OBJECTS ===
            try:
                nodes_array = g.nodes.array()
                if 'NodesArray' not in core_objects:
                    core_objects['NodesArray'] = nodes_array
                    print(f"    Found NodesArray: {type(nodes_array)}")
            except: pass
            
            try:
                num_array = g.nodes.ids()
                if 'NumArray' not in core_objects:
                    core_objects['NumArray'] = num_array
                    print(f"    Found NumArray: {type(num_array)}")
            except: pass
            
            try:
                subgraph_array = g.nodes.group_by('category')
                if 'SubgraphArray' not in core_objects:
                    core_objects['SubgraphArray'] = subgraph_array
                    print(f"    Found SubgraphArray: {type(subgraph_array)}")
                
                # From SubgraphArray, get TableArray
                try:
                    table_array = subgraph_array.table()
                    if 'TableArray' not in core_objects:
                        core_objects['TableArray'] = table_array
                        print(f"    Found TableArray: {type(table_array)}")
                except: pass
                
                # Get individual Subgraph from SubgraphArray
                try:
                    if len(subgraph_array) > 0:
                        subgraph = subgraph_array[0]
                        if 'Subgraph' not in core_objects:
                            core_objects['Subgraph'] = subgraph
                            print(f"    Found Subgraph: {type(subgraph)}")
                except: pass
                
            except: pass
            
            # === MATRIX OBJECTS ===
            try:
                matrix = g.to_matrix()
                if 'GraphMatrix' not in core_objects:
                    core_objects['GraphMatrix'] = matrix
                    print(f"    Found GraphMatrix: {type(matrix)}")
            except: pass
            
            try:
                nodes_matrix = g.nodes.matrix()
                # This might be the same as GraphMatrix, but check
                matrix_type = type(nodes_matrix).__name__
                if matrix_type not in core_objects:
                    core_objects[matrix_type] = nodes_matrix
                    print(f"    Found {matrix_type}: {type(nodes_matrix)}")
            except: pass
            
            # === COMPONENTS ===
            try:
                components = g.connected_components()
                if 'ComponentsArray' not in core_objects:
                    core_objects['ComponentsArray'] = components
                    print(f"    Found ComponentsArray: {type(components)}")
            except: pass
            
            # === BASE TYPES (if we can create them) ===
            try:
                # Try to get a BaseTable from conversion
                base_table = nodes_table.base_table() if 'nodes_table' in locals() else None
                if base_table and 'BaseTable' not in core_objects:
                    core_objects['BaseTable'] = base_table
                    print(f"    Found BaseTable: {type(base_table)}")
            except: pass
            
        except Exception as e:
            print(f"Note: Could not create test objects: {e}")
        
        print(f"  Total objects discovered: {len(core_objects)}")
        object_names = list(core_objects.keys())
        print(f"  Object types: {', '.join(object_names)}")
        
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
