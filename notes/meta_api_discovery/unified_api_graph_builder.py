#!/usr/bin/env python3
"""
Unified API Graph Builder

Combines api_discovery_results.json and meta_api_test_results.json into a comprehensive
groggy graph where every method is represented, with generic nodes for return types.

This script ensures ALL methods are tested and discoverable, not just connected ones.
"""

import json
import groggy as gr
from pathlib import Path
from typing import Dict, Any, Set, List, Tuple


class UnifiedAPIGraphBuilder:
    def __init__(self):
        self.graph = gr.Graph()
        self.node_registry = {}  # name -> node_id mapping
        self.method_registry = {}  # method_name -> edge_id mapping
        self.return_type_nodes = {}  # return_type -> node_id mapping
        
    def get_or_create_node(self, name: str, node_type: str = "object", **attrs) -> int:
        """Get existing node or create new one with metadata."""
        if name not in self.node_registry:
            node_id = self.graph.add_node()
            self.graph.set_node_attr(node_id, 'name', name)
            self.graph.set_node_attr(node_id, 'type', node_type)
            
            # Add any additional attributes
            for key, value in attrs.items():
                self.graph.set_node_attr(node_id, key, str(value))
                
            self.node_registry[name] = node_id
            print(f"Created {node_type} node: {name}")
        
        return self.node_registry[name]
    
    def create_return_type_node(self, return_type: str) -> int:
        """Create generic node for return types like 'self', 'int', 'str', etc."""
        if return_type not in self.return_type_nodes:
            # Normalize return type name
            normalized_name = f"ReturnType_{return_type.replace(' ', '_').replace('<', '').replace('>', '')}"
            node_id = self.get_or_create_node(
                normalized_name, 
                "return_type",
                original_type=return_type,
                is_generic=True
            )
            self.return_type_nodes[return_type] = node_id
            
        return self.return_type_nodes[return_type]
    
    def add_method_edge(self, source_obj: str, method_name: str, return_type: str = None, **attrs) -> int:
        """Add method as edge between source object and return type."""
        # Get or create source node
        source_id = self.get_or_create_node(source_obj, "api_object")
        
        # Determine target node
        if return_type and return_type.strip():
            clean_return_type = return_type.strip()
            
            if clean_return_type == "self" or clean_return_type == source_obj:
                target_id = source_id  # Self-reference
            else:
                # Check if return type is a known API object first
                if clean_return_type in self.node_registry:
                    target_id = self.node_registry[clean_return_type]
                else:
                    # Create proper return type node (not generic)
                    target_id = self.create_return_type_node(clean_return_type)
        else:
            # Unknown return type
            target_id = self.create_return_type_node("unknown")
        
        # Create edge for method
        edge_id = self.graph.add_edge(source_id, target_id)
        self.graph.set_edge_attr(edge_id, 'method_name', method_name)
        self.graph.set_edge_attr(edge_id, 'source_object', source_obj)
        self.graph.set_edge_attr(edge_id, 'return_type', return_type or "unknown")
        
        # Add all additional attributes
        for key, value in attrs.items():
            self.graph.set_edge_attr(edge_id, key, str(value))
        
        # Track in registry
        method_key = f"{source_obj}.{method_name}"
        self.method_registry[method_key] = edge_id
        
        return edge_id
    
    def load_discovery_data(self, discovery_file: str) -> Dict[str, Any]:
        """Load API discovery results (preferring enhanced version with parameter types)."""
        # Try enhanced version first
        enhanced_file = discovery_file.replace('.json', '_enhanced.json')
        if Path(enhanced_file).exists():
            with open(enhanced_file, 'r') as f:
                data = json.load(f)
            print(f"Loaded enhanced discovery data from {enhanced_file}")
            return data
        
        # Fall back to original
        if not Path(discovery_file).exists():
            print(f"Warning: Discovery file not found: {discovery_file}")
            return {}
            
        with open(discovery_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded discovery data from {discovery_file}")
        return data
    
    def load_test_results(self, test_file: str) -> Dict[str, Any]:
        """Load test results data."""
        if not Path(test_file).exists():
            print(f"Warning: Test results file not found: {test_file}")
            return {}
            
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded test results from {test_file}")
        return data
    
    def process_discovery_data(self, discovery_data: Dict[str, Any]):
        """Process discovery data and add to graph."""
        print("Processing discovery data...")
        
        # Process the objects section
        objects_data = discovery_data.get('objects', {})
        
        for obj_name, obj_data in objects_data.items():
            if isinstance(obj_data, dict):
                # Create object node
                self.get_or_create_node(obj_name, "api_object", **{
                    'discovered': True,
                    'method_count': str(obj_data.get('method_count', 0)),
                    'object_type': obj_data.get('type', 'unknown')
                })
                
                # Process methods (they are in a list format)
                methods = obj_data.get('methods', [])
                for method_info in methods:
                    if isinstance(method_info, dict):
                        method_name = method_info.get('name', 'unknown')
                        return_type = method_info.get('return_type', 'unknown')
                        
                        # Get enhanced signature and parameter types if available
                        enhanced_signature = method_info.get('enhanced_signature', method_info.get('signature', ''))
                        parameter_types = method_info.get('parameter_types', {})
                        
                        edge_id = self.add_method_edge(
                            obj_name,
                            method_name,
                            return_type,
                            data_source='discovery',
                            doc=method_info.get('doc', '')[:200] if method_info.get('doc') else '',
                            signature=method_info.get('signature', ''),
                            enhanced_signature=enhanced_signature,
                            is_property=str(method_info.get('is_property', False)),
                            requires_parameters=str(method_info.get('requires_parameters', [])),
                            discovered=True
                        )
                        
                        # Add parameter type information if available
                        if parameter_types:
                            self.graph.set_edge_attr(edge_id, 'has_parameter_types', True)
                            self.graph.set_edge_attr(edge_id, 'parameter_types_json', json.dumps(parameter_types))
                            for param_name, param_type in parameter_types.items():
                                self.graph.set_edge_attr(edge_id, f'param_{param_name}_type', param_type)
                        else:
                            self.graph.set_edge_attr(edge_id, 'has_parameter_types', False)
    
    def process_test_results(self, test_data: Dict[str, Any]):
        """Process test results and add to existing edges or create new ones."""
        print("Processing test results...")
        
        # Process the test_results section (it's a list)
        test_results = test_data.get('test_results', [])
        
        for test_result in test_results:
            if isinstance(test_result, dict):
                obj_name = test_result.get('object', 'Unknown')
                method_name = test_result.get('method', 'unknown')
                
                method_key = f"{obj_name}.{method_name}"
                
                # Find existing edge or create new one
                if method_key in self.method_registry:
                    edge_id = self.method_registry[method_key]
                else:
                    # Create new edge for tested but undiscovered method
                    edge_id = self.add_method_edge(
                        obj_name,
                        method_name,
                        test_result.get('return_type', 'unknown'),
                        data_source='test_only'
                    )
                
                # Add test result attributes
                self.graph.set_edge_attr(edge_id, 'tested', True)
                self.graph.set_edge_attr(edge_id, 'test_status', test_result.get('status', 'unknown'))
                self.graph.set_edge_attr(edge_id, 'test_requires_params', str(test_result.get('requires_params', False)))
                
                # Add error if present
                if test_result.get('error'):
                    self.graph.set_edge_attr(edge_id, 'test_error', test_result.get('error', ''))
                
                # Add result if present
                if test_result.get('result') is not None:
                    self.graph.set_edge_attr(edge_id, 'test_result', str(test_result.get('result', '')))
                
                # IMPORTANT: Update return type if test provides actual type info
                test_return_type = test_result.get('return_type')
                if test_return_type and test_return_type not in [None, 'null', 'unknown', '']:
                    # Update the edge's return type and create proper target node
                    self.update_method_return_type(edge_id, test_return_type, 'test_execution')
                    print(f"Found real return type: {obj_name}.{method_name} -> {test_return_type}")
                
                # Also try to infer return type from test result string for successful tests
                elif test_result.get('result') and test_result.get('status') in ['success', 'success_with_params']:
                    inferred_type = self.infer_return_type_from_result(test_result.get('result', ''))
                    if inferred_type and inferred_type != 'unknown':
                        self.update_method_return_type(edge_id, inferred_type, 'inferred_from_result')
                        print(f"Inferred return type: {obj_name}.{method_name} -> {inferred_type}")
    
    def update_method_return_type(self, edge_id: int, return_type: str, source: str):
        """Update method edge to point to proper return type node."""
        # Create or get the return type node
        if return_type in self.node_registry:
            target_id = self.node_registry[return_type]
        else:
            target_id = self.create_return_type_node(return_type)
        
        # Update edge attributes
        self.graph.set_edge_attr(edge_id, 'return_type', return_type)
        self.graph.set_edge_attr(edge_id, 'return_type_source', source)
        
        # Update edge target to point to the proper return type node
        # Note: In groggy, we can't change edge targets directly, so we store the target info as attribute
        self.graph.set_edge_attr(edge_id, 'return_type_node_id', str(target_id))
    
    def infer_return_type_from_result(self, result_str: str) -> str:
        """Infer return type from test result string."""
        result_str = str(result_str).strip()
        
        # Common patterns in groggy results
        if result_str.startswith('BaseTable['):
            return 'BaseTable'
        elif result_str.startswith('NodesTable['):
            return 'NodesTable'
        elif result_str.startswith('EdgesTable['):
            return 'EdgesTable'
        elif result_str.startswith('GraphTable['):
            return 'GraphTable'
        elif result_str.startswith('BaseArray[') or result_str.startswith('BaseArray('):
            return 'BaseArray'
        elif result_str.startswith('GraphArray('):
            return 'GraphArray'
        elif result_str.startswith('ComponentsArray('):
            return 'ComponentsArray'
        elif result_str.startswith('Graph('):
            return 'Graph'
        elif result_str.startswith('Subgraph('):
            return 'Subgraph'
        elif result_str.startswith('Matrix('):
            return 'Matrix'
        elif result_str.startswith('[') and result_str.endswith(']'):
            return 'list'
        elif result_str in ['True', 'False']:
            return 'bool'
        elif result_str.replace('.', '').replace('-', '').isdigit():
            return 'int' if '.' not in result_str else 'float'
        elif result_str.startswith('"') and result_str.endswith('"'):
            return 'str'
        elif result_str.startswith("'") and result_str.endswith("'"):
            return 'str'
        elif '(' in result_str and ')' in result_str and ',' in result_str:
            return 'tuple'
        else:
            return 'unknown'
    
    def export_graph_bundle(self, output_prefix: str = "unified_api_graph"):
        """Export the unified graph as table bundle."""
        print("Exporting graph bundle...")
        
        # Get tables
        nodes_table = self.graph.nodes.table()
        edges_table = self.graph.edges.table()
        
        # Export as CSV for inspection
        nodes_table.to_csv(f"{output_prefix}/{output_prefix}_nodes.csv")
        edges_table.to_csv(f"{output_prefix}/{output_prefix}_edges.csv")
        
        # Commit graph state
        commit_id = self.graph.commit(
            "Unified API discovery and test results",
            "unified_api_builder"
        )
        
        print(f"Graph committed: {commit_id}")
        print(f"Exported {output_prefix}/{output_prefix}_nodes.csv and {output_prefix}/{output_prefix}_edges.csv")
        
        return {
            'nodes_table': nodes_table,
            'edges_table': edges_table,
            'commit_id': commit_id,
            'graph': self.graph
        }
    
    def print_statistics(self):
        """Print comprehensive statistics about the unified graph."""
        print(f"\n=== Unified API Graph Statistics ===")
        print(f"Total nodes: {len(self.node_registry)}")
        print(f"Total edges (methods): {len(self.method_registry)}")
        
        # Use the CSV export to analyze since direct table access is complex
        # Export and read back the CSV for analysis
        import pandas as pd
        
        try:
            # Read the exported CSV files
            nodes_df = pd.read_csv('unified_api_graph/unified_api_graph_nodes.csv')
            edges_df = pd.read_csv('unified_api_graph/unified_api_graph_edges.csv')
            
            # Count node types
            node_types = nodes_df['type'].value_counts().to_dict()
            print(f"\nNode types:")
            for node_type, count in sorted(node_types.items()):
                print(f"  {node_type}: {count}")
            
            # Count data sources
            data_sources = edges_df['data_source'].value_counts().to_dict()
            print(f"\nData sources:")
            for source, count in sorted(data_sources.items()):
                print(f"  {source}: {count}")
            
            # Count test coverage
            tested_count = edges_df['tested'].value_counts().get(True, 0)
            discovered_count = edges_df['discovered'].value_counts().get(True, 0)
            
            print(f"\nMethod coverage:")
            print(f"  Discovered methods: {discovered_count}")
            print(f"  Tested methods: {tested_count}")
            print(f"  Total unique methods: {len(edges_df)}")
            
            # Show return type distribution
            return_types = edges_df['return_type'].value_counts().head(10).to_dict()
            print(f"\nTop return types:")
            for ret_type, count in return_types.items():
                print(f"  {ret_type}: {count}")
            
            # Return type source breakdown
            if 'return_type_source' in edges_df.columns:
                type_sources = edges_df['return_type_source'].value_counts().to_dict()
                print(f"\nReturn type sources:")
                for source, count in sorted(type_sources.items()):
                    print(f"  {source}: {count}")
            
            # Count methods with known vs unknown return types
            known_types = len(edges_df[edges_df['return_type'] != 'unknown'])
            unknown_types = len(edges_df[edges_df['return_type'] == 'unknown'])
            print(f"\nReturn type coverage:")
            print(f"  Known return types: {known_types}")
            print(f"  Unknown return types: {unknown_types}")
            print(f"  Coverage: {known_types/(known_types+unknown_types)*100:.1f}%")
            
            # Parameter type coverage
            if 'has_parameter_types' in edges_df.columns:
                param_typed = len(edges_df[edges_df['has_parameter_types'] == True])  # Python boolean True
                param_untyped = len(edges_df[edges_df['has_parameter_types'] == False])  # Python boolean False
                total_param_methods = param_typed + param_untyped
                
                print(f"\nParameter type coverage:")
                print(f"  Methods with parameter types: {param_typed}")
                print(f"  Methods without parameter types: {param_untyped}")
                if total_param_methods > 0:
                    print(f"  Coverage: {param_typed/total_param_methods*100:.1f}%")
                else:
                    print(f"  Coverage: No parameter type data found")
                
            # Test status breakdown
            if 'test_status' in edges_df.columns:
                test_statuses = edges_df['test_status'].value_counts().to_dict()
                print(f"\nTest status breakdown:")
                for status, count in sorted(test_statuses.items()):
                    print(f"  {status}: {count}")
                    
        except Exception as e:
            print(f"Error reading CSV files for analysis: {e}")
            print(f"Basic counts - Nodes: {len(self.node_registry)}, Methods: {len(self.method_registry)}")


def main():
    """Main execution function."""
    print("Building Unified API Graph...")
    
    builder = UnifiedAPIGraphBuilder()
    
    # File paths
    discovery_file = "api_discovery_results.json"
    test_file = "meta_api_test_results.json"
    
    # Load data
    discovery_data = builder.load_discovery_data(discovery_file)
    test_data = builder.load_test_results(test_file)
    
    # Process data
    if discovery_data:
        builder.process_discovery_data(discovery_data)
    
    if test_data:
        builder.process_test_results(test_data)
    
    # Export results
    result = builder.export_graph_bundle("unified_api_graph")
    
    # Print statistics
    builder.print_statistics()
    
    print(f"\n=== Export Complete ===")
    print(f"Files created:")
    print(f"  - unified_api_graph_nodes.csv")
    print(f"  - unified_api_graph_edges.csv")
    print(f"Graph object available for further analysis")
    
    return result


if __name__ == "__main__":
    result = main()