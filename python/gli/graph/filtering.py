"""
Enhanced filtering capabilities for GLI graphs.

This module provides advanced filtering functionality including:
- String-based query language
- Subgraph creation from filters
- Query compilation and execution
"""

import re
import ast
import operator
from typing import Any, Dict, List, Union, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Graph
    from .subgraph import Subgraph

class QueryCompiler:
    """Compiles string-based queries into executable filter functions."""
    
    # Supported operators (order matters - longer operators first!)
    OPERATORS = {
        '>=': operator.ge,
        '<=': operator.le,
        '!=': operator.ne,
        '==': operator.eq,
        '>': operator.gt,
        '<': operator.lt,
        'not in': lambda x, y: x not in y,
        'in': lambda x, y: x in y,
    }
    
    # Logical operators
    LOGICAL_OPS = {
        'AND': operator.and_,
        'OR': operator.or_,
        'NOT': operator.not_,
    }
    
    @classmethod
    def compile_node_query(cls, query_str: str) -> Callable[[str, Dict[str, Any]], bool]:
        """
        Compile a string query into a function for node filtering.
        
        Args:
            query_str: Query string like "role == 'Manager'" or "salary > 80000"
            
        Returns:
            Function that takes (node_id, attributes) and returns bool
        """
        def filter_func(node_id: str, attributes: Dict[str, Any]) -> bool:
            return cls._evaluate_query(query_str, attributes)
        
        return filter_func
    
    @classmethod
    def compile_edge_query(cls, query_str: str) -> Callable[[str, str, str, Dict[str, Any]], bool]:
        """
        Compile a string query into a function for edge filtering.
        
        Args:
            query_str: Query string for edge attributes
            
        Returns:
            Function that takes (edge_id, source, target, attributes) and returns bool
        """
        def filter_func(edge_id: str, source: str, target: str, attributes: Dict[str, Any]) -> bool:
            # Add edge-specific variables to the context
            extended_attrs = attributes.copy()
            extended_attrs.update({
                'edge_id': edge_id,
                'source': source,
                'target': target
            })
            return cls._evaluate_query(query_str, extended_attrs)
        
        return filter_func
    
    @classmethod
    def _evaluate_query(cls, query_str: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a query string against a context dictionary.
        
        Args:
            query_str: The query string to evaluate
            context: Dictionary of available variables
            
        Returns:
            Boolean result of the query
        """
        try:
            # Handle logical operators (AND, OR, NOT)
            query_str = cls._preprocess_logical_operators(query_str)
            
            # Split by logical operators while preserving them
            parts = cls._split_logical_expression(query_str)
            
            if len(parts) == 1:
                # Simple expression
                return cls._evaluate_simple_expression(parts[0], context)
            else:
                # Complex expression with logical operators
                return cls._evaluate_logical_expression(parts, context)
                
        except Exception as e:
            raise ValueError(f"Failed to evaluate query '{query_str}': {e}")
    
    @classmethod
    def _preprocess_logical_operators(cls, query_str: str) -> str:
        """Normalize logical operators."""
        # Convert to uppercase and handle different formats
        query_str = re.sub(r'\band\b', 'AND', query_str, flags=re.IGNORECASE)
        query_str = re.sub(r'\bor\b', 'OR', query_str, flags=re.IGNORECASE)
        query_str = re.sub(r'\bnot\b', 'NOT', query_str, flags=re.IGNORECASE)
        return query_str
    
    @classmethod
    def _split_logical_expression(cls, query_str: str) -> List[str]:
        """Split expression by logical operators while preserving them."""
        # For now, handle simple AND/OR cases
        if ' AND ' in query_str:
            parts = []
            for part in query_str.split(' AND '):
                parts.append(part.strip())
                parts.append('AND')
            return parts[:-1]  # Remove last AND
        elif ' OR ' in query_str:
            parts = []
            for part in query_str.split(' OR '):
                parts.append(part.strip())
                parts.append('OR')
            return parts[:-1]  # Remove last OR
        else:
            return [query_str.strip()]
    
    @classmethod
    def _evaluate_logical_expression(cls, parts: List[str], context: Dict[str, Any]) -> bool:
        """Evaluate a logical expression with multiple parts."""
        result = cls._evaluate_simple_expression(parts[0], context)
        
        i = 1
        while i < len(parts):
            operator_str = parts[i]
            if i + 1 < len(parts):
                right_result = cls._evaluate_simple_expression(parts[i + 1], context)
                
                if operator_str == 'AND':
                    result = result and right_result
                elif operator_str == 'OR':
                    result = result or right_result
                
                i += 2
            else:
                break
        
        return result
    
    @classmethod
    def _evaluate_simple_expression(cls, expr: str, context: Dict[str, Any]) -> bool:
        """Evaluate a simple comparison expression."""
        expr = expr.strip()
        
        # Find the operator
        for op_str, op_func in cls.OPERATORS.items():
            if op_str in expr:
                parts = expr.split(op_str, 1)
                if len(parts) == 2:
                    left_part = parts[0].strip()
                    right_part = parts[1].strip()
                    
                    # Get the left value (attribute name)
                    if left_part in context:
                        left_value = context[left_part]
                    else:
                        raise ValueError(f"Unknown attribute: {left_part}")
                    
                    # Parse the right value (literal)
                    right_value = cls._parse_literal(right_part)
                    
                    # Handle type coercion for numeric comparisons
                    if op_str in ['>', '>=', '<', '<=']:
                        try:
                            # Try to convert both to same numeric type
                            if isinstance(left_value, (int, float)) and isinstance(right_value, str):
                                try:
                                    right_value = float(right_value) if '.' in right_value else int(right_value)
                                except ValueError:
                                    pass
                            elif isinstance(right_value, (int, float)) and isinstance(left_value, str):
                                try:
                                    left_value = float(left_value) if '.' in left_value else int(left_value)
                                except ValueError:
                                    pass
                        except (ValueError, TypeError):
                            pass
                    
                    # Apply the operator
                    try:
                        return op_func(left_value, right_value)
                    except TypeError:
                        # If comparison fails, return False for safety
                        return False
        
        raise ValueError(f"No valid operator found in expression: {expr}")
    
    @classmethod
    def _parse_literal(cls, value_str: str) -> Any:
        """Parse a literal value from a string."""
        value_str = value_str.strip()
        
        # Remove quotes for strings
        if (value_str.startswith("'") and value_str.endswith("'")) or \
           (value_str.startswith('"') and value_str.endswith('"')):
            return value_str[1:-1]
        
        # Try to parse as boolean first (before numbers)
        if value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False
        
        # Try to parse as None
        if value_str.lower() == 'none':
            return None
        
        # Try to parse as number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # Return as string if all else fails
        return value_str


class SubgraphCreator:
    """Creates subgraphs from filtering operations."""
    
    @staticmethod
    def create_node_subgraph(source_graph: 'Graph', node_ids: List[str], filter_criteria) -> 'Subgraph':
        """
        Create a subgraph containing only the specified nodes and edges between them.
        
        Args:
            source_graph: The original graph
            node_ids: List of node IDs to include in the subgraph
            filter_criteria: The filter criteria used to create this subgraph
            
        Returns:
            New Subgraph instance containing the filtered data
        """
        from .subgraph import Subgraph  # Import here to avoid circular imports
        
        # Create new subgraph with same backend and directed setting
        subgraph = Subgraph(
            parent_graph=source_graph, 
            filter_criteria=str(filter_criteria),
            metadata={"filter_type": "nodes"},
            backend=source_graph.backend,
            directed=source_graph.directed
        )
        
        # Add all specified nodes
        for node_id in node_ids:
            node = source_graph.get_node(node_id)
            if node:
                subgraph.add_node(node_id, **node.attributes)
        
        # Add edges between the included nodes
        for edge_id in source_graph.edges:
            edge = source_graph.edges[edge_id]
            if edge.source in node_ids and edge.target in node_ids:
                subgraph.add_edge(edge.source, edge.target, edge_id=edge_id, **edge.attributes)
        
        return subgraph
    
    @staticmethod
    def create_edge_subgraph(source_graph: 'Graph', edge_ids: List[str], filter_criteria) -> 'Subgraph':
        """
        Create a subgraph containing only the specified edges and their connected nodes.
        
        Args:
            source_graph: The original graph
            edge_ids: List of edge IDs to include in the subgraph
            filter_criteria: The filter criteria used to create this subgraph
            
        Returns:
            New Subgraph instance containing the filtered data
        """
        from .subgraph import Subgraph  # Import here to avoid circular imports
        
        # Create new subgraph with same backend and directed setting
        subgraph = Subgraph(
            parent_graph=source_graph, 
            filter_criteria=str(filter_criteria),
            metadata={"filter_type": "edges"},
            backend=source_graph.backend,
            directed=source_graph.directed
        )
        
        # Handle both edge IDs and tuples (for Rust backend compatibility)
        actual_edge_ids = []
        for item in edge_ids:
            if isinstance(item, tuple):
                # Handle tuple format (source, target) from Rust backend
                source, target = item
                # Find the actual edge ID
                for edge_id in source_graph.edges:
                    edge = source_graph.edges[edge_id]
                    if edge.source == source and edge.target == target:
                        actual_edge_ids.append(edge_id)
                        break
            else:
                # Already an edge ID
                actual_edge_ids.append(item)
        
        # Collect all nodes that are connected by the specified edges
        connected_nodes = set()
        for edge_id in actual_edge_ids:
            if edge_id in source_graph.edges:
                edge = source_graph.edges[edge_id]
                connected_nodes.add(edge.source)
                connected_nodes.add(edge.target)
        
        # Add all connected nodes
        for node_id in connected_nodes:
            node = source_graph.get_node(node_id)
            if node:
                subgraph.add_node(node_id, **node.attributes)
        
        # Add the specified edges
        for edge_id in actual_edge_ids:
            if edge_id in source_graph.edges:
                edge = source_graph.edges[edge_id]
                subgraph.add_edge(edge.source, edge.target, edge_id=edge_id, **edge.attributes)
        
        return subgraph


# Enhanced filter functions that can be added to the Graph class
def enhanced_filter_nodes(
    graph: 'Graph',
    filter_criteria: Union[Callable[[str, Dict[str, Any]], bool], Dict[str, Any], str],
    return_graph: bool = False
) -> Union[List[str], 'Subgraph']:
    """
    Enhanced node filtering with optional subgraph creation.
    
    Args:
        graph: The graph to filter
        filter_criteria: Filter function, attribute dict, or query string
        return_graph: If True, return a Subgraph object; if False, return node IDs
        
    Returns:
        List of node IDs or Subgraph
    """
    # Determine the filter function
    if isinstance(filter_criteria, str):
        # String query - compile to function
        filter_func = QueryCompiler.compile_node_query(filter_criteria)
    elif callable(filter_criteria):
        # Already a function
        filter_func = filter_criteria
    elif isinstance(filter_criteria, dict):
        # Dictionary-based filtering - convert to function
        def dict_filter(node_id: str, attributes: Dict[str, Any]) -> bool:
            for attr, value in filter_criteria.items():
                if attr not in attributes or attributes[attr] != value:
                    return False
            return True
        filter_func = dict_filter
    else:
        raise ValueError("filter_criteria must be a callable, dict, or string")
    
    # Apply the filter using the original logic
    if graph.use_rust:
        result = []
        for node_id in graph.nodes:
            node = graph.get_node(node_id)
            if node and filter_func(node_id, node.attributes):
                result.append(node_id)
    else:
        effective_nodes, _, _ = graph._get_effective_data()
        result = []
        for node_id, node in effective_nodes.items():
            if filter_func(node_id, node.attributes):
                result.append(node_id)
    
    # Return based on requested format
    if return_graph:
        return SubgraphCreator.create_node_subgraph(graph, result, filter_criteria)
    else:
        return result


def enhanced_filter_edges(
    graph: 'Graph',
    filter_criteria: Union[Callable[[str, str, str, Dict[str, Any]], bool], Dict[str, Any], str],
    return_graph: bool = False
) -> Union[List[str], 'Subgraph']:
    """
    Enhanced edge filtering with optional subgraph creation.
    
    Args:
        graph: The graph to filter
        filter_criteria: Filter function, attribute dict, or query string
        return_graph: If True, return a Subgraph object; if False, return edge IDs
        
    Returns:
        List of edge IDs or Subgraph
    """
    # Determine the filter function
    if isinstance(filter_criteria, str):
        # String query - compile to function
        filter_func = QueryCompiler.compile_edge_query(filter_criteria)
    elif callable(filter_criteria):
        # Already a function
        filter_func = filter_criteria
    elif isinstance(filter_criteria, dict):
        # Dictionary-based filtering - convert to function
        def dict_filter(edge_id: str, source: str, target: str, attributes: Dict[str, Any]) -> bool:
            for attr, value in filter_criteria.items():
                if attr not in attributes or attributes[attr] != value:
                    return False
            return True
        filter_func = dict_filter
    else:
        raise ValueError("filter_criteria must be a callable, dict, or string")
    
    # Apply the filter using the original logic
    if graph.use_rust:
        result = []
        for edge_id in graph.edges:
            edge = graph.edges[edge_id]
            if edge and filter_func(edge_id, edge.source, edge.target, edge.attributes):
                result.append(edge_id)
    else:
        _, effective_edges, _ = graph._get_effective_data()
        result = []
        for edge_id, edge in effective_edges.items():
            if filter_func(edge_id, edge.source, edge.target, edge.attributes):
                result.append(edge_id)
    
    # Return based on requested format
    if return_graph:
        return SubgraphCreator.create_edge_subgraph(graph, result, filter_criteria)
    else:
        return result
