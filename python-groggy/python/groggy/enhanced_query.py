"""
Enhanced Query Engine - Logical operators and complex expressions

This module provides an enhanced query interface that supports logical operators
(AND, OR, NOT) and complex expressions by applying filters in Python rather than
relying on the Rust core for logical combinations.
"""

import re
from typing import Union, List, Set
from . import Graph
from .query_parser import parse_node_query, parse_edge_query
from ._groggy import NodeFilter, EdgeFilter

class EnhancedQueryEngine:
    """Enhanced query engine with support for logical operators."""
    
    def __init__(self, graph: Graph):
        self.graph = graph
    
    def filter_nodes(self, query: str):
        """Filter nodes using enhanced query syntax with logical operators."""
        return self._evaluate_node_query(query)
    
    def filter_edges(self, query: str):
        """Filter edges using enhanced query syntax with logical operators."""
        return self._evaluate_edge_query(query)
    
    def _evaluate_node_query(self, query: str):
        """Evaluate a node query with logical operators."""
        # Normalize the query
        query = query.strip()
        normalized_query = query.upper()
        
        # Check for logical operators
        if ' AND ' in normalized_query:
            return self._evaluate_and_node_query(query)
        elif ' OR ' in normalized_query:
            return self._evaluate_or_node_query(query)
        elif normalized_query.startswith('NOT '):
            return self._evaluate_not_node_query(query)
        else:
            # Single filter - use the original parsing
            node_filter = parse_node_query(query)
            return self.graph.filter_nodes(node_filter)
    
    def _evaluate_edge_query(self, query: str):
        """Evaluate an edge query with logical operators."""
        # Normalize the query
        query = query.strip()
        normalized_query = query.upper()
        
        # Check for logical operators
        if ' AND ' in normalized_query:
            return self._evaluate_and_edge_query(query)
        elif ' OR ' in normalized_query:
            return self._evaluate_or_edge_query(query)
        elif normalized_query.startswith('NOT '):
            return self._evaluate_not_edge_query(query)
        else:
            # Single filter - use the original parsing
            edge_filter = parse_edge_query(query)
            return self.graph.filter_edges(edge_filter)
    
    def _evaluate_and_node_query(self, query: str):
        """Evaluate an AND node query by intersecting results."""
        parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise ValueError(f"Complex AND expressions with more than 2 parts not yet supported: {query}")
        
        # Get results for each part
        left_result = self._evaluate_node_query(parts[0].strip())
        right_result = self._evaluate_node_query(parts[1].strip())
        
        # Intersect the node sets
        left_nodes = set(left_result.nodes)
        right_nodes = set(right_result.nodes)
        intersection_nodes = left_nodes & right_nodes
        
        # Create a subgraph with the intersection
        return self._create_node_subgraph(intersection_nodes, f"AND({parts[0].strip()}, {parts[1].strip()})")
    
    def _evaluate_or_node_query(self, query: str):
        """Evaluate an OR node query by unioning results."""
        parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise ValueError(f"Complex OR expressions with more than 2 parts not yet supported: {query}")
        
        # Get results for each part
        left_result = self._evaluate_node_query(parts[0].strip())
        right_result = self._evaluate_node_query(parts[1].strip())
        
        # Union the node sets
        left_nodes = set(left_result.nodes)
        right_nodes = set(right_result.nodes)
        union_nodes = left_nodes | right_nodes
        
        # Create a subgraph with the union
        return self._create_node_subgraph(union_nodes, f"OR({parts[0].strip()}, {parts[1].strip()})")
    
    def _evaluate_not_node_query(self, query: str):
        """Evaluate a NOT node query by complementing results."""
        # Remove 'NOT ' from the beginning
        inner_query = re.sub(r'^NOT\s+', '', query, flags=re.IGNORECASE).strip()
        
        # Get all nodes
        all_nodes = set(self.graph.node_ids)
        
        # Get nodes that match the inner query
        inner_result = self._evaluate_node_query(inner_query)
        matching_nodes = set(inner_result.nodes)
        
        # Complement: all nodes that don't match
        complement_nodes = all_nodes - matching_nodes
        
        # Create a subgraph with the complement
        return self._create_node_subgraph(complement_nodes, f"NOT({inner_query})")
    
    def _evaluate_and_edge_query(self, query: str):
        """Evaluate an AND edge query by intersecting results."""
        parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise ValueError(f"Complex AND expressions with more than 2 parts not yet supported: {query}")
        
        # Get results for each part
        left_result = self._evaluate_edge_query(parts[0].strip())
        right_result = self._evaluate_edge_query(parts[1].strip())
        
        # Intersect the edge sets
        left_edges = set(left_result if isinstance(left_result, list) else left_result.edges)
        right_edges = set(right_result if isinstance(right_result, list) else right_result.edges)
        intersection_edges = left_edges & right_edges
        
        return list(intersection_edges)
    
    def _evaluate_or_edge_query(self, query: str):
        """Evaluate an OR edge query by unioning results."""
        parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise ValueError(f"Complex OR expressions with more than 2 parts not yet supported: {query}")
        
        # Get results for each part
        left_result = self._evaluate_edge_query(parts[0].strip())
        right_result = self._evaluate_edge_query(parts[1].strip())
        
        # Union the edge sets
        left_edges = set(left_result if isinstance(left_result, list) else left_result.edges)
        right_edges = set(right_result if isinstance(right_result, list) else right_result.edges)
        union_edges = left_edges | right_edges
        
        return list(union_edges)
    
    def _evaluate_not_edge_query(self, query: str):
        """Evaluate a NOT edge query by complementing results."""
        # Remove 'NOT ' from the beginning
        inner_query = re.sub(r'^NOT\s+', '', query, flags=re.IGNORECASE).strip()
        
        # Get all edges
        all_edges = set(self.graph.edge_ids)
        
        # Get edges that match the inner query
        inner_result = self._evaluate_edge_query(inner_query)
        matching_edges = set(inner_result if isinstance(inner_result, list) else inner_result.edges)
        
        # Complement: all edges that don't match
        complement_edges = all_edges - matching_edges
        
        return list(complement_edges)
    
    def _create_node_subgraph(self, node_ids: Set[int], description: str):
        """Create a subgraph from a set of node IDs."""
        
        # Find all edges that connect nodes in the set
        connecting_edges = []
        for edge_id in self.graph.edge_ids:
            edge_view = self.graph.edges[edge_id]
            source = edge_view.source
            target = edge_view.target
            if source in node_ids and target in node_ids:
                connecting_edges.append(edge_id)
        
        return EnhancedSubgraph(node_ids, connecting_edges, description)

class EnhancedSubgraph:
    """A subgraph-like object that combines multiple filters with logical operators."""
    
    def __init__(self, nodes, edges, subgraph_type):
        self.nodes = list(nodes)
        self.edges = edges
        self.subgraph_type = subgraph_type
    
    def __repr__(self):
        return f"EnhancedSubgraph({len(self.nodes)} nodes, {len(self.edges)} edges, {self.subgraph_type})"

# Convenience functions for backward compatibility
def enhanced_filter_nodes(graph: Graph, query: str):
    """Filter nodes using enhanced query syntax."""
    engine = EnhancedQueryEngine(graph)
    return engine.filter_nodes(query)

def enhanced_filter_edges(graph: Graph, query: str):
    """Filter edges using enhanced query syntax."""
    engine = EnhancedQueryEngine(graph)
    return engine.filter_edges(query)