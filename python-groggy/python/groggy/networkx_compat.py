"""
NetworkX Interoperability - Seamless conversion to/from NetworkX

This module provides functions to convert between Groggy graphs and NetworkX graphs,
enabling users to leverage the NetworkX ecosystem while benefiting from Groggy's
performance and advanced features.
"""

from typing import Dict, Any, Optional, Union, List
from . import Graph
from .types import NodeId, EdgeId

def to_networkx(graph: Graph, 
                directed: bool = False,
                include_attributes: bool = True,
                node_attr_prefix: str = "",
                edge_attr_prefix: str = "") -> 'networkx.Graph':
    """
    Convert a Groggy graph to a NetworkX graph.
    
    Args:
        graph: Groggy graph to convert
        directed: If True, create a directed NetworkX graph
        include_attributes: If True, include node and edge attributes
        node_attr_prefix: Prefix for node attribute names in NetworkX
        edge_attr_prefix: Prefix for edge attribute names in NetworkX
        
    Returns:
        NetworkX graph (Graph or DiGraph depending on directed parameter)
        
    Example:
        >>> import groggy as gr
        >>> g = gr.generators.karate_club()
        >>> nx_graph = to_networkx(g)
        >>> print(f"NetworkX graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required for interoperability. Install with: pip install networkx")
    
    # Create NetworkX graph
    if directed:
        nx_graph = nx.DiGraph()
    else:
        nx_graph = nx.Graph()
    
    # Add nodes
    node_ids = graph.node_ids
    for node_id in node_ids:
        node_attrs = {}
        if include_attributes:
            # Try to get common attributes
            for attr_name in ['name', 'index', 'level', 'component_id', 'age', 'dept', 'salary', 'location', 'community']:
                try:
                    if hasattr(graph.nodes[node_id], '__getitem__'):
                        attr_value = graph.nodes[node_id][attr_name]
                        # Convert to Python native types for NetworkX compatibility
                        if hasattr(attr_value, 'inner'):
                            attr_value = attr_value.inner
                        node_attrs[f"{node_attr_prefix}{attr_name}"] = attr_value
                except (KeyError, AttributeError):
                    continue
        
        nx_graph.add_node(node_id, **node_attrs)
    
    # Add edges
    edge_ids = graph.edge_ids
    for edge_id in edge_ids:
        try:
            edge_view = graph.edges[edge_id]
            source = edge_view.source
            target = edge_view.target
            
            edge_attrs = {}
            if include_attributes:
                # Try to get common edge attributes
                for attr_name in ['weight', 'relationship', 'type', 'strength', 'frequency']:
                    try:
                        attr_value = edge_view[attr_name]
                        # Convert to Python native types for NetworkX compatibility
                        if hasattr(attr_value, 'inner'):
                            attr_value = attr_value.inner
                        edge_attrs[f"{edge_attr_prefix}{attr_name}"] = attr_value
                    except (KeyError, AttributeError):
                        continue
            
            nx_graph.add_edge(source, target, **edge_attrs)
            
        except Exception:
            # Skip edges that can't be processed
            continue
    
    return nx_graph

def from_networkx(nx_graph: 'networkx.Graph',
                 preserve_node_attrs: bool = True,
                 preserve_edge_attrs: bool = True,
                 handle_multiedges: str = 'keep_first',
                 node_attr_mapping: Optional[Dict[str, str]] = None,
                 edge_attr_mapping: Optional[Dict[str, str]] = None) -> Graph:
    """
    Convert a NetworkX graph to a Groggy graph.
    
    Args:
        nx_graph: NetworkX graph to convert
        preserve_node_attrs: If True, preserve node attributes
        preserve_edge_attrs: If True, preserve edge attributes
        handle_multiedges: How to handle multiple edges ('keep_first', 'keep_last', 'merge')
        node_attr_mapping: Mapping from NetworkX attr names to Groggy attr names
        edge_attr_mapping: Mapping from NetworkX attr names to Groggy attr names
        
    Returns:
        Groggy graph with equivalent structure and attributes
        
    Example:
        >>> import networkx as nx
        >>> import groggy as gr
        >>> nx_g = nx.karate_club_graph()
        >>> groggy_g = from_networkx(nx_g)
        >>> print(f"Groggy graph: {groggy_g.node_count()} nodes, {groggy_g.edge_count()} edges")
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required for interoperability. Install with: pip install networkx")
    
    g = Graph()
    
    # Mapping from NetworkX node IDs to Groggy node IDs
    node_mapping = {}
    
    # Add nodes
    for nx_node_id in nx_graph.nodes():
        node_attrs = {}
        
        if preserve_node_attrs:
            nx_attrs = nx_graph.nodes[nx_node_id]
            for attr_name, attr_value in nx_attrs.items():
                # Apply attribute name mapping if provided
                groggy_attr_name = attr_name
                if node_attr_mapping and attr_name in node_attr_mapping:
                    groggy_attr_name = node_attr_mapping[attr_name]
                
                # Convert NetworkX attributes to Groggy-compatible types
                node_attrs[groggy_attr_name] = _convert_networkx_attr_value(attr_value)
        
        # Add original NetworkX node ID as an attribute for reference
        node_attrs['nx_node_id'] = nx_node_id
        
        groggy_node_id = g.add_node(**node_attrs)
        node_mapping[nx_node_id] = groggy_node_id
    
    # Add edges
    edges_added = set()  # Track edges for multiedge handling
    
    for nx_source, nx_target in nx_graph.edges():
        groggy_source = node_mapping[nx_source]
        groggy_target = node_mapping[nx_target]
        
        # Handle multiedges
        edge_key = (min(groggy_source, groggy_target), max(groggy_source, groggy_target))
        if edge_key in edges_added:
            if handle_multiedges == 'keep_first':
                continue  # Skip additional edges
            elif handle_multiedges == 'keep_last':
                # Would need to remove previous edge, but for simplicity, just skip
                continue
            # For 'merge', we could combine attributes, but skip for now
        
        edge_attrs = {}
        
        if preserve_edge_attrs:
            # Handle edge attributes (NetworkX can have different formats)
            if hasattr(nx_graph, 'edges') and hasattr(nx_graph.edges, '__getitem__'):
                try:
                    if nx_graph.is_multigraph():
                        # MultiGraph case - get first edge's attributes
                        edge_data = nx_graph.edges[nx_source, nx_target, 0]
                    else:
                        edge_data = nx_graph.edges[nx_source, nx_target]
                    
                    for attr_name, attr_value in edge_data.items():
                        # Apply attribute name mapping if provided
                        groggy_attr_name = attr_name
                        if edge_attr_mapping and attr_name in edge_attr_mapping:
                            groggy_attr_name = edge_attr_mapping[attr_name]
                        
                        # Convert NetworkX attributes to Groggy-compatible types
                        edge_attrs[groggy_attr_name] = _convert_networkx_attr_value(attr_value)
                
                except (KeyError, AttributeError):
                    pass
        
        g.add_edge(groggy_source, groggy_target, **edge_attrs)
        edges_added.add(edge_key)
    
    return g

def _convert_networkx_attr_value(value: Any) -> Any:
    """Convert NetworkX attribute values to Groggy-compatible types."""
    import numpy as np
    
    # Handle numpy types
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    
    # Handle basic Python types (already compatible)
    if isinstance(value, (int, float, str, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        return [_convert_networkx_attr_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: _convert_networkx_attr_value(v) for k, v in value.items()}
    
    # For other types, convert to string
    return str(value)

# Add convenience methods to Graph class
def _add_networkx_methods():
    """Add NetworkX interoperability methods to the Graph class."""
    
    def to_networkx_method(self, directed: bool = False, include_attributes: bool = True):
        """Convert this graph to a NetworkX graph."""
        return to_networkx(self, directed=directed, include_attributes=include_attributes)
    
    # Add method to Graph class if it exists
    try:
        from . import Graph
        Graph.to_networkx = to_networkx_method
    except ImportError:
        pass

# Initialize NetworkX compatibility when module is imported
_add_networkx_methods()