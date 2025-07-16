# groggy/__init__.py
"""
Groggy: A high-performance graph library with Rust backend.

This module provides a clean Python API for graph operations,
backed by a fast Rust implementation with SIMD acceleration.
"""

import json
from typing import Any, List, Union

# Import the Rust backend
try:
    from groggy._core import (
        FastGraph as _FastGraph,
        NodeId as _NodeId, 
        EdgeId as _EdgeId,
        NodeCollection as _NodeCollection,
        EdgeCollection as _EdgeCollection,
        NodeProxy as _NodeProxy,
        EdgeProxy as _EdgeProxy,
    )
except ImportError as e:
    raise ImportError(f"Failed to import Rust backend: {e}")


class NodeProxy:
    """Python wrapper for NodeProxy that handles JSON serialization automatically."""
    
    def __init__(self, proxy: _NodeProxy):
        self._proxy = proxy
    
    def set_attr(self, key: str, value: Any) -> None:
        """Set an attribute on this node. Value is automatically JSON-serialized."""
        json_value = json.dumps(value)
        self._proxy.set_attr(key, json_value)
    
    def get_attr(self, key: str) -> Any:
        """Get an attribute from this node. Returns the Python object."""
        json_value = self._proxy.get_attr(key)
        if json_value is None:
            return None
        return json.loads(json_value)
    
    def __repr__(self):
        return f"NodeProxy({self._proxy})"


class EdgeProxy:
    """Python wrapper for EdgeProxy that handles JSON serialization automatically."""
    
    def __init__(self, proxy: _EdgeProxy):
        self._proxy = proxy
    
    def set_attr(self, key: str, value: Any) -> None:
        """Set an attribute on this edge. Value is automatically JSON-serialized."""
        json_value = json.dumps(value)
        self._proxy.set_attr(key, json_value)
    
    def get_attr(self, key: str) -> Any:
        """Get an attribute from this edge. Returns the Python object."""
        json_value = self._proxy.get_attr(key)
        if json_value is None:
            return None
        return json.loads(json_value)
    
    def __repr__(self):
        return f"EdgeProxy({self._proxy})"


class NodeCollection:
    """Python wrapper for NodeCollection with improved API."""
    
    def __init__(self, collection: _NodeCollection):
        self._collection = collection
    
    def add(self, nodes: Union[List[_NodeId], _NodeId]) -> None:
        """Add nodes to the collection."""
        if isinstance(nodes, _NodeId):
            nodes = [nodes]
        self._collection.add(nodes)
    
    def get(self, node_id: _NodeId) -> NodeProxy:
        """Get a node proxy for the given node ID."""
        proxy = self._collection.get(node_id)
        return NodeProxy(proxy) if proxy else None
    
    def size(self) -> int:
        """Get the number of nodes in the collection."""
        return self._collection.size()
    
    def ids(self) -> List[str]:
        """Get all node IDs in the collection."""
        return self._collection.ids()


class EdgeCollection:
    """Python wrapper for EdgeCollection with improved API."""
    
    def __init__(self, collection: _EdgeCollection):
        self._collection = collection
    
    def add(self, edges: Union[List[_EdgeId], _EdgeId]) -> None:
        """Add edges to the collection."""
        if isinstance(edges, _EdgeId):
            edges = [edges]
        self._collection.add(edges)
    
    def get(self, edge_id: _EdgeId) -> EdgeProxy:
        """Get an edge proxy for the given edge ID."""
        proxy = self._collection.get(edge_id)
        return EdgeProxy(proxy) if proxy else None
    
    def size(self) -> int:
        """Get the number of edges in the collection."""
        return self._collection.size()
    
    def ids(self) -> List[str]:
        """Get all edge IDs in the collection."""
        return self._collection.ids()


class Graph:
    """
    Main graph class with clean Python API.
    
    This wraps the Rust FastGraph with a more Pythonic interface
    and automatic JSON handling for attributes.
    """
    
    def __init__(self):
        self._graph = _FastGraph()
    
    def nodes(self) -> NodeCollection:
        """Get the node collection for this graph."""
        return NodeCollection(self._graph.nodes())
    
    def edges(self) -> EdgeCollection:
        """Get the edge collection for this graph."""
        return EdgeCollection(self._graph.edges())
    
    def __repr__(self):
        return f"Graph(nodes={self.nodes().size()}, edges={self.edges().size()})"


# Export the ID types directly 
NodeId = _NodeId
EdgeId = _EdgeId

# Export main classes
__all__ = ['Graph', 'NodeId', 'EdgeId', 'NodeProxy', 'EdgeProxy', 'NodeCollection', 'EdgeCollection']
