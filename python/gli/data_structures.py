"""
Node and Edge data structures
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Union

# Type alias for node/edge identifiers  
NodeID = Union[str, int]


@dataclass
class Node:
    """Graph node with attributes"""
    id: NodeID
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def get_attribute(self, key: str, default=None):
        return self.attributes.get(key, default)
    
    def set_attribute(self, key: str, value: Any):
        new_attrs = self.attributes.copy()
        new_attrs[key] = value
        return Node(self.id, new_attrs)
    
    # Dict-like access methods
    def get(self, key: str, default=None):
        return self.attributes.get(key, default)
    
    def __getitem__(self, key: str):
        return self.attributes[key]
    
    def __setitem__(self, key: str, value: Any):
        self.attributes[key] = value
    
    def __contains__(self, key: str):
        return key in self.attributes
    
    def keys(self):
        return self.attributes.keys()
    
    def values(self):
        return self.attributes.values()
    
    def items(self):
        return self.attributes.items()
    
    def __iter__(self):
        return iter(self.attributes)


@dataclass 
class Edge:
    """Graph edge with attributes"""
    source: NodeID
    target: NodeID
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self):
        return f"{self.source}->{self.target}"
    
    def get_attribute(self, key: str, default=None):
        return self.attributes.get(key, default)
    
    def set_attribute(self, key: str, value: Any):
        new_attrs = self.attributes.copy()
        new_attrs[key] = value
        return Edge(self.source, self.target, new_attrs)
    
    # Dict-like access methods
    def get(self, key: str, default=None):
        return self.attributes.get(key, default)
    
    def __getitem__(self, key: str):
        return self.attributes[key]
    
    def __setitem__(self, key: str, value: Any):
        self.attributes[key] = value
    
    def __contains__(self, key: str):
        return key in self.attributes
    
    def keys(self):
        return self.attributes.keys()
    
    def values(self):
        return self.attributes.values()
    
    def items(self):
        return self.attributes.items()
    
    def __iter__(self):
        return iter(self.attributes)


class GraphDelta:
    """Tracks changes to a graph for efficient updates"""
    
    def __init__(self):
        self.added_nodes = {}
        self.added_edges = {}
        self.removed_nodes = set()
        self.removed_edges = set()
        self.modified_nodes = {}
        self.modified_edges = {}
    
    def clear(self):
        """Clear all delta information"""
        self.added_nodes.clear()
        self.added_edges.clear()
        self.removed_nodes.clear()
        self.removed_edges.clear()
        self.modified_nodes.clear()
        self.modified_edges.clear()
    
    def is_empty(self):
        """Check if delta has any changes"""
        return not (self.added_nodes or self.added_edges or 
                   self.removed_nodes or self.removed_edges or
                   self.modified_nodes or self.modified_edges)
