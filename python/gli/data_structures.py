"""
Node and Edge data structures
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Node:
    """Graph node with attributes"""
    id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def get_attribute(self, key: str, default=None):
        return self.attributes.get(key, default)
    
    def set_attribute(self, key: str, value: Any):
        new_attrs = self.attributes.copy()
        new_attrs[key] = value
        return Node(self.id, new_attrs)


@dataclass 
class Edge:
    """Graph edge with attributes"""
    source: str
    target: str
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
