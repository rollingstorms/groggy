"""
Lazy views and data structures for zero-copy operations
"""

from typing import Dict, Any, Set


class LazyDict:
    """Zero-copy dictionary view that combines base dict with delta changes"""
    
    def __init__(self, base_dict: Dict = None, delta_added: Dict = None, 
                 delta_removed: Set = None, delta_modified: Dict = None):
        self.base = base_dict or {}
        self.added = delta_added or {}
        self.removed = delta_removed or set()
        self.modified = delta_modified or {}
    
    def __getitem__(self, key):
        if key in self.removed:
            raise KeyError(key)
        if key in self.modified:
            return self.modified[key]
        if key in self.added:
            return self.added[key]
        return self.base[key]
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def __contains__(self, key):
        if key in self.removed:
            return False
        return key in self.added or key in self.modified or key in self.base
    
    def keys(self):
        # Combine keys from all sources, excluding removed
        all_keys = set(self.base.keys()) | set(self.added.keys()) | set(self.modified.keys())
        return all_keys - self.removed
    
    def values(self):
        return [self[key] for key in self.keys()]
    
    def items(self):
        return [(key, self[key]) for key in self.keys()]
    
    def __len__(self):
        return len(self.keys())
    
    def __iter__(self):
        return iter(self.keys())
    
    def copy(self):
        """Force materialization to regular dict"""
        return {key: self[key] for key in self.keys()}


class NodeView:
    """View of nodes in Rust backend"""
    
    def __init__(self, rust_core):
        self._rust_core = rust_core
    
    def __len__(self):
        return self._rust_core.node_count()
    
    def __contains__(self, node_id):
        try:
            self._rust_core.get_node_attributes(node_id)
            return True
        except:
            return False
    
    def __getitem__(self, node_id):
        attrs = self._rust_core.get_node_attributes(node_id)
        from .data_structures import Node
        return Node(node_id, attrs)
    
    def keys(self):
        return self._rust_core.get_node_ids()
    
    def values(self):
        return [self[node_id] for node_id in self.keys()]
    
    def items(self):
        return [(node_id, self[node_id]) for node_id in self.keys()]


class EdgeView:
    """View of edges in Rust backend"""
    
    def __init__(self, rust_core):
        self._rust_core = rust_core
    
    def __len__(self):
        return self._rust_core.edge_count()
    
    # TODO: Implement edge access methods when Rust backend supports it
