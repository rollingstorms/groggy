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
    """View of nodes - works with both dict and Rust backend"""
    
    def __init__(self, data):
        if isinstance(data, dict):
            # Working with cached dict data
            self._data = data
            self._rust_core = None
        else:
            # Working with Rust core
            self._rust_core = data
            self._data = None
    
    def __len__(self):
        if self._data is not None:
            return len(self._data)
        else:
            return self._rust_core.node_count()
    
    def __contains__(self, node_id):
        if self._data is not None:
            return node_id in self._data
        else:
            try:
                self._rust_core.get_node_attributes(node_id)
                return True
            except:
                return False
    
    def __getitem__(self, node_id):
        if self._data is not None:
            return self._data[node_id]
        else:
            attrs = self._rust_core.get_node_attributes(node_id)
            from .data_structures import Node
            return Node(node_id, attrs)
    
    def keys(self):
        if self._data is not None:
            return self._data.keys()
        else:
            return self._rust_core.get_node_ids()
    
    def __iter__(self):
        return iter(self.keys())
    
    def values(self):
        if self._data is not None:
            return self._data.values()
        else:
            return [self[node_id] for node_id in self.keys()]
    
    def items(self):
        if self._data is not None:
            return list(self._data.items())  # Convert to list for subscriptability
        else:
            return [(node_id, self[node_id]) for node_id in self.keys()]
    
    def items_iter(self):
        """Iterator version of items() for memory efficiency"""
        if self._data is not None:
            return iter(self._data.items())
        else:
            return ((node_id, self[node_id]) for node_id in self.keys())


class EdgeView:
    """View of edges - works with both dict and Rust backend"""
    
    def __init__(self, data):
        if isinstance(data, dict):
            # Working with cached dict data
            self._data = data
            self._rust_core = None
        else:
            # Working with Rust core
            self._rust_core = data
            self._data = None
    
    def __len__(self):
        if self._data is not None:
            return len(self._data)
        else:
            return self._rust_core.edge_count()
    
    def __contains__(self, edge_id):
        if self._data is not None:
            return edge_id in self._data
        else:
            try:
                parts = edge_id.split('->')
                if len(parts) != 2:
                    return False
                source, target = parts
                self._rust_core.get_edge_attributes(source, target)
                return True
            except:
                return False
    
    def __getitem__(self, edge_id):
        if self._data is not None:
            return self._data[edge_id]
        else:
            parts = edge_id.split('->')
            if len(parts) != 2:
                raise KeyError(f"Invalid edge ID format: {edge_id}")
            source, target = parts
            attrs = self._rust_core.get_edge_attributes(source, target)
            from .data_structures import Edge
            return Edge(source, target, attrs)
    
    def keys(self):
        if self._data is not None:
            return self._data.keys()
        else:
            return self._rust_core.get_edge_ids()
    
    def __iter__(self):
        return iter(self.keys())
    
    def values(self):
        if self._data is not None:
            return self._data.values()
        else:
            return [self[edge_id] for edge_id in self.keys()]
    
    def items(self):
        if self._data is not None:
            return list(self._data.items())  # Convert to list for subscriptability
        else:
            return [(edge_id, self[edge_id]) for edge_id in self.keys()]
    
    def items_iter(self):
        """Iterator version of items() for memory efficiency"""
        if self._data is not None:
            return iter(self._data.items())
        else:
            return ((edge_id, self[edge_id]) for edge_id in self.keys())
