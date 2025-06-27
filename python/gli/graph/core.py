"""
Core Graph class implementation
"""

from typing import Dict, List, Any, Optional, Callable, Union

# Type alias for node/edge identifiers
NodeID = Union[str, int]
EdgeID = Union[str, int]
from ..data_structures import Node, Edge
from ..views import LazyDict, NodeView, EdgeView
from .batch import BatchOperationContext
from .state import StateMixin

# Import detection for backend
try:
    from .. import _core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class Graph(StateMixin):
    """High-level Graph interface with automatic Rust/Python backend selection"""
    
    def __init__(self, nodes=None, edges=None, graph_attributes=None, backend=None, max_auto_states: int = 10):
        # Backend selection - use global setting or override
        if backend is None:
            # Import here to avoid circular import
            from .. import get_current_backend
            backend = get_current_backend()
        
        self.backend = backend
        self.use_rust = (backend == 'rust')
        
        if self.use_rust:
            # Rust handles processing, Python manages state/branches
            self._rust_core = _core.FastGraph()
            self._rust_store = _core.GraphStore()
            self._init_rust_backend(nodes, edges, graph_attributes)
            
            # Python manages state and branching
            self.current_hash = None
            self.auto_states = []
            self.max_auto_states = max_auto_states
            
            # Cache for lazy rendering
            self._cache = {}
            self._cache_valid = False
        else:
            # Full Python fallback
            self._init_python_backend(nodes, edges, graph_attributes)
            
            # Python state management for fallback
            from collections import deque
            self._states = {}  # Use _states for Python backend
            self.auto_states = deque(maxlen=max_auto_states)
            self.commits = {}
            self.current_hash = None
        
        # Branch management - current_branch tracked locally, branches via property
        self.current_branch = "main" 
        self.branch_heads = {}
        
        self.graph_attributes = graph_attributes or {}
        self._current_time = 0
    
    def _init_rust_backend(self, nodes, edges, graph_attributes):
        """Initialize Rust backend"""
        if nodes:
            node_data = [(node_id, node.attributes) for node_id, node in nodes.items()]
            self._rust_core.add_nodes(node_data)
        
        if edges:
            edge_data = [(edge.source, edge.target, edge.attributes) for edge in edges.values()]
            self._rust_core.add_edges(edge_data)
    
    def _init_python_backend(self, nodes, edges, graph_attributes):
        """Initialize Python backend (fallback)"""
        # Use regular dict for better performance (use private attributes to avoid property conflict)
        self._nodes = dict(nodes or {})
        self._edges = dict(edges or {})
        
        # Track insertion order more efficiently
        self.node_order = {}
        self.edge_order = {}
        
        # Optimized change tracking
        self._pending_delta = None
        self._is_modified = False
        self._effective_cache = None
        self._cache_valid = True

    @classmethod
    def empty(cls, backend=None):
        """Create an empty graph"""
        return cls(backend=backend)
    
    @classmethod
    def from_node_list(cls, node_ids: List[str], node_attrs: Dict[str, List[Any]] = None, backend=None):
        """Create graph from vectorized node data (NumPy-style)"""
        nodes = {}
        
        if node_attrs:
            # Vectorized attribute assignment
            for i, node_id in enumerate(node_ids):
                attrs = {attr_name: attr_values[i] if i < len(attr_values) else None
                        for attr_name, attr_values in node_attrs.items()}
                nodes[node_id] = Node(node_id, attrs)
        else:
            # Batch create nodes without attributes
            nodes = {node_id: Node(node_id) for node_id in node_ids}
        
        return cls(nodes, {}, {}, backend=backend)
    
    @classmethod
    def from_edge_list(cls, edges: List[tuple], node_attrs: Dict[str, List[Any]] = None, 
                      edge_attrs: Dict[str, List[Any]] = None, backend=None):
        """Create graph from edge list (NetworkX-style)"""
        nodes = {}
        graph_edges = {}
        
        # Extract unique nodes
        node_set = set()
        for edge in edges:
            node_set.add(edge[0])
            node_set.add(edge[1])
        
        # Create nodes with attributes if provided
        if node_attrs:
            for i, node_id in enumerate(sorted(node_set)):
                attrs = {attr_name: attr_values[i] if i < len(attr_values) else None
                        for attr_name, attr_values in node_attrs.items()}
                nodes[node_id] = Node(node_id, attrs)
        else:
            nodes = {node_id: Node(node_id) for node_id in node_set}
        
        # Create edges with attributes if provided
        if edge_attrs:
            for i, edge in enumerate(edges):
                edge_id = f"{edge[0]}->{edge[1]}"
                attrs = {attr_name: attr_values[i] if i < len(attr_values) else None
                        for attr_name, attr_values in edge_attrs.items()}
                graph_edges[edge_id] = Edge(edge[0], edge[1], attrs)
        else:
            for edge in edges:
                edge_id = f"{edge[0]}->{edge[1]}"
                graph_edges[edge_id] = Edge(edge[0], edge[1])
        
        return cls(nodes, graph_edges, {}, backend=backend)

    # Properties for nodes and edges access
    @property
    def nodes(self):
        """Access nodes as a lazy dict-like interface"""
        if self.use_rust:
            if not self._cache_valid or 'nodes' not in self._cache:
                node_ids = self._rust_core.get_node_ids()
                node_dict = {}
                for node_id in node_ids:
                    attrs = self._rust_core.get_node_attributes(node_id)
                    node_dict[node_id] = Node(node_id, attrs)
                self._cache['nodes'] = NodeView(node_dict)
                self._cache_valid = True
            return self._cache['nodes']
        else:
            effective_nodes, _, _ = self._get_effective_data()
            return NodeView(effective_nodes)

    @property
    def edges(self):
        """Access edges as a lazy dict-like interface"""
        if self.use_rust:
            if not self._cache_valid or 'edges' not in self._cache:
                edge_ids = self._rust_core.get_edge_ids()
                edge_dict = {}
                for edge_id in edge_ids:
                    # Parse edge_id to get source and target
                    if "->" in edge_id:
                        source, target = edge_id.split("->", 1)
                        attrs = self._rust_core.get_edge_attributes(source, target)
                        edge_dict[edge_id] = Edge(source, target, attrs)
                self._cache['edges'] = EdgeView(edge_dict)
                self._cache_valid = True
            return self._cache['edges']
        else:
            _, effective_edges, _ = self._get_effective_data()
            return EdgeView(effective_edges)

    @property
    def states(self):
        """Get all states with their hashes (lazy-loaded view into backend)"""
        if self.use_rust:
            # Get comprehensive state info from Rust backend
            stats = self._rust_store.get_stats()
            
            # Get current hash
            try:
                current_hash = self._rust_store.get_current_hash()
            except:
                current_hash = None
            
            # Get branch hashes (these represent saved states)
            branch_list = self._rust_store.list_branches()
            branch_hashes = [hash_val for name, hash_val in branch_list if hash_val != 'initial']
            
            # Combine with auto_states
            all_state_hashes = list(set(branch_hashes + [s for s in self.auto_states if s]))
            if current_hash and current_hash not in all_state_hashes:
                all_state_hashes.append(current_hash)
            
            return {
                'total_states': stats.get('total_states', 0),
                'pooled_nodes': stats.get('pooled_nodes', 0),
                'pooled_edges': stats.get('pooled_edges', 0),
                'node_refs_tracked': stats.get('node_refs_tracked', 0),
                'edge_refs_tracked': stats.get('edge_refs_tracked', 0),
                'current_hash': current_hash,
                'state_hashes': all_state_hashes,
                'auto_states': list(self.auto_states),
                'branches_count': len(branch_list)
            }
        else:
            return getattr(self, '_states', {})
    
    @property 
    def branches(self):
        """Get current branches from backend (lazy-loaded view)"""
        if self.use_rust:
            # Direct call to Rust backend
            rust_branches = self._rust_store.list_branches()
            branch_dict = {}
            for name, hash_val in rust_branches:
                branch_dict[name] = hash_val
            return branch_dict
        else:
            return getattr(self, '_branches', {})

    def add_node(self, node_id: NodeID = None, **attributes) -> str:
        """Add a node to the graph"""
        if node_id is None:
            import uuid
            node_id = f"node_{uuid.uuid4().hex[:8]}"
        
        # Convert node_id to string for consistent backend handling
        node_id_str = str(node_id)
        
        if self.use_rust:
            self._rust_core.add_node(node_id_str, attributes)
            self._invalidate_cache()
        else:
            # Python fallback implementation
            if not self._is_modified and node_id_str in self._nodes:
                return node_id_str
            
            self._init_delta()
            
            if node_id_str not in self._pending_delta.added_nodes:
                new_node = Node(node_id_str, attributes)
                self._pending_delta.added_nodes[node_id_str] = new_node
                self._update_cache_for_node_add(node_id_str, new_node)
        
        return node_id_str

    def add_edge(self, source: NodeID, target: NodeID, **attributes) -> str:
        """Add an edge to the graph"""
        # Convert node IDs to strings for consistent backend handling
        source_str = str(source)
        target_str = str(target)
        edge_id = f"{source_str}->{target_str}"
        
        if self.use_rust:
            self._rust_core.add_edge(source_str, target_str, attributes)
            self._invalidate_cache()
        else:
            # Python fallback implementation
            if not self._is_modified and edge_id in self._edges:
                return edge_id
            
            self._init_delta()
            
            # Check in pending delta
            if edge_id in self._pending_delta.added_edges:
                return edge_id
            
            # Batch node creation - only check/create nodes that don't exist
            effective_nodes, _, _ = self._get_effective_data()
            
            if source not in effective_nodes and source not in self._pending_delta.added_nodes:
                self._pending_delta.added_nodes[source] = Node(source)
            if target not in effective_nodes and target not in self._pending_delta.added_nodes:
                self._pending_delta.added_nodes[target] = Node(target)
            
            new_edge = Edge(source, target, attributes)
            self._pending_delta.added_edges[edge_id] = new_edge
            
            # Try incremental cache update instead of invalidation
            self._update_cache_for_edge_add(edge_id, new_edge)
        
        return edge_id

    def batch_operations(self):
        """Context manager for efficient batch operations"""
        return BatchOperationContext(self)

    def node_count(self) -> int:
        """Get the number of nodes"""
        if self.use_rust:
            return self._rust_core.node_count()
        else:
            effective_nodes, _, _ = self._get_effective_data()
            return len(effective_nodes)

    def edge_count(self) -> int:
        """Get the number of edges"""
        if self.use_rust:
            return self._rust_core.edge_count()
        else:
            _, effective_edges, _ = self._get_effective_data()
            return len(effective_edges)

    def get_node(self, node_id: NodeID) -> Optional[Node]:
        """Get a specific node by ID"""
        node_id_str = str(node_id)
        if self.use_rust:
            attrs = self._rust_core.get_node_attributes(node_id_str)
            if attrs is not None:
                return Node(node_id_str, attrs)
            return None
        else:
            effective_nodes, _, _ = self._get_effective_data()
            return effective_nodes.get(node_id_str)

    def get_edge(self, source: NodeID, target: NodeID) -> Optional[Edge]:
        """Get a specific edge by source and target node IDs"""
        source_str = str(source)
        target_str = str(target)
        if self.use_rust:
            attrs = self._rust_core.get_edge_attributes(source_str, target_str)
            if attrs is not None:
                return Edge(source_str, target_str, attrs)
            return None
        else:
            _, effective_edges, _ = self._get_effective_data()
            edge_id = f"{source_str}->{target_str}"
            return effective_edges.get(edge_id)

    def get_node_ids(self) -> List[str]:
        """Get all node IDs"""
        if self.use_rust:
            return self._rust_core.get_node_ids()
        else:
            effective_nodes, _, _ = self._get_effective_data()
            return list(effective_nodes.keys())

    def get_neighbors(self, node_id: NodeID, direction: str = 'both') -> List[str]:
        """Get neighbors of a node"""
        node_id_str = str(node_id)
        if self.use_rust:
            # Rust backend currently only supports 'both' direction
            return self._rust_core.get_neighbors(node_id_str)
        else:
            neighbors = set()
            _, effective_edges, _ = self._get_effective_data()
            
            for edge in effective_edges.values():
                if direction in ['out', 'both'] and edge.source == node_id_str:
                    neighbors.add(edge.target)
                if direction in ['in', 'both'] and edge.target == node_id_str:
                    neighbors.add(edge.source)
            
            return list(neighbors)

    # High-level wrapper methods to avoid direct _rust_core access
    def filter_nodes(self, filter_func: Union[Callable[[NodeID, Dict[str, Any]], bool], Dict[str, Any]]) -> List[str]:
        """Filter nodes by lambda function or attribute values
        
        Args:
            filter_func: Either a callable that takes (node_id, attributes) and returns bool,
                        or a dictionary of attribute filters
            
        Returns:
            List of node IDs that match the filter
        """
        if callable(filter_func):
            # Lambda function filtering
            if self.use_rust:
                # For Rust backend, we need to iterate through all nodes
                result = []
                for node_id in self.nodes:
                    node = self.get_node(node_id)
                    if node and filter_func(node_id, node.attributes):
                        result.append(node_id)
                return result
            else:
                # Python fallback implementation
                effective_nodes, _, _ = self._get_effective_data()
                result = []
                for node_id, node in effective_nodes.items():
                    if filter_func(node_id, node.attributes):
                        result.append(node_id)
                return result
        else:
            # Dictionary-based attribute filtering
            if self.use_rust:
                return self._rust_core.filter_nodes_by_attributes(filter_func)
            else:
                # Python fallback implementation
                effective_nodes, _, _ = self._get_effective_data()
                result = []
                for node_id, node in effective_nodes.items():
                    match = True
                    for attr, value in filter_func.items():
                        if attr not in node.attributes or node.attributes[attr] != value:
                            match = False
                            break
                    if match:
                        result.append(node_id)
                return result

    def filter_edges(self, filter_func: Union[Callable[[str, NodeID, NodeID, Dict[str, Any]], bool], Dict[str, Any]]) -> List[str]:
        """Filter edges by lambda function or attribute values
        
        Args:
            filter_func: Either a callable that takes (edge_id, source, target, attributes) and returns bool,
                        or a dictionary of attribute filters
            
        Returns:
            List of edge IDs that match the filter
        """
        if callable(filter_func):
            # Lambda function filtering
            if self.use_rust:
                # For Rust backend, we need to iterate through all edges
                result = []
                for edge_id in self.edges:
                    edge = self.edges[edge_id]  # Use the EdgeView instead of get_edge
                    if edge and filter_func(edge_id, edge.source, edge.target, edge.attributes):
                        result.append(edge_id)
                return result
            else:
                # Python fallback implementation
                _, effective_edges, _ = self._get_effective_data()
                result = []
                for edge_id, edge in effective_edges.items():
                    if filter_func(edge_id, edge.source, edge.target, edge.attributes):
                        result.append(edge_id)
                return result
        else:
            # Dictionary-based attribute filtering
            if self.use_rust:
                return self._rust_core.filter_edges_by_attributes(filter_func)
            else:
                # Python fallback implementation
                _, effective_edges, _ = self._get_effective_data()
                result = []
                for edge_id, edge in effective_edges.items():
                    match = True
                    for attr, value in filter_func.items():
                        if attr not in edge.attributes or edge.attributes[attr] != value:
                            match = False
                            break
                    if match:
                        result.append(edge_id)
                return result

    def set_node_attribute(self, node_id: NodeID, attribute: str, value: Any):
        """Set a single attribute on a node
        
        Args:
            node_id: ID of the node
            attribute: Name of the attribute
            value: Value to set
        """
        node_id_str = str(node_id)
        if self.use_rust:
            self._rust_core.set_node_attribute(node_id_str, attribute, value)
            self._invalidate_cache()
        else:
            # Python fallback implementation
            if node_id_str in self._nodes:
                self._nodes[node_id].attributes[attribute] = value
            elif hasattr(self, '_pending_delta') and node_id in self._pending_delta.added_nodes:
                self._pending_delta.added_nodes[node_id].attributes[attribute] = value
            else:
                raise KeyError(f"Node {node_id} not found")

    def set_edge_attribute(self, source: NodeID, target: NodeID, attribute: str, value: Any):
        """Set a single attribute on an edge
        
        Args:
            source: Source node ID
            target: Target node ID
            attribute: Name of the attribute
            value: Value to set
        """
        source_str = str(source)
        target_str = str(target)
        edge_id = f"{source_str}->{target_str}"
        if self.use_rust:
            self._rust_core.set_edge_attribute(source_str, target_str, attribute, value)
            self._invalidate_cache()
        else:
            # Python fallback implementation
            if edge_id in self._edges:
                self._edges[edge_id].attributes[attribute] = value
            elif hasattr(self, '_pending_delta') and edge_id in self._pending_delta.added_edges:
                self._pending_delta.added_edges[edge_id].attributes[attribute] = value
            else:
                raise KeyError(f"Edge {edge_id} not found")

    def set_node_attributes(self, node_id: NodeID, attributes: Dict[str, Any]):
        """Set multiple attributes on a node
        
        Args:
            node_id: ID of the node
            attributes: Dictionary of attributes to set
        """
        for attr, value in attributes.items():
            self.set_node_attribute(node_id, attr, value)

    def set_edge_attributes(self, source: NodeID, target: NodeID, attributes: Dict[str, Any]):
        """Set multiple attributes on an edge
        
        Args:
            source: Source node ID
            target: Target node ID
            attributes: Dictionary of attributes to set
        """
        for attr, value in attributes.items():
            self.set_edge_attribute(source, target, attr, value)

    def set_nodes_attributes_batch(self, node_attrs: Dict[NodeID, Dict[str, Any]]):
        """Set attributes on multiple nodes efficiently
        
        Args:
            node_attrs: Dictionary mapping node_id -> {attribute: value}
        """
        if self.use_rust:
            # Convert node IDs to strings and prepare for Rust backend
            rust_node_attrs = {}
            for node_id, attributes in node_attrs.items():
                rust_node_attrs[str(node_id)] = attributes
            
            self._rust_core.set_nodes_attributes_batch(rust_node_attrs)
            self._invalidate_cache()
        else:
            # Python fallback - just iterate
            for node_id, attributes in node_attrs.items():
                self.set_node_attributes(node_id, attributes)

    def set_edges_attributes_batch(self, edge_attrs: Dict[tuple, Dict[str, Any]]):
        """Set attributes on multiple edges efficiently
        
        Args:
            edge_attrs: Dictionary mapping (source, target) -> {attribute: value}
        """
        if self.use_rust:
            # Convert to format expected by Rust backend
            rust_edge_attrs = {}
            for (source, target), attributes in edge_attrs.items():
                rust_edge_attrs[(str(source), str(target))] = attributes
            
            self._rust_core.set_edges_attributes_batch(rust_edge_attrs)
            self._invalidate_cache()
        else:
            # Python fallback - just iterate
            for (source, target), attributes in edge_attrs.items():
                self.set_edge_attributes(source, target, attributes)

    def get_node_attributes(self, node_id: NodeID) -> Dict[str, Any]:
        """Get all attributes of a node
        
        Args:
            node_id: ID of the node
            
        Returns:
            Dictionary of node attributes
        """
        node = self.get_node(node_id)
        if node:
            return dict(node.attributes)
        else:
            raise KeyError(f"Node {node_id} not found")

    def get_edge_attributes(self, source: NodeID, target: NodeID) -> Dict[str, Any]:
        """Get all attributes of an edge
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Dictionary of edge attributes
        """
        source_str = str(source)
        target_str = str(target)
        edge = self.get_edge(source_str, target_str)
        if edge:
            return dict(edge.attributes)
        else:
            edge_id = f"{source_str}->{target_str}"
            raise KeyError(f"Edge {edge_id} not found")

    def _invalidate_cache(self):
        """Invalidate the lazy rendering cache"""
        if self.use_rust and hasattr(self, '_cache'):
            self._cache_valid = False
            self._cache.clear()

    # Python fallback helper methods
    def _init_delta(self):
        """Initialize delta tracking for Python backend"""
        if not hasattr(self, '_pending_delta') or self._pending_delta is None:
            from ..data_structures import GraphDelta
            self._pending_delta = GraphDelta()
            self._is_modified = True

    def _get_effective_data(self):
        """Get effective nodes, edges, and attributes for Python backend"""
        if self._effective_cache is not None and self._cache_valid:
            return self._effective_cache
        
        # Start with base data
        effective_nodes = self._nodes.copy()
        effective_edges = self._edges.copy()
        effective_attrs = {}
        
        # Apply pending changes
        if self._pending_delta:
            effective_nodes.update(self._pending_delta.added_nodes)
            effective_edges.update(self._pending_delta.added_edges)
            
            # Remove deleted items
            for node_id in self._pending_delta.removed_nodes:
                effective_nodes.pop(node_id, None)
            for edge_id in self._pending_delta.removed_edges:
                effective_edges.pop(edge_id, None)
        
        self._effective_cache = (effective_nodes, effective_edges, effective_attrs)
        self._cache_valid = True
        
        return self._effective_cache

    def _update_cache_for_node_add(self, node_id: NodeID, node: Node):
        """Update cache when adding a node"""
        if self._effective_cache:
            self._effective_cache[0][node_id] = node

    def _update_cache_for_edge_add(self, edge_id: str, edge: Edge):
        """Update cache when adding an edge"""
        if self._effective_cache:
            self._effective_cache[1][edge_id] = edge

    def _apply_batch_operations(self, batch_nodes, batch_edges):
        """Apply batched operations efficiently"""
        if not batch_nodes and not batch_edges:
            return
        
        if self.use_rust:
            # Apply to Rust backend
            if batch_nodes:
                node_data = [(node_id, node.attributes) for node_id, node in batch_nodes.items()]
                self._rust_core.add_nodes(node_data)
            
            if batch_edges:
                edge_data = [(edge.source, edge.target, edge.attributes) for edge in batch_edges.values()]
                self._rust_core.add_edges(edge_data)
            
            self._invalidate_cache()
        else:
            # Apply to Python backend
            self._init_delta()
            self._pending_delta.added_nodes.update(batch_nodes)
            self._pending_delta.added_edges.update(batch_edges)
            self._cache_valid = False

    # Backward compatibility aliases
    def filter_nodes_by_attributes(self, filters: Dict[str, Any]) -> List[str]:
        """Legacy alias for filter_nodes with attribute dictionary"""
        return self.filter_nodes(filters)
    
    def filter_edges_by_attributes(self, filters: Dict[str, Any]) -> List[str]:
        """Legacy alias for filter_edges with attribute dictionary"""
        return self.filter_edges(filters)
