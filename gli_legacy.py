"""
GLI - Graph Language Interface
Core library for graph manipulation with git-like state management and dynamic module loading.
"""

import hashlib
import json
import time
import importlib.util
# import yaml  # Optional, comment out for now
import weakref
from collections import OrderedDict, deque
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import functools

# Try to use faster xxhash if available, fallback to hashlib
try:
    import xxhash
    def fast_hash(data: str) -> str:
        return xxhash.xxh64(data.encode()).hexdigest()[:16]
except ImportError:
    def fast_hash(data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()[:16]


def performance_monitor(func):
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Store timing info (could be extended to write to log/metrics)
        duration = end_time - start_time
        if duration > 0.01:  # Only log slow operations (>10ms)
            print(f"PERF: {func.__name__} took {duration*1000:.2f}ms")
        
        return result
    return wrapper


class LazyDict:
    """Zero-copy dictionary view that combines base dict with delta changes"""
    
    def __init__(self, base_dict: Dict, delta_added: Dict = None, delta_removed: set = None, delta_modified: Dict = None):
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


class BatchOperationContext:
    """Context manager for efficient batch operations"""
    
    def __init__(self, graph: 'Graph'):
        self.graph = graph
        self.batch_nodes = {}
        self.batch_edges = {}
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.batch_nodes or self.batch_edges:
            self.graph._apply_batch_operations(self.batch_nodes, self.batch_edges)
    
    def add_node(self, node_id: str, **attributes):
        """Queue node for batch addition"""
        if node_id not in self.graph.nodes and node_id not in self.batch_nodes:
            self.batch_nodes[node_id] = Node(node_id, attributes)
    
    def add_edge(self, source: str, target: str, **attributes):
        """Queue edge for batch addition"""
        edge_id = f"{source}->{target}"
        if edge_id not in self.graph.edges and edge_id not in self.batch_edges:
            # Ensure nodes exist
            if source not in self.graph.nodes and source not in self.batch_nodes:
                self.batch_nodes[source] = Node(source)
            if target not in self.graph.nodes and target not in self.batch_nodes:
                self.batch_nodes[target] = Node(target)
            
            self.batch_edges[edge_id] = Edge(source, target, attributes)


class ContentPool:
    """Content-addressed storage for nodes and edges to avoid duplication"""
    
    def __init__(self):
        self.nodes = {}  # content_hash -> Node
        self.edges = {}  # content_hash -> Edge
        self.node_refs = {}  # track references
        self.edge_refs = {}  # track references
        # Add caches for hash computation
        self._node_hash_cache = {}  # (node_id, attrs_hash) -> content_hash
        self._edge_hash_cache = {}  # (source, target, attrs_hash) -> content_hash
    
    def _node_content_hash(self, node: 'Node') -> str:
        # Use simpler hash for better performance
        attrs_str = json.dumps(node.attributes, sort_keys=True, separators=(',', ':'))
        cache_key = (node.id, hash(attrs_str))
        
        if cache_key not in self._node_hash_cache:
            content = f"{node.id}:{attrs_str}"
            self._node_hash_cache[cache_key] = fast_hash(content)
        
        return self._node_hash_cache[cache_key]
    
    def _edge_content_hash(self, edge: 'Edge') -> str:
        attrs_str = json.dumps(edge.attributes, sort_keys=True, separators=(',', ':'))
        cache_key = (edge.source, edge.target, hash(attrs_str))
        
        if cache_key not in self._edge_hash_cache:
            content = f"{edge.source}->{edge.target}:{attrs_str}"
            self._edge_hash_cache[cache_key] = fast_hash(content)
        
        return self._edge_hash_cache[cache_key]
    
    def intern_node(self, node: 'Node') -> str:
        """Store node in pool, return content hash"""
        content_hash = self._node_content_hash(node)
        if content_hash not in self.nodes:
            self.nodes[content_hash] = node
            self.node_refs[content_hash] = 0
        self.node_refs[content_hash] += 1
        return content_hash
    
    def intern_edge(self, edge: 'Edge') -> str:
        """Store edge in pool, return content hash"""
        content_hash = self._edge_content_hash(edge)
        if content_hash not in self.edges:
            self.edges[content_hash] = edge
            self.edge_refs[content_hash] = 0
        self.edge_refs[content_hash] += 1
        return content_hash
    
    def get_node(self, content_hash: str) -> Optional['Node']:
        return self.nodes.get(content_hash)
    
    def get_edge(self, content_hash: str) -> Optional['Edge']:
        return self.edges.get(content_hash)
    
    def release_node(self, content_hash: str):
        """Decrement reference count, cleanup if unused"""
        if content_hash in self.node_refs:
            self.node_refs[content_hash] -= 1
            if self.node_refs[content_hash] <= 0:
                self.nodes.pop(content_hash, None)
                self.node_refs.pop(content_hash, None)
    
    def release_edge(self, content_hash: str):
        """Decrement reference count, cleanup if unused"""
        if content_hash in self.edge_refs:
            self.edge_refs[content_hash] -= 1
            if self.edge_refs[content_hash] <= 0:
                self.edges.pop(content_hash, None)
                self.edge_refs.pop(content_hash, None)


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
    

@dataclass
class CompactGraphDelta:
    """Compact delta using content hashes instead of full objects"""
    added_nodes: Dict[str, str] = field(default_factory=dict)  # node_id -> content_hash
    removed_nodes: set = field(default_factory=set)  # node_ids
    modified_nodes: Dict[str, str] = field(default_factory=dict)  # node_id -> content_hash
    added_edges: Dict[str, str] = field(default_factory=dict)  # edge_id -> content_hash
    removed_edges: set = field(default_factory=set)  # edge_ids
    modified_edges: Dict[str, str] = field(default_factory=dict)  # edge_id -> content_hash
    modified_graph_attrs: Dict[str, Any] = field(default_factory=dict)
    
    def is_empty(self) -> bool:
        """Check if delta has any changes"""
        return not any([
            self.added_nodes, self.removed_nodes, self.modified_nodes,
            self.added_edges, self.removed_edges, self.modified_edges,
            self.modified_graph_attrs
        ])


@dataclass
class Branch:
    """Represents a named branch in the graph store"""
    name: str
    current_hash: str
    created_from: str  # hash where branch was created
    created_at: float = field(default_factory=time.time)
    description: str = ""
    
    # Subgraph branching support
    is_subgraph: bool = False
    subgraph_filter: Optional[Dict[str, Any]] = None  # Filter criteria for subgraph
    parent_branch: Optional[str] = None  # Parent branch for subgraph


@dataclass
class GraphState:
    """Optimized graph state using deltas and content addressing"""
    hash: str
    parent_hash: Optional[str] = None
    operation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    branch_name: Optional[str] = None  # Track which branch this state belongs to
    
    # For root states: full content
    nodes: Optional[Dict[str, str]] = None  # node_id -> content_hash
    edges: Optional[Dict[str, str]] = None  # edge_id -> content_hash
    graph_attributes: Optional[Dict[str, Any]] = None
    
    # For incremental states: delta only
    delta: Optional[CompactGraphDelta] = None
    
    # Subgraph support
    is_subgraph_state: bool = False
    subgraph_metadata: Optional[Dict[str, Any]] = None
    
    def is_root(self) -> bool:
        return self.parent_hash is None
    
    def to_dict(self, content_pool: ContentPool):
        """Serialize for hashing"""
        if self.is_root():
            # Reconstruct full content for hashing
            nodes_data = {}
            edges_data = {}
            
            for node_id, content_hash in (self.nodes or {}).items():
                node = content_pool.get_node(content_hash)
                if node:
                    nodes_data[node_id] = {'id': node.id, 'attributes': node.attributes}
            
            for edge_id, content_hash in (self.edges or {}).items():
                edge = content_pool.get_edge(content_hash)
                if edge:
                    edges_data[edge_id] = {'source': edge.source, 'target': edge.target, 'attributes': edge.attributes}
            
            return {
                'nodes': nodes_data,
                'edges': edges_data,
                'graph_attributes': self.graph_attributes or {}
            }
        else:
            # Delta state - compute effective content for hashing
            return self._compute_effective_content()
    
    def _compute_effective_content(self):
        """Compute effective state by applying delta chain"""
        return {
            'delta_hash': str(hash(str(self.delta))),
            'parent': self.parent_hash,
            'operation': self.operation
        }


@dataclass
class GraphDelta:
    """Tracks pending changes to avoid full graph copies"""
    added_nodes: Dict[str, Node] = field(default_factory=dict)
    removed_nodes: set = field(default_factory=set)
    modified_nodes: Dict[str, Node] = field(default_factory=dict)
    added_edges: Dict[str, Edge] = field(default_factory=dict)
    removed_edges: set = field(default_factory=set)
    modified_edges: Dict[str, Edge] = field(default_factory=dict)
    modified_graph_attrs: Dict[str, Any] = field(default_factory=dict)


class Graph:
    """Optimized Graph with lazy copy-on-write and batch operations"""
    
    def __init__(self, nodes=None, edges=None, graph_attributes=None, graph_store=None):
        # Use regular dict for better performance
        self.nodes = dict(nodes or {})
        self.edges = dict(edges or {})
        self.graph_attributes = graph_attributes or {}
        self.graph_store = graph_store
        self._current_time = 0
        
        # Track insertion order more efficiently
        self.node_order = {}
        self.edge_order = {}
        
        # Optimized change tracking
        self._pending_delta: Optional[GraphDelta] = None
        self._is_modified = False
        self._effective_cache = None  # Cache effective data computation
        self._cache_valid = True
    
        # Branch and subgraph metadata
        self.branch_name: Optional[str] = None
        self.is_subgraph = False
        self.subgraph_metadata = {}
    
    @classmethod
    def empty(cls, graph_store=None):
        """Create empty graph"""
        return cls(graph_store=graph_store)
    
    @classmethod
    def from_node_list(cls, node_ids: List[str], node_attrs: Dict[str, List[Any]] = None, graph_store=None):
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
        
        return cls(nodes, {}, {}, graph_store)
    
    @classmethod
    def from_edge_list(cls, edges: List[tuple], node_attrs: Dict[str, List[Any]] = None, 
                      edge_attrs: Dict[str, List[Any]] = None, graph_store=None):
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
        for i, edge in enumerate(edges):
            source, target = edge[0], edge[1]
            edge_id = f"{source}->{target}"
            
            if edge_attrs:
                attrs = {attr_name: attr_values[i] if i < len(attr_values) else None
                        for attr_name, attr_values in edge_attrs.items()}
            else:
                attrs = {}
            
            graph_edges[edge_id] = Edge(source, target, attrs)
        
        return cls(nodes, graph_edges, {}, graph_store)

    def _next_time(self):
        self._current_time += 1
        return self._current_time
    
    def _invalidate_cache(self):
        """Invalidate effective data cache"""
        self._effective_cache = None
        self._cache_valid = False
    
    def _update_cache_for_node_add(self, node_id: str, node: Node):
        """Incrementally update cache when adding node"""
        if self._cache_valid and self._effective_cache:
            effective_nodes, effective_edges, effective_attrs = self._effective_cache
            if hasattr(effective_nodes, 'added'):
                # LazyDict - just add to added dict
                effective_nodes.added[node_id] = node
            else:
                # Regular dict - invalidate to be safe
                self._invalidate_cache()
    
    def _update_cache_for_edge_add(self, edge_id: str, edge: Edge):
        """Incrementally update cache when adding edge"""
        if self._cache_valid and self._effective_cache:
            effective_nodes, effective_edges, effective_attrs = self._effective_cache
            if hasattr(effective_edges, 'added'):
                # LazyDict - just add to added dict
                effective_edges.added[edge_id] = edge
            else:
                # Regular dict - invalidate to be safe
                self._invalidate_cache()
    
    def _init_delta(self):
        """Initialize delta tracking for copy-on-write - now truly lazy"""
        if self._is_modified:
            return
            
        self._is_modified = True
        self._pending_delta = GraphDelta()
        self._invalidate_cache()
        
        # Don't copy collections immediately - do it on first actual write
    
    def _ensure_writable(self):
        """Ensure collections are writable (true copy-on-write)"""
        if self._is_modified and self.nodes is not None:
            # Only copy when we actually need to write
            self.nodes = self.nodes.copy()
            self.edges = self.edges.copy()
            self.graph_attributes = self.graph_attributes.copy()
            self.node_order = self.node_order.copy()
            self.edge_order = self.edge_order.copy()
            # Mark as None to avoid re-copying
            self._original_refs = None
    
    def _get_effective_data(self):
        """Get effective nodes/edges including pending changes - zero-copy lazy views"""
        if self._cache_valid and self._effective_cache is not None:
            return self._effective_cache
        
        if not self._pending_delta:
            result = (self.nodes, self.edges, self.graph_attributes)
            self._effective_cache = result
            self._cache_valid = True
            return result
        
        delta = self._pending_delta
        
        # Create lazy views instead of copying - dramatically faster!
        effective_nodes = LazyDict(
            self.nodes, 
            delta.added_nodes, 
            delta.removed_nodes, 
            delta.modified_nodes
        )
        
        # For edges, we need to handle node removal edge case
        edges_to_remove = set()
        if delta.removed_nodes:
            # Only compute this if we actually have removed nodes
            for edge_id, edge in self.edges.items():
                if edge.source in delta.removed_nodes or edge.target in delta.removed_nodes:
                    edges_to_remove.add(edge_id)
        
        # Combine removed edges with edges removed due to node removal
        all_removed_edges = delta.removed_edges | edges_to_remove
        
        effective_edges = LazyDict(
            self.edges,
            delta.added_edges,
            all_removed_edges,
            delta.modified_edges
        )
        
        # Graph attributes - only copy if modified
        if delta.modified_graph_attrs:
            effective_attrs = {**self.graph_attributes, **delta.modified_graph_attrs}
        else:
            effective_attrs = self.graph_attributes
        
        result = (effective_nodes, effective_edges, effective_attrs)
        self._effective_cache = result
        self._cache_valid = True
        return result
    
    def batch_operations(self):
        """Context manager for efficient batch operations"""
        return BatchOperationContext(self)
    
    def _apply_batch_operations(self, batch_nodes: Dict[str, Node], batch_edges: Dict[str, Edge]):
        """Apply batched operations efficiently"""
        if not batch_nodes and not batch_edges:
            return
        
        self._init_delta()
        
        # Add all nodes
        for node_id, node in batch_nodes.items():
            if node_id not in self.nodes:
                self._pending_delta.added_nodes[node_id] = node
        
        # Add all edges  
        for edge_id, edge in batch_edges.items():
            if edge_id not in self.edges:
                self._pending_delta.added_edges[edge_id] = edge
        
        self._invalidate_cache()
    
    def add_node(self, node_id: str, **attributes) -> 'Graph':
        """Add node with optimized checking"""
        # Quick check without effective data computation for performance
        if not self._is_modified and node_id in self.nodes:
            return self
        
        # Initialize delta first, then check
        self._init_delta()
        
        # Check in pending delta
        if node_id in self._pending_delta.added_nodes:
            return self
        
        new_node = Node(node_id, attributes)
        self._pending_delta.added_nodes[node_id] = new_node
        
        # Try incremental cache update instead of invalidation
        self._update_cache_for_node_add(node_id, new_node)
        
        return self
    
    def add_edge(self, source: str, target: str, **attributes) -> 'Graph':
        """Add edge with optimized node creation"""
        edge_id = f"{source}->{target}"
        
        # Quick duplicate check
        if not self._is_modified and edge_id in self.edges:
            return self
        
        # Initialize delta first
        self._init_delta()
        
        # Check in pending delta
        if edge_id in self._pending_delta.added_edges:
            return self
        
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
        
        return self
    
    def batch_add_nodes(self, node_data: List[tuple]) -> 'Graph':
        """Add multiple nodes efficiently"""
        if not node_data:
            return self
        
        self._init_delta()
        
        for item in node_data:
            if len(item) == 1:
                node_id = item[0]
                attributes = {}
            else:
                node_id, attributes = item[0], item[1]
            
            if node_id not in self.nodes and node_id not in self._pending_delta.added_nodes:
                self._pending_delta.added_nodes[node_id] = Node(node_id, attributes)
        
        self._invalidate_cache()
        return self
    
    def batch_add_edges(self, edge_data: List[tuple]) -> 'Graph':
        """Add multiple edges efficiently"""
        if not edge_data:
            return self
        
        self._init_delta()
        effective_nodes, _, _ = self._get_effective_data()
        
        # Pre-collect all nodes we need to create
        nodes_to_create = set()
        edges_to_create = []
        
        for item in edge_data:
            if len(item) == 2:
                source, target = item
                attributes = {}
            else:
                source, target, attributes = item[0], item[1], item[2]
            
            edge_id = f"{source}->{target}"
            if edge_id not in self.edges and edge_id not in self._pending_delta.added_edges:
                edges_to_create.append((source, target, attributes))
                
                if source not in effective_nodes and source not in self._pending_delta.added_nodes:
                    nodes_to_create.add(source)
                if target not in effective_nodes and target not in self._pending_delta.added_nodes:
                    nodes_to_create.add(target)
        
        # Batch create nodes
        for node_id in nodes_to_create:
            self._pending_delta.added_nodes[node_id] = Node(node_id)
        
        # Batch create edges
        for source, target, attributes in edges_to_create:
            edge_id = f"{source}->{target}"
            self._pending_delta.added_edges[edge_id] = Edge(source, target, attributes)
        
        self._invalidate_cache()
        return self
    
    def _apply_pending_changes(self):
        """Apply pending delta changes"""
        if not self._pending_delta:
            return
            
        delta = self._pending_delta
        
        # Apply node changes
        for node_id, node in delta.added_nodes.items():
            self.nodes[node_id] = node
            if node_id not in self.node_order:
                self.node_order[node_id] = self._next_time()
        
        for node_id in delta.removed_nodes:
            self.nodes.pop(node_id, None)
            self.node_order.pop(node_id, None)
            # Remove connected edges
            edges_to_remove = [eid for eid, edge in self.edges.items() 
                             if edge.source == node_id or edge.target == node_id]
            for eid in edges_to_remove:
                self.edges.pop(eid, None)
                self.edge_order.pop(eid, None)
        
        for node_id, node in delta.modified_nodes.items():
            if node_id in self.nodes:
                self.nodes[node_id] = node
        
        # Apply edge changes
        for edge_id, edge in delta.added_edges.items():
            self.edges[edge_id] = edge
            if edge_id not in self.edge_order:
                self.edge_order[edge_id] = self._next_time()
        
        for edge_id in delta.removed_edges:
            self.edges.pop(edge_id, None)
            self.edge_order.pop(edge_id, None)
        
        for edge_id, edge in delta.modified_edges.items():
            if edge_id in self.edges:
                self.edges[edge_id] = edge
        
        # Apply graph attribute changes
        self.graph_attributes.update(delta.modified_graph_attrs)
        
        self._pending_delta = None
    
    def snapshot(self) -> 'Graph':
        """Create optimized snapshot"""
        if not self._pending_delta:
            # No changes, return shallow copy with shared data
            new_graph = Graph(self.nodes, self.edges, self.graph_attributes, self.graph_store)
            new_graph.node_order = self.node_order
            new_graph.edge_order = self.edge_order
            new_graph._current_time = self._current_time
            return new_graph
        
        # Apply changes efficiently
        self._apply_pending_changes()
        
        # Create new graph with copied data
        new_graph = Graph(
            dict(self.nodes),  # Use dict() for faster copying
            dict(self.edges), 
            self.graph_attributes.copy(), 
            self.graph_store
        )
        new_graph.node_order = self.node_order.copy()
        new_graph.edge_order = self.edge_order.copy()
        new_graph._current_time = self._current_time
        
        return new_graph
    
    def create_subgraph(self, node_filter: Callable[[Node], bool] = None, 
                       edge_filter: Callable[[Edge], bool] = None,
                       node_ids: set = None, include_edges: bool = True) -> 'Graph':
        """Create a subgraph based on filters or node IDs"""
        effective_nodes, effective_edges, effective_attrs = self._get_effective_data()
        
        # Determine which nodes to include
        if node_ids:
            filtered_nodes = {nid: node for nid, node in effective_nodes.items() if nid in node_ids}
        elif node_filter:
            filtered_nodes = {nid: node for nid, node in effective_nodes.items() if node_filter(node)}
        else:
            filtered_nodes = effective_nodes.copy()
        
        # Filter edges
        filtered_edges = {}
        if include_edges:
            for eid, edge in effective_edges.items():
                # Include edge if both endpoints are in filtered nodes
                if edge.source in filtered_nodes and edge.target in filtered_nodes:
                    if not edge_filter or edge_filter(edge):
                        filtered_edges[eid] = edge
        
        # Create subgraph
        subgraph = Graph(filtered_nodes, filtered_edges, effective_attrs.copy(), self.graph_store)
        subgraph.is_subgraph = True
        subgraph.subgraph_metadata = {
            'parent_graph_hash': getattr(self, '_state_hash', None),
            'node_count': len(filtered_nodes),
            'edge_count': len(filtered_edges),
            'created_at': time.time(),
            'filter_type': 'node_ids' if node_ids else 'function' if node_filter else 'all'
        }
        
        return subgraph
    
    def get_subgraph_by_attribute(self, node_attr: str, attr_value: Any) -> 'Graph':
        """Create subgraph containing nodes with specific attribute value"""
        return self.create_subgraph(
            node_filter=lambda node: node.get_attribute(node_attr) == attr_value
        )
    
    def get_connected_component(self, start_node_id: str) -> 'Graph':
        """Get connected component containing the specified node"""
        effective_nodes, effective_edges, _ = self._get_effective_data()
        
        if start_node_id not in effective_nodes:
            return Graph.empty(self.graph_store)
        
        # BFS to find connected component
        visited = set()
        queue = [start_node_id]
        visited.add(start_node_id)
        
        while queue:
            current = queue.pop(0)
            for edge in effective_edges.values():
                neighbor = None
                if edge.source == current and edge.target not in visited:
                    neighbor = edge.target
                elif edge.target == current and edge.source not in visited:
                    neighbor = edge.source
                
                if neighbor and neighbor in effective_nodes:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return self.create_subgraph(node_ids=visited)
    
    def to_networkx(self):
        """Export to NetworkX (if available)"""
        try:
            import networkx as nx
            effective_nodes, effective_edges, _ = self._get_effective_data()
            G = nx.Graph()
            for node in effective_nodes.values():
                G.add_node(node.id, **node.attributes)
            for edge in effective_edges.values():
                G.add_edge(edge.source, edge.target, **edge.attributes)
            return G
        except ImportError:
            raise ImportError("NetworkX not available")
    
    def to_graphml(self) -> str:
        """Export to GraphML format"""
        effective_nodes, effective_edges, _ = self._get_effective_data()
        
        nodes_xml = []
        for node in effective_nodes.values():
            attrs = ' '.join(f'{k}="{v}"' for k, v in node.attributes.items())
            nodes_xml.append(f'    <node id="{node.id}" {attrs}/>')
        
        edges_xml = []
        for edge in effective_edges.values():
            attrs = ' '.join(f'{k}="{v}"' for k, v in edge.attributes.items())
            edges_xml.append(f'    <edge source="{edge.source}" target="{edge.target}" {attrs}/>')
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<graphml>
  <graph>
{chr(10).join(nodes_xml)}
{chr(10).join(edges_xml)}
  </graph>
</graphml>"""


class EventBus:
    """Simple event system for module coordination"""
    
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, event_name: str, callback: Callable, *args, **kwargs):
        """Subscribe to event"""
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append((callback, args, kwargs))
    
    def emit(self, event_name: str, data: Any = None):
        """Emit event to all subscribers"""
        if event_name in self.subscribers:
            for callback, args, kwargs in self.subscribers[event_name]:
                try:
                    callback(data, *args, **kwargs)
                except Exception as e:
                    print(f"Error in event handler for {event_name}: {e}")


class ModuleRegistry:
    """Dynamic module loading and management"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.modules = {}
        self.config_path = config_path
        if config_path:
            self.load_from_config()
    
    def load_from_config(self):
        """Load modules from YAML config"""
        # TODO: Re-enable when yaml is available
        # with open(self.config_path, 'r') as f:
        #     config = yaml.safe_load(f)
        # 
        # for module_config in config.get('modules', []):
        #     self.load_module(module_config)
        pass
    
    def load_module(self, module_config: Dict):
        """Dynamically load a module"""
        name = module_config['name']
        path = module_config['path']
        
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Validate required interface
        if not hasattr(module, 'process'):
            raise ValueError(f"Module {name} missing required 'process' function")
        
        self.modules[name] = {
            'process': module.process,
            'config': module_config,
            'metadata': getattr(module, 'METADATA', {}),
            'module_ref': module
        }
    
    def get_module(self, name: str):
        """Get loaded module"""
        return self.modules.get(name)
    
    def list_modules(self) -> List[str]:
        """List available modules"""
        return list(self.modules.keys())


class GraphStore:
    """Optimized Graph Store with branching support"""
    
    def __init__(self, max_auto_states: int = 10, prune_old_states: bool = True, 
                 snapshot_interval: int = 50, enable_disk_cache: bool = False):
        # State management
        self.states = {}  # hash -> GraphState
        self.auto_states = deque(maxlen=max_auto_states)
        self.commits = {}  # hash -> commit info
        self.current_hash = None
        self.current_graph = Graph.empty(self)
        
        # Content pool for deduplication
        self.content_pool = ContentPool()
        
        # Optimization settings
        self.prune_old_states = prune_old_states
        self.snapshot_interval = snapshot_interval  # Create full snapshot every N states
        self.enable_disk_cache = enable_disk_cache
        self.state_count = 0
        
        # Caching
        self._reconstructed_cache = {}  # hash -> reconstructed Graph (weak refs)
        
        # Module system
        self.module_registry = ModuleRegistry()
        self.event_bus = EventBus()
        
        # Branching support
        self.branches = {}  # branch_name -> Branch
        self.current_branch = "main"
        self.branch_heads = {}  # branch_name -> current_hash
        
        # Initialize with empty state
        self._create_initial_state()
        
        # Initialize main branch
        self.branches["main"] = Branch(
            name="main",
            current_hash=self.current_hash,
            created_from=self.current_hash,
            description="Main branch"
        )
        self.branch_heads["main"] = self.current_hash
    
    def _compute_hash(self, graph_data: Dict) -> str:
        """Compute hash for graph state using fast hashing"""
        content = json.dumps(graph_data, sort_keys=True)
        return fast_hash(content)
    
    def _create_initial_state(self):
        """Create initial empty state as root"""
        initial_state = GraphState(
            hash="initial",
            nodes={},
            edges={},
            graph_attributes={},
            operation="initialize"
        )
        self.states["initial"] = initial_state
        self.current_hash = "initial"
        self.auto_states.append("initial")
    
    def _should_create_snapshot(self) -> bool:
        """Determine if we should create a full snapshot instead of delta"""
        # Always create snapshots for now to fix the reconstruction issue
        # TODO: Re-enable delta states once reconstruction is working properly
        return True
        # return (self.state_count % self.snapshot_interval == 0 or 
        #         len(self.auto_states) == 1)  # Always snapshot after initial
    
    def _create_snapshot_state(self, graph: Graph, operation: str) -> str:
        """Create full snapshot state"""
        # Store all nodes and edges in content pool
        node_hashes = {}
        edge_hashes = {}
        
        for node_id, node in graph.nodes.items():
            content_hash = self.content_pool.intern_node(node)
            node_hashes[node_id] = content_hash
        
        for edge_id, edge in graph.edges.items():
            content_hash = self.content_pool.intern_edge(edge)
            edge_hashes[edge_id] = content_hash
        
        state_data = {
            'nodes': {nid: {'id': graph.nodes[nid].id, 'attributes': graph.nodes[nid].attributes} 
                     for nid in graph.nodes},
            'edges': {eid: {'source': graph.edges[eid].source, 'target': graph.edges[eid].target, 
                           'attributes': graph.edges[eid].attributes} for eid in graph.edges},
            'graph_attributes': graph.graph_attributes
        }
        
        state_hash = self._compute_hash(state_data)
        
        if state_hash not in self.states:
            state = GraphState(
                hash=state_hash,
                nodes=node_hashes,
                edges=edge_hashes,
                graph_attributes=graph.graph_attributes.copy(),
                parent_hash=self.current_hash,
                operation=operation
            )
            self.states[state_hash] = state
        
        return state_hash
    
    def _create_compact_delta(self, old_graph: Graph, new_graph: Graph) -> CompactGraphDelta:
        """Create compact delta between two graphs"""
        delta = CompactGraphDelta()
        
        # Compare nodes
        old_nodes = set(old_graph.nodes.keys())
        new_nodes = set(new_graph.nodes.keys())
        
        # Added nodes
        for node_id in new_nodes - old_nodes:
            node = new_graph.nodes[node_id]
            content_hash = self.content_pool.intern_node(node)
            delta.added_nodes[node_id] = content_hash
        
        # Removed nodes
        delta.removed_nodes = old_nodes - new_nodes
        
        # Modified nodes
        for node_id in old_nodes & new_nodes:
            old_node = old_graph.nodes[node_id]
            new_node = new_graph.nodes[node_id]
            if old_node.attributes != new_node.attributes:
                content_hash = self.content_pool.intern_node(new_node)
                delta.modified_nodes[node_id] = content_hash
        
        # Compare edges (similar logic)
        old_edges = set(old_graph.edges.keys())
        new_edges = set(new_graph.edges.keys())
        
        for edge_id in new_edges - old_edges:
            edge = new_graph.edges[edge_id]
            content_hash = self.content_pool.intern_edge(edge)
            delta.added_edges[edge_id] = content_hash
        
        delta.removed_edges = old_edges - new_edges
        
        for edge_id in old_edges & new_edges:
            old_edge = old_graph.edges[edge_id]
            new_edge = new_graph.edges[edge_id]
            if old_edge.attributes != new_edge.attributes:
                content_hash = self.content_pool.intern_edge(new_edge)
                delta.modified_edges[edge_id] = content_hash
        
        # Graph attributes
        if old_graph.graph_attributes != new_graph.graph_attributes:
            delta.modified_graph_attrs = new_graph.graph_attributes
        
        return delta
    
    def _reconstruct_graph_from_state(self, state_hash: str) -> Graph:
        """Reconstruct graph from state (with caching)"""
        # Check cache first
        if state_hash in self._reconstructed_cache:
            cached_ref = self._reconstructed_cache[state_hash]()
            if cached_ref is not None:
                return cached_ref
        
        state = self.states[state_hash]
        
        # Always reconstruct from snapshot states for now
        # Reconstruct from content pool
        nodes = {}
        edges = {}
        
        for node_id, content_hash in (state.nodes or {}).items():
            node = self.content_pool.get_node(content_hash)
            if node:
                nodes[node_id] = node
        
        for edge_id, content_hash in (state.edges or {}).items():
            edge = self.content_pool.get_edge(content_hash)
            if edge:
                edges[edge_id] = edge
        
        graph = Graph(nodes, edges, state.graph_attributes or {}, self)
        
        # Cache with weak reference
        self._reconstructed_cache[state_hash] = weakref.ref(graph)
        return graph
    
    def _apply_delta_to_graph(self, base_graph: Graph, delta: CompactGraphDelta) -> Graph:
        """Apply compact delta to reconstruct graph"""
        # Handle None delta case
        if delta is None:
            return base_graph
            
        nodes = base_graph.nodes.copy()
        edges = base_graph.edges.copy()
        attrs = base_graph.graph_attributes.copy()
        
        # Apply node changes
        for node_id, content_hash in delta.added_nodes.items():
            node = self.content_pool.get_node(content_hash)
            if node:
                nodes[node_id] = node
        
        for node_id in delta.removed_nodes:
            nodes.pop(node_id, None)
            # Remove connected edges
            edges_to_remove = [eid for eid, edge in edges.items() 
                             if edge.source == node_id or edge.target == node_id]
            for eid in edges_to_remove:
                edges.pop(eid, None)
        
        for node_id, content_hash in delta.modified_nodes.items():
            node = self.content_pool.get_node(content_hash)
            if node and node_id in nodes:
                nodes[node_id] = node
        
        # Apply edge changes
        for edge_id, content_hash in delta.added_edges.items():
            edge = self.content_pool.get_edge(content_hash)
            if edge:
                edges[edge_id] = edge
        
        for edge_id in delta.removed_edges:
            edges.pop(edge_id, None)
        
        for edge_id, content_hash in delta.modified_edges.items():
            edge = self.content_pool.get_edge(content_hash)
            if edge and edge_id in edges:
                edges[edge_id] = edge
        
        # Apply attribute changes
        attrs.update(delta.modified_graph_attrs)
        
        return Graph(nodes, edges, attrs, self)
    
    def get_current_graph(self) -> Graph:
        """Get current graph state with lazy reconstruction"""
        if self.current_hash in self.states:
            return self._reconstruct_graph_from_state(self.current_hash)
        return self.current_graph
    
    def _create_state(self, graph: Graph, operation: str) -> str:
        """Create new state from graph (legacy method for compatibility)"""
        return self._create_snapshot_state(graph, operation)
    
    def create_branch(self, branch_name: str, from_hash: str = None, 
                     description: str = "", from_subgraph: Graph = None) -> str:
        """Create a new branch from specified hash or current state"""
        if branch_name in self.branches:
            raise ValueError(f"Branch '{branch_name}' already exists")
        
        # Determine starting point
        if from_subgraph:
            # Create branch from subgraph
            base_hash = self._create_subgraph_snapshot(from_subgraph, f"branch_{branch_name}_base")
            is_subgraph = True
            subgraph_filter = from_subgraph.subgraph_metadata
            parent_branch = from_subgraph.branch_name or self.current_branch
        else:
            base_hash = from_hash or self.current_hash
            is_subgraph = False
            subgraph_filter = None
            parent_branch = None
        
        # Verify base hash exists
        if base_hash not in self.states:
            raise ValueError(f"Hash '{base_hash}' not found")
        
        # Create branch
        branch = Branch(
            name=branch_name,
            current_hash=base_hash,
            created_from=base_hash,
            description=description,
            is_subgraph=is_subgraph,
            subgraph_filter=subgraph_filter,
            parent_branch=parent_branch
        )
        
        self.branches[branch_name] = branch
        self.branch_heads[branch_name] = base_hash
        
        # Emit event
        self.event_bus.emit('branch_created', {
            'branch_name': branch_name,
            'from_hash': base_hash,
            'is_subgraph': is_subgraph
        })
        
        return branch_name
    
    def _create_subgraph_snapshot(self, subgraph: Graph, operation: str) -> str:
        """Create a snapshot specifically for subgraph"""
        # Force snapshot to apply any pending changes
        snapshot_graph = subgraph.snapshot()
        
        # Store nodes and edges in content pool
        node_hashes = {}
        edge_hashes = {}
        
        for node_id, node in snapshot_graph.nodes.items():
            content_hash = self.content_pool.intern_node(node)
            node_hashes[node_id] = content_hash
        
        for edge_id, edge in snapshot_graph.edges.items():
            content_hash = self.content_pool.intern_edge(edge)
            edge_hashes[edge_id] = content_hash
        
        # Create state data for hashing
        state_data = {
            'nodes': {nid: {'id': snapshot_graph.nodes[nid].id, 'attributes': snapshot_graph.nodes[nid].attributes} 
                     for nid in snapshot_graph.nodes},
            'edges': {eid: {'source': snapshot_graph.edges[eid].source, 'target': snapshot_graph.edges[eid].target, 
                           'attributes': snapshot_graph.edges[eid].attributes} for eid in snapshot_graph.edges},
            'graph_attributes': snapshot_graph.graph_attributes,
            'subgraph_metadata': snapshot_graph.subgraph_metadata
        }
        
        state_hash = self._compute_hash(state_data)
        
        if state_hash not in self.states:
            state = GraphState(
                hash=state_hash,
                nodes=node_hashes,
                edges=edge_hashes,
                graph_attributes=snapshot_graph.graph_attributes.copy(),
                parent_hash=self.current_hash,
                operation=operation,
                is_subgraph_state=True,
                subgraph_metadata=snapshot_graph.subgraph_metadata
            )
            self.states[state_hash] = state
        
        return state_hash
    
    def switch_branch(self, branch_name: str) -> Graph:
        """Switch to specified branch"""
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist")
        
        old_branch = self.current_branch
        self.current_branch = branch_name
        branch = self.branches[branch_name]
        
        # Update current state
        self.current_hash = branch.current_hash
        self.current_graph = self._reconstruct_graph_from_state(self.current_hash)
        self.current_graph.branch_name = branch_name
        
        # Emit event
        self.event_bus.emit('branch_switched', {
            'from_branch': old_branch,
            'to_branch': branch_name,
            'hash': self.current_hash
        })
        
        return self.current_graph
    
    def merge_branch(self, source_branch: str, target_branch: str = None, 
                    strategy: str = "auto", message: str = "") -> str:
        """Merge source branch into target branch"""
        if source_branch not in self.branches:
            raise ValueError(f"Source branch '{source_branch}' does not exist")
        
        target_branch = target_branch or self.current_branch
        if target_branch not in self.branches:
            raise ValueError(f"Target branch '{target_branch}' does not exist")
        
        source_graph = self._reconstruct_graph_from_state(self.branches[source_branch].current_hash)
        target_graph = self._reconstruct_graph_from_state(self.branches[target_branch].current_hash)
        
        # Handle subgraph merging
        if self.branches[source_branch].is_subgraph:
            merged_graph = self._merge_subgraph(source_graph, target_graph, strategy)
        else:
            merged_graph = self._merge_graphs(source_graph, target_graph, strategy)
        
        # Switch to target branch and update
        old_branch = self.current_branch
        self.switch_branch(target_branch)
        merge_hash = self.update_graph(merged_graph, f"merge_{source_branch}_into_{target_branch}")
        
        # Update branch head
        self.branches[target_branch].current_hash = merge_hash
        self.branch_heads[target_branch] = merge_hash
        
        # Commit the merge
        commit_message = message or f"Merge branch '{source_branch}' into '{target_branch}'"
        self.commit(commit_message)
        
        # Emit event
        self.event_bus.emit('branch_merged', {
            'source_branch': source_branch,
            'target_branch': target_branch,
            'merge_hash': merge_hash,
            'strategy': strategy
        })
        
        # Restore original branch if different
        if old_branch != target_branch:
            self.switch_branch(old_branch)
        
        return merge_hash
    
    def _merge_graphs(self, source: Graph, target: Graph, strategy: str) -> Graph:
        """Merge two full graphs"""
        if strategy == "auto":
            # Simple union merge - combine all nodes and edges
            merged = Graph(target.nodes.copy(), target.edges.copy(), 
                          target.graph_attributes.copy(), self)
            
            # Add nodes from source
            for node_id, node in source.nodes.items():
                if node_id not in merged.nodes:
                    merged = merged.add_node(node_id, **node.attributes)
                else:
                    # Merge attributes (source takes precedence)
                    for key, value in node.attributes.items():
                        merged = merged.set_node_attribute(node_id, key, value)
            
            # Add edges from source
            for edge_id, edge in source.edges.items():
                if edge_id not in merged.edges:
                    merged = merged.add_edge(edge.source, edge.target, **edge.attributes)
                else:
                    # Merge attributes (source takes precedence)
                    for key, value in edge.attributes.items():
                        merged = merged.set_edge_attribute(edge.source, edge.target, key, value)
            
            return merged.snapshot()
        
        else:
            raise ValueError(f"Merge strategy '{strategy}' not implemented")
    
    def _merge_subgraph(self, subgraph: Graph, target: Graph, strategy: str) -> Graph:
        """Merge a subgraph back into the target graph"""
        if strategy == "auto":
            # Update target graph with subgraph changes
            merged = Graph(target.nodes.copy(), target.edges.copy(), 
                          target.graph_attributes.copy(), self)
            
            # Update nodes that exist in subgraph
            for node_id, node in subgraph.nodes.items():
                if node_id in merged.nodes:
                    # Update existing node with subgraph changes
                    for key, value in node.attributes.items():
                        merged = merged.set_node_attribute(node_id, key, value)
                else:
                    # Add new node from subgraph
                    merged = merged.add_node(node_id, **node.attributes)
            
            # Update edges that exist in subgraph
            for edge_id, edge in subgraph.edges.items():
                if edge_id in merged.edges:
                    # Update existing edge
                    for key, value in edge.attributes.items():
                        merged = merged.set_edge_attribute(edge.source, edge.target, key, value)
                else:
                    # Add new edge from subgraph
                    merged = merged.add_edge(edge.source, edge.target, **edge.attributes)
            
            return merged.snapshot()
        
        else:
            raise ValueError(f"Subgraph merge strategy '{strategy}' not implemented")
    
    def delete_branch(self, branch_name: str, force: bool = False):
        """Delete a branch"""
        if branch_name == "main":
            raise ValueError("Cannot delete main branch")
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist")
        
        if branch_name == self.current_branch and not force:
            raise ValueError(f"Cannot delete current branch '{branch_name}' without force=True")
        
        # Switch to main if deleting current branch
        if branch_name == self.current_branch:
            self.switch_branch("main")
        
        # Remove branch
        del self.branches[branch_name]
        del self.branch_heads[branch_name]
        
        # Emit event
        self.event_bus.emit('branch_deleted', {'branch_name': branch_name})
    
    def list_branches(self) -> List[Dict[str, Any]]:
        """List all branches with metadata"""
        branches = []
        for name, branch in self.branches.items():
            state = self.states.get(branch.current_hash)
            branches.append({
                'name': name,
                'current': name == self.current_branch,
                'hash': branch.current_hash,
                'created_from': branch.created_from,
                'created_at': branch.created_at,
                'description': branch.description,
                'is_subgraph': branch.is_subgraph,
                'parent_branch': branch.parent_branch,
                'last_operation': state.operation if state else None,
                'last_modified': state.timestamp if state else None
            })
        return branches
    
    def get_branch_diff(self, branch1: str, branch2: str) -> Dict[str, Any]:
        """Get differences between two branches"""
        if branch1 not in self.branches or branch2 not in self.branches:
            raise ValueError("One or both branches do not exist")
        
        graph1 = self._reconstruct_graph_from_state(self.branches[branch1].current_hash)
        graph2 = self._reconstruct_graph_from_state(self.branches[branch2].current_hash)
        
        diff = {
            'nodes_only_in_branch1': set(graph1.nodes.keys()) - set(graph2.nodes.keys()),
            'nodes_only_in_branch2': set(graph2.nodes.keys()) - set(graph1.nodes.keys()),
            'edges_only_in_branch1': set(graph1.edges.keys()) - set(graph2.edges.keys()),
            'edges_only_in_branch2': set(graph2.edges.keys()) - set(graph1.edges.keys()),
            'modified_nodes': {},
            'modified_edges': {}
        }
        
        # Check for modified nodes
        common_nodes = set(graph1.nodes.keys()) & set(graph2.nodes.keys())
        for node_id in common_nodes:
            if graph1.nodes[node_id].attributes != graph2.nodes[node_id].attributes:
                diff['modified_nodes'][node_id] = {
                    'branch1': graph1.nodes[node_id].attributes,
                    'branch2': graph2.nodes[node_id].attributes
                }
        
        # Check for modified edges
        common_edges = set(graph1.edges.keys()) & set(graph2.edges.keys())
        for edge_id in common_edges:
            if graph1.edges[edge_id].attributes != graph2.edges[edge_id].attributes:
                diff['modified_edges'][edge_id] = {
                    'branch1': graph1.edges[edge_id].attributes,
                    'branch2': graph2.edges[edge_id].attributes
                }
        
        return diff
    
    def update_graph(self, new_graph: Graph, operation: str = "update"):
        """Update current graph state and branch head"""
        # Force snapshot to apply any pending changes
        snapshot_graph = new_graph.snapshot()
        
        old_graph = self.get_current_graph()
        
        # Decide whether to create snapshot or delta
        if self._should_create_snapshot():
            new_hash = self._create_snapshot_state(snapshot_graph, operation)
        else:
            new_hash = self._create_delta_state(old_graph, snapshot_graph, operation)
        
        # Update current state
        self.current_hash = new_hash
        self.auto_states.append(new_hash)
        self.current_graph = snapshot_graph
        self.state_count += 1
        
        # Update branch head
        if self.current_branch in self.branches:
            self.branches[self.current_branch].current_hash = new_hash
            self.branch_heads[self.current_branch] = new_hash
        
        # Prune old states if enabled
        if self.prune_old_states and len(self.auto_states) > self.auto_states.maxlen // 2:
            self._prune_old_states()
        
        # Emit event
        self.event_bus.emit('graph_updated', {
            'operation': operation,
            'hash': new_hash
        })
        
        return new_hash
    
    def _prune_old_states(self):
        """Remove old states to free memory"""
        # Keep committed states and recent states
        keep_hashes = set(self.commits.keys())
        keep_hashes.update(list(self.auto_states)[-self.auto_states.maxlen//2:])
        
        # Remove old states
        to_remove = [h for h in self.states.keys() if h not in keep_hashes and h != "initial"]
        
        for state_hash in to_remove:
            state = self.states.pop(state_hash, None)
            if state and state.delta:
                # Release content pool references
                for content_hash in state.delta.added_nodes.values():
                    self.content_pool.release_node(content_hash)
                for content_hash in state.delta.modified_nodes.values():
                    self.content_pool.release_node(content_hash)
                for content_hash in state.delta.added_edges.values():
                    self.content_pool.release_edge(content_hash)
                for content_hash in state.delta.modified_edges.values():
                    self.content_pool.release_edge(content_hash)
            elif state and state.nodes:
                # Release content pool references for snapshot states
                for content_hash in state.nodes.values():
                    self.content_pool.release_node(content_hash)
                for content_hash in state.edges.values():
                    self.content_pool.release_edge(content_hash)
            
            # Clear cache
            self._reconstructed_cache.pop(state_hash, None)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        return {
            'total_states': len(self.states),
            'pooled_nodes': len(self.content_pool.nodes),
            'pooled_edges': len(self.content_pool.edges),
            'cached_reconstructions': len(self._reconstructed_cache),
            'auto_states_length': len(self.auto_states),
            'committed_states': len(self.commits)
        }
    
    def commit(self, message: str = "") -> str:
        """Explicitly commit current state"""
        if self.current_hash:
            self.commits[self.current_hash] = {
                'message': message,
                'timestamp': time.time()
            }
            self.event_bus.emit('graph_committed', {
                'hash': self.current_hash,
                'message': message
            })
        return self.current_hash
    
    def undo(self) -> Graph:
        """Undo to previous state"""
        if len(self.auto_states) > 1:
            current_idx = list(self.auto_states).index(self.current_hash)
            if current_idx > 0:
                prev_hash = self.auto_states[current_idx - 1]
                self.current_hash = prev_hash
                self.current_graph = self.get_current_graph()
                return self.current_graph
        return self.current_graph
    
    def get_history(self, commits_only: bool = False) -> List[Dict]:
        """Get state history"""
        if commits_only:
            return [{'hash': h, 'info': info} for h, info in self.commits.items()]
        else:
            return [{'hash': h, 'state': self.states[h]} for h in self.auto_states if h in self.states]
    
    def run_module(self, module_name: str, **params) -> Graph:
        """Execute a module on current graph"""
        module = self.module_registry.get_module(module_name)
        if not module:
            raise ValueError(f"Module {module_name} not found")
        
        current_graph = self.get_current_graph()
        new_graph = module['process'](current_graph, **params)
        
        # Update state
        self.update_graph(new_graph, f"run_{module_name}")
        
        return new_graph
    
    def load_module_config(self, config_path: str):
        """Load modules from config file"""
        self.module_registry = ModuleRegistry(config_path)
        
        # Set up auto-triggers
        for module_name, module_info in self.module_registry.modules.items():
            triggers = module_info['config'].get('triggers', [])
            for trigger in triggers:
                self.event_bus.subscribe(trigger, self._auto_run_module, module_name)
    
    def _auto_run_module(self, event_data, module_name):
        """Auto-run module on event trigger"""
        try:
            self.run_module(module_name)
        except Exception as e:
            print(f"Error auto-running module {module_name}: {e}")


# Optimized convenience function
def create_random_graph(n_nodes: int = 10, edge_probability: float = 0.3) -> Graph:
    """Create a random graph efficiently using vectorized operations"""
    import random
    
    # Create node IDs vectorized
    node_ids = [f"node_{i}" for i in range(n_nodes)]
    
    # Create edges vectorized
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_probability:
                edges.append((f"node_{i}", f"node_{j}"))
    
    # Use fast constructor
    graph = Graph.from_edge_list(edges)
    return graph.snapshot()


# Add convenience functions for branching workflows
def create_clustering_workflow(store: GraphStore, graph: Graph, 
                              algorithms: List[str] = None) -> List[str]:
    """Create branches for different clustering algorithms"""
    algorithms = algorithms or ['kmeans', 'spectral', 'hierarchical']
    branches = []
    
    for algo in algorithms:
        branch_name = f"clustering_{algo}"
        try:
            store.create_branch(branch_name, description=f"Clustering with {algo}")
            branches.append(branch_name)
        except ValueError:
            # Branch already exists
            pass
    
    return branches


def create_subgraph_branch(store: GraphStore, subgraph: Graph, 
                          branch_name: str, description: str = "") -> str:
    """Create a branch from a subgraph for isolated processing"""
    return store.create_branch(
        branch_name, 
        from_subgraph=subgraph,
        description=description or f"Subgraph branch: {branch_name}"
    )


# ...existing code for main section...
if __name__ == "__main__":
    # Example usage
    store = GraphStore()
    
    # Create a simple graph
    graph = create_random_graph(5, 0.4)
    store.update_graph(graph, "create_random_graph")
    
    # Commit this state
    store.commit("Initial random graph")
    
    # Add a node
    graph = store.get_current_graph()
    graph = graph.add_node("new_node", color="red")
    store.update_graph(graph, "add_node")
    
    # Show history
    print("History:")
    for entry in store.get_history():
        print(f"  {entry['hash']}: {entry['state'].operation}")
    
    print(f"\nCurrent graph has {len(store.get_current_graph().nodes)} nodes")