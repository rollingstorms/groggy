"""
High-performance Graph implementation with Rust backend
"""

from typing import Dict, List, Any, Optional, Callable, Union
from .data_structures import Node, Edge
from .views import LazyDict, NodeView, EdgeView

# Import detection for backend
try:
    from . import _core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


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
    
    def add_node(self, node_id: str = None, **attributes):
        """Queue node for batch addition"""
        if node_id is None:
            # Generate ID like the main add_node method
            import uuid
            node_id = f"node_{uuid.uuid4().hex[:8]}"
        
        if node_id not in self.graph.nodes and node_id not in self.batch_nodes:
            self.batch_nodes[node_id] = Node(node_id, attributes)
        
        return node_id
    
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
        
        return edge_id


class Graph:
    """High-level Graph interface with automatic Rust/Python backend selection"""
    
    def __init__(self, nodes=None, edges=None, graph_attributes=None, graph_store=None, backend=None):
        # Backend selection - use global setting or override
        if backend is None:
            # Import here to avoid circular import
            from . import get_current_backend
            backend = get_current_backend()
        
        self.backend = backend
        self.use_rust = (backend == 'rust')
        
        if self.use_rust:
            self._rust_core = _core.FastGraph()
            self._init_rust_backend(nodes, edges, graph_attributes)
        else:
            self._init_python_backend(nodes, edges, graph_attributes)
        
        self.graph_store = graph_store
        self.graph_attributes = graph_attributes or {}
        self._current_time = 0
        
        # Branch and subgraph metadata
        self.branch_name: Optional[str] = None
        self.is_subgraph = False
        self.subgraph_metadata = {}
    
    def _init_rust_backend(self, nodes, edges, graph_attributes):
        """Initialize Rust backend"""
        if nodes:
            node_data = [(node_id, node.attributes) for node_id, node in nodes.items()]
            self._rust_core.batch_add_nodes(node_data)
        
        if edges:
            edge_data = [(edge.source, edge.target, edge.attributes) for edge in edges.values()]
            self._rust_core.batch_add_edges(edge_data)
    
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
    def empty(cls, graph_store=None, backend=None):
        """Create empty graph"""
        return cls(graph_store=graph_store, backend=backend)
    
    @classmethod
    def from_node_list(cls, node_ids: List[str], node_attrs: Dict[str, List[Any]] = None, 
                      graph_store=None, backend=None):
        """Create graph from vectorized node data (NumPy-style)"""
        if backend == 'rust' or (backend is None and RUST_AVAILABLE):
            # Use Rust backend
            graph = cls.empty(graph_store, backend='rust')
            
            if node_attrs:
                node_data = []
                for i, node_id in enumerate(node_ids):
                    attrs = {attr_name: attr_values[i] if i < len(attr_values) else None
                            for attr_name, attr_values in node_attrs.items()}
                    node_data.append((node_id, attrs))
                graph._rust_core.batch_add_nodes(node_data)
            else:
                node_data = [(node_id, {}) for node_id in node_ids]
                graph._rust_core.batch_add_nodes(node_data)
            
            return graph
        else:
            # Use Python backend
            nodes = {}
            
            if node_attrs:
                for i, node_id in enumerate(node_ids):
                    attrs = {attr_name: attr_values[i] if i < len(attr_values) else None
                            for attr_name, attr_values in node_attrs.items()}
                    nodes[node_id] = Node(node_id, attrs)
            else:
                nodes = {node_id: Node(node_id) for node_id in node_ids}
            
            return cls(nodes, {}, {}, graph_store)
    
    @classmethod
    def from_edge_list(cls, edges: List[tuple], node_attrs: Dict[str, List[Any]] = None, 
                      edge_attrs: Dict[str, List[Any]] = None, graph_store=None, backend=None):
        """Create graph from edge list (NetworkX-style)"""
        if backend == 'rust' or (backend is None and RUST_AVAILABLE):
            # Use Rust backend
            graph = cls.empty(graph_store, backend='rust')
            
            # Prepare edge data
            edge_data = []
            for i, edge in enumerate(edges):
                source, target = edge[0], edge[1]
                if edge_attrs:
                    attrs = {attr_name: attr_values[i] if i < len(attr_values) else None
                            for attr_name, attr_values in edge_attrs.items()}
                else:
                    attrs = {}
                edge_data.append((source, target, attrs))
            
            graph._rust_core.batch_add_edges(edge_data)
            return graph
        else:
            # Use Python backend
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
    
    @property 
    def nodes(self):
        """Get nodes view"""
        if self.use_rust:
            return NodeView(self._rust_core)
        else:
            effective_nodes, _, _ = self._get_effective_data()
            return effective_nodes
    
    @property
    def edges(self):
        """Get edges view"""  
        if self.use_rust:
            return EdgeView(self._rust_core)
        else:
            _, effective_edges, _ = self._get_effective_data()
            return effective_edges
    
    def add_node(self, node_id: str = None, **attributes) -> str:
        """Add node with optimized checking. Returns the node_id."""
        import uuid
        
        # Auto-generate node_id if not provided
        if node_id is None:
            node_id = f"node_{uuid.uuid4().hex[:8]}"
        
        if self.use_rust:
            # Pass attributes to Rust backend
            if attributes:
                self._rust_core.add_node(node_id, attributes)
            else:
                self._rust_core.add_node(node_id, None)
            return node_id
        else:
            # Python implementation (fallback)
            # Quick check without effective data computation for performance
            if not self._is_modified and node_id in self._nodes:
                return node_id
            
            # Initialize delta first, then check
            self._init_delta()
            
            # Check in pending delta
            if node_id in self._pending_delta.added_nodes:
                return node_id
            
            new_node = Node(node_id, attributes)
            self._pending_delta.added_nodes[node_id] = new_node
            
            # Try incremental cache update instead of invalidation
            self._update_cache_for_node_add(node_id, new_node)
            
            return node_id
    
    def add_edge(self, source: str, target: str, **attributes) -> str:
        """Add edge with optimized node creation. Returns the edge_id."""
        if self.use_rust:
            # Pass attributes to Rust backend
            if attributes:
                self._rust_core.add_edge(source, target, attributes)
            else:
                self._rust_core.add_edge(source, target, None)
            return f"{source}->{target}"
        else:
            # Python implementation (fallback)
            edge_id = f"{source}->{target}"
            
            # Quick duplicate check
            if not self._is_modified and edge_id in self._edges:
                return edge_id
            
            # Initialize delta first
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
    
    def create_subgraph(self, node_filter: Callable[[Node], bool] = None, 
                       edge_filter: Callable[[Edge], bool] = None,
                       node_ids: set = None, include_edges: bool = True) -> 'Graph':
        """Create a subgraph based on filters or node IDs"""
        if self.use_rust:
            # For Rust backend, we need to implement subgraph filtering in Python
            # since the Rust backend doesn't have subgraph methods yet
            subgraph = Graph.empty(self.graph_store, backend='rust')
            
            # Get all nodes and filter them
            all_node_ids = self.get_node_ids()
            filtered_node_ids = []
            
            for node_id in all_node_ids:
                include_node = False
                
                if node_ids and node_id in node_ids:
                    include_node = True
                elif node_filter:
                    node = self.get_node(node_id)
                    include_node = node_filter(node)
                elif not node_ids and not node_filter:
                    include_node = True
                    
                if include_node:
                    filtered_node_ids.append(node_id)
            
            # Add filtered nodes to subgraph
            for node_id in filtered_node_ids:
                node = self.get_node(node_id)
                # Copy the node with all its attributes
                new_node_id = subgraph.add_node(**dict(node.attributes))
                # Keep track of mapping between old and new node IDs
                if not hasattr(subgraph, '_node_id_mapping'):
                    subgraph._node_id_mapping = {}
                subgraph._node_id_mapping[node_id] = new_node_id
            
            # Add edges between included nodes
            if include_edges and hasattr(subgraph, '_node_id_mapping'):
                for edge_id in self.edges:
                    edge = self.get_edge(edge_id)
                    if edge.source in filtered_node_ids and edge.target in filtered_node_ids:
                        include_edge = True
                        if edge_filter:
                            include_edge = edge_filter(edge)
                        
                        if include_edge:
                            # Use mapped node IDs for the subgraph
                            source_mapped = subgraph._node_id_mapping[edge.source]
                            target_mapped = subgraph._node_id_mapping[edge.target]
                            subgraph.add_edge(source_mapped, target_mapped, **dict(edge.attributes))
            
            return subgraph
        else:
            # Python implementation
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
                'created_at': __import__('time').time(),
                'filter_type': 'node_ids' if node_ids else 'function' if node_filter else 'all'
            }
            
            return subgraph
    
    def get_connected_component(self, start_node_id: str) -> 'Graph':
        """Get connected component containing the specified node"""
        if self.use_rust:
            # Implement connected component traversal for Rust backend
            if start_node_id not in self.get_node_ids():
                return Graph.empty(self.graph_store, backend='rust')
            
            visited = set()
            to_visit = [start_node_id]
            component_nodes = []
            
            while to_visit:
                current = to_visit.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                component_nodes.append(current)
                
                # Add unvisited neighbors
                neighbors = self.get_neighbors(current)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        to_visit.append(neighbor)
            
            # Create subgraph with connected component
            return self.create_subgraph(node_ids=set(component_nodes))
        else:
            # Python implementation
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
    
    def snapshot(self) -> 'Graph':
        """Create optimized snapshot"""
        if self.use_rust:
            # Rust backend handles immutability automatically
            return self
        else:
            # Python implementation
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
    
    # Python backend methods (only used when Rust is not available)
    def _init_delta(self):
        """Initialize delta tracking for copy-on-write - now truly lazy"""
        if hasattr(self, '_is_modified') and self._is_modified:
            return
            
        self._is_modified = True
        from .delta import GraphDelta
        self._pending_delta = GraphDelta()
        self._invalidate_cache()
    
    def _invalidate_cache(self):
        """Invalidate effective data cache"""
        self._effective_cache = None
        self._cache_valid = False
    
    def _update_cache_for_node_add(self, node_id: str, node: Node):
        """Incrementally update cache when adding node"""
        if hasattr(self, '_cache_valid') and self._cache_valid and self._effective_cache:
            effective_nodes, effective_edges, effective_attrs = self._effective_cache
            if hasattr(effective_nodes, 'added'):
                # LazyDict - just add to added dict
                effective_nodes.added[node_id] = node
            else:
                # Regular dict - invalidate to be safe
                self._invalidate_cache()
    
    def _update_cache_for_edge_add(self, edge_id: str, edge: Edge):
        """Incrementally update cache when adding edge"""
        if hasattr(self, '_cache_valid') and self._cache_valid and self._effective_cache:
            effective_nodes, effective_edges, effective_attrs = self._effective_cache
            if hasattr(effective_edges, 'added'):
                # LazyDict - just add to added dict
                effective_edges.added[edge_id] = edge
            else:
                # Regular dict - invalidate to be safe
                self._invalidate_cache()
    
    def _get_effective_data(self):
        """Get effective nodes/edges including pending changes - zero-copy lazy views"""
        if not hasattr(self, '_cache_valid'):
            return self._nodes, self._edges, self.graph_attributes
            
        if self._cache_valid and self._effective_cache is not None:
            return self._effective_cache
        
        if not self._pending_delta:
            result = (self._nodes, self._edges, self.graph_attributes)
            self._effective_cache = result
            self._cache_valid = True
            return result
        
        delta = self._pending_delta
        
        # Create lazy views instead of copying - dramatically faster!
        effective_nodes = LazyDict(
            self._nodes, 
            delta.added_nodes, 
            delta.removed_nodes, 
            delta.modified_nodes
        )
        
        # For edges, we need to handle node removal edge case
        edges_to_remove = set()
        if delta.removed_nodes:
            # Only compute this if we actually have removed nodes
            for edge_id, edge in self._edges.items():
                if edge.source in delta.removed_nodes or edge.target in delta.removed_nodes:
                    edges_to_remove.add(edge_id)
        
        # Combine removed edges with edges removed due to node removal
        all_removed_edges = delta.removed_edges | edges_to_remove
        
        effective_edges = LazyDict(
            self._edges,
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
    
    def _apply_batch_operations(self, batch_nodes, batch_edges):
        """Apply batched operations efficiently"""
        if not batch_nodes and not batch_edges:
            return
        
        if self.use_rust:
            # For Rust backend, apply operations directly
            for node_id, node in batch_nodes.items():
                if node_id not in self.nodes:
                    self._rust_core.add_node(node_id, node.attributes)
            
            for edge_id, edge in batch_edges.items():
                if edge_id not in self.edges:
                    self._rust_core.add_edge(edge.source, edge.target, edge.attributes)
        else:
            # Python backend implementation
            if not hasattr(self, '_init_delta'):
                return
            
            self._init_delta()
            
            # Add all nodes
            for node_id, node in batch_nodes.items():
                if node_id not in self._nodes:
                    self._pending_delta.added_nodes[node_id] = node
            
            # Add all edges  
            for edge_id, edge in batch_edges.items():
                if edge_id not in self._edges:
                    self._pending_delta.added_edges[edge_id] = edge
            
            self._invalidate_cache()
    
    def _apply_pending_changes(self):
        """Apply pending delta changes"""
        if not hasattr(self, '_pending_delta') or not self._pending_delta:
            return
            
        delta = self._pending_delta
        
        # Apply node changes
        for node_id, node in delta.added_nodes.items():
            self._nodes[node_id] = node
            if hasattr(self, 'node_order') and node_id not in self.node_order:
                self.node_order[node_id] = self._next_time()
        
        # Apply other delta operations...
        self._pending_delta = None
    
    def _next_time(self):
        self._current_time += 1
        return self._current_time
    
    def _get_effective_nodes(self):
        """Get effective nodes including pending changes"""
        effective_nodes, _, _ = self._get_effective_data()
        return effective_nodes
    
    def _get_effective_edges(self):
        """Get effective edges including pending changes"""
        _, effective_edges, _ = self._get_effective_data()
        return effective_edges
    
    def get_node_ids(self) -> List[str]:
        """Get all node IDs in the graph"""
        if self.use_rust:
            return self._rust_core.get_node_ids()
        else:
            # Python implementation
            effective_nodes = self._get_effective_nodes()
            return list(effective_nodes.keys())
    
    def node_count(self) -> int:
        """Get the number of nodes in the graph"""
        if self.use_rust:
            return self._rust_core.node_count()
        else:
            # Python implementation
            effective_nodes = self._get_effective_nodes()
            return len(effective_nodes)
    
    def edge_count(self) -> int:
        """Get the number of edges in the graph"""
        if self.use_rust:
            return self._rust_core.edge_count()
        else:
            # Python implementation
            effective_edges = self._get_effective_edges()
            return len(effective_edges)
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node"""
        if self.use_rust:
            return self._rust_core.get_neighbors(node_id)
        else:
            # Python implementation
            neighbors = []
            effective_edges = self._get_effective_edges()
            for edge in effective_edges.values():
                if edge.source == node_id:
                    neighbors.append(edge.target)
                elif edge.target == node_id:
                    neighbors.append(edge.source)
            return neighbors
    
    def get_node(self, node_id: str) -> Node:
        """Get node by ID"""
        if self.use_rust:
            try:
                attrs = self._rust_core.get_node_attributes(node_id)
                return Node(node_id, attrs or {})
            except:
                raise KeyError(f"Node {node_id} not found")
        else:
            # Python implementation
            effective_nodes = self._get_effective_nodes()
            if node_id not in effective_nodes:
                raise KeyError(f"Node {node_id} not found")
            return effective_nodes[node_id]
    
    def get_edge(self, edge_id: str) -> Edge:
        """Get edge by ID"""
        if self.use_rust:
            # For now, edges are stored as "source->target" format
            parts = edge_id.split('->')
            if len(parts) != 2:
                raise KeyError(f"Invalid edge ID format: {edge_id}")
            source, target = parts
            try:
                attrs = self._rust_core.get_edge_attributes(source, target)
                return Edge(source, target, attrs or {})
            except:
                raise KeyError(f"Edge {edge_id} not found")
        else:
            # Python implementation
            effective_edges = self._get_effective_edges()
            if edge_id not in effective_edges:
                raise KeyError(f"Edge {edge_id} not found")
            return effective_edges[edge_id]
    
    def set_node_attribute(self, node_id: str, key: str, value: Any) -> 'Graph':
        """Set a node attribute"""
        if self.use_rust:
            try:
                self._rust_core.set_node_attribute(node_id, key, value)
                return self
            except:
                raise KeyError(f"Node {node_id} not found")
        else:
            # Python implementation
            effective_nodes = self._get_effective_nodes()
            if node_id not in effective_nodes:
                raise KeyError(f"Node {node_id} not found")
            
            # Get current node and create modified version
            current_node = effective_nodes[node_id]
            new_attributes = dict(current_node.attributes)
            new_attributes[key] = value
            
            # Add to delta
            self._init_delta()
            self._pending_delta.modified_nodes[node_id] = Node(node_id, new_attributes)
            self._invalidate_cache()
            
            return self
    
    def set_edge_attribute(self, edge_id: str, key: str, value: Any) -> 'Graph':
        """Set an edge attribute"""
        if self.use_rust:
            parts = edge_id.split('->')
            if len(parts) != 2:
                raise KeyError(f"Invalid edge ID format: {edge_id}")
            source, target = parts
            try:
                self._rust_core.set_edge_attribute(source, target, key, value)
                return self
            except:
                raise KeyError(f"Edge {edge_id} not found")
        else:
            # Python implementation
            effective_edges = self._get_effective_edges()
            if edge_id not in effective_edges:
                raise KeyError(f"Edge {edge_id} not found")
            
            # Get current edge and create modified version
            current_edge = effective_edges[edge_id]
            new_attributes = dict(current_edge.attributes)
            new_attributes[key] = value
            
            # Add to delta
            self._init_delta()
            self._pending_delta.modified_edges[edge_id] = Edge(current_edge.source, current_edge.target, new_attributes)
            self._invalidate_cache()
            
            return self

    # ========== OPTIMIZED BATCH FILTERING METHODS ==========
    
    def batch_filter_nodes(self, **filters) -> List[str]:
        """
        Efficiently filter nodes by attributes using batch operations.
        
        Args:
            **filters: Attribute key-value pairs to filter by
            
        Returns:
            List of node IDs matching all filters
            
        Example:
            # Find all people over 25 in Chicago
            node_ids = g.batch_filter_nodes(age=25, city="Chicago")
        """
        if self.use_rust:
            # Use optimized Rust backend filtering
            return self._rust_core.batch_filter_nodes_by_attributes(filters)
        else:
            # Python fallback with some optimization
            effective_nodes = self._get_effective_nodes()
            results = []
            
            for node_id, node in effective_nodes.items():
                match = True
                for key, expected_value in filters.items():
                    if key not in node.attributes or node.attributes[key] != expected_value:
                        match = False
                        break
                if match:
                    results.append(node_id)
            
            return results
    
    def batch_filter_edges(self, **filters) -> List[tuple]:
        """
        Efficiently filter edges by attributes using batch operations.
        
        Args:
            **filters: Attribute key-value pairs to filter by
            
        Returns:
            List of (source, target) tuples for edges matching all filters
        """
        if self.use_rust:
            return self._rust_core.batch_filter_edges_by_attributes(filters)
        else:
            effective_edges = self._get_effective_edges()
            results = []
            
            for edge_id, edge in effective_edges.items():
                match = True
                for key, expected_value in filters.items():
                    if key not in edge.attributes or edge.attributes[key] != expected_value:
                        match = False
                        break
                if match:
                    results.append((edge.source, edge.target))
            
            return results

    def create_subgraph_fast(self, node_ids: List[str] = None, **attribute_filters) -> 'Graph':
        """
        Create a subgraph using optimized batch operations.
        
        Args:
            node_ids: Specific node IDs to include
            **attribute_filters: Filter nodes by attributes
            
        Returns:
            New Graph instance containing the filtered subgraph
        """
        if self.use_rust:
            if node_ids:
                # Use optimized Rust subgraph creation
                rust_subgraph = self._rust_core.get_subgraph_by_node_ids(node_ids)
                # Wrap in Graph instance
                new_graph = Graph(backend='rust')
                new_graph._rust_core = rust_subgraph
                return new_graph
            elif attribute_filters:
                # First filter nodes, then create subgraph
                filtered_node_ids = self.batch_filter_nodes(**attribute_filters)
                return self.create_subgraph_fast(node_ids=filtered_node_ids)
            else:
                return self.copy()
        else:
            # Use existing create_subgraph method for Python backend
            if attribute_filters:
                def node_filter(node):
                    return all(node.attributes.get(k) == v for k, v in attribute_filters.items())
                return self.create_subgraph(node_filter=node_filter)
            elif node_ids:
                return self.create_subgraph(node_ids=set(node_ids))
            else:
                return self.copy()

    def get_k_hop_neighborhood(self, start_node: str, k: int = 1) -> List[str]:
        """
        Get all nodes within k hops of the start node.
        
        Args:
            start_node: Starting node ID
            k: Number of hops (default 1)
            
        Returns:
            List of node IDs within k hops
        """
        if self.use_rust:
            return self._rust_core.get_k_hop_neighborhood(start_node, k)
        else:
            visited = set()
            current_layer = {start_node}
            visited.add(start_node)
            
            for _ in range(k):
                next_layer = set()
                for node_id in current_layer:
                    neighbors = self.get_neighbors(node_id)
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_layer.add(neighbor)
                
                current_layer = next_layer
                if not current_layer:
                    break
            
            return list(visited)

    # ========== BATCH ATTRIBUTE SETTING ==========
    
    def batch_set_node_attributes(self, node_attributes: Dict[str, Dict[str, Any]]) -> 'Graph':
        """
        Efficiently set attributes for multiple nodes in one operation.
        
        Args:
            node_attributes: Dict mapping node_id -> {attribute_key: value}
            
        Returns:
            Self for method chaining
            
        Example:
            g.batch_set_node_attributes({
                "alice": {"age": 31, "status": "active"},
                "bob": {"age": 26, "status": "inactive"}
            })
        """
        if self.use_rust:
            # Use Rust backend batch operation if available
            try:
                self._rust_core.batch_set_node_attributes(node_attributes)
            except AttributeError:
                # Fallback to individual operations
                for node_id, attributes in node_attributes.items():
                    for key, value in attributes.items():
                        self.set_node_attribute(node_id, key, value)
        else:
            # Python implementation with delta tracking
            self._init_delta()
            effective_nodes = self._get_effective_nodes()
            
            for node_id, new_attributes in node_attributes.items():
                if node_id not in effective_nodes:
                    continue  # Skip non-existent nodes
                
                current_node = effective_nodes[node_id]
                updated_attributes = dict(current_node.attributes)
                updated_attributes.update(new_attributes)
                
                self._pending_delta.modified_nodes[node_id] = Node(node_id, updated_attributes)
            
            self._invalidate_cache()
        
        return self
    
    def batch_set_edge_attributes(self, edge_attributes: Dict[str, Dict[str, Any]]) -> 'Graph':
        """
        Efficiently set attributes for multiple edges in one operation.
        
        Args:
            edge_attributes: Dict mapping edge_id -> {attribute_key: value}
            
        Returns:
            Self for method chaining
            
        Example:
            g.batch_set_edge_attributes({
                "alice->bob": {"weight": 0.8, "type": "friend"},
                "bob->charlie": {"weight": 0.6, "type": "colleague"}
            })
        """
        if self.use_rust:
            # Fallback to individual operations for now
            for edge_id, attributes in edge_attributes.items():
                for key, value in attributes.items():
                    self.set_edge_attribute(edge_id, key, value)
        else:
            # Python implementation with delta tracking
            self._init_delta()
            effective_edges = self._get_effective_edges()
            
            for edge_id, new_attributes in edge_attributes.items():
                if edge_id not in effective_edges:
                    continue  # Skip non-existent edges
                
                current_edge = effective_edges[edge_id]
                updated_attributes = dict(current_edge.attributes)
                updated_attributes.update(new_attributes)
                
                self._pending_delta.modified_edges[edge_id] = Edge(
                    current_edge.source, 
                    current_edge.target, 
                    updated_attributes
                )
            
            self._invalidate_cache()
        
        return self

    def batch_update_node_attributes(self, node_updates: Dict[str, Callable[[Dict], Dict]]) -> 'Graph':
        """
        Update node attributes using functions for complex transformations.
        
        Args:
            node_updates: Dict mapping node_id -> update_function
            
        Returns:
            Self for method chaining
            
        Example:
            g.batch_update_node_attributes({
                "alice": lambda attrs: {**attrs, "age": attrs["age"] + 1},
                "bob": lambda attrs: {**attrs, "score": attrs.get("score", 0) * 1.1}
            })
        """
        if self.use_rust:
            # For Rust backend, get current attributes and apply updates
            node_ids = list(node_updates.keys())
            current_attrs = self.batch_get_node_attributes(node_ids)
            
            batch_updates = {}
            for i, node_id in enumerate(node_ids):
                if i < len(current_attrs):
                    update_func = node_updates[node_id]
                    new_attrs = update_func(current_attrs[i])
                    batch_updates[node_id] = new_attrs
            
            return self.batch_set_node_attributes(batch_updates)
        else:
            # Python implementation
            effective_nodes = self._get_effective_nodes()
            batch_updates = {}
            
            for node_id, update_func in node_updates.items():
                if node_id in effective_nodes:
                    current_attrs = dict(effective_nodes[node_id].attributes)
                    new_attrs = update_func(current_attrs)
                    batch_updates[node_id] = new_attrs
            
            return self.batch_set_node_attributes(batch_updates)

    # ========== PERFORMANCE MONITORING ==========
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the graph"""
        stats = {
            'backend': self.backend,
            'node_count': self.node_count(),
            'edge_count': self.edge_count(),
            'rust_available': self.use_rust
        }
        
        if self.use_rust:
            try:
                rust_stats = self._rust_core.get_stats()
                stats.update(rust_stats)
            except AttributeError:
                # get_stats method not available in Rust backend
                stats['rust_stats'] = 'not_available'
        
        return stats

    def optimized_query(self, query_type: str, **params) -> Any:
        """
        Execute optimized queries using the best available backend method.
        
        Args:
            query_type: Type of query ('filter_nodes', 'filter_edges', 'subgraph', 'k_hop', 'batch_attrs')
            **params: Query-specific parameters
            
        Returns:
            Query results
        """
        if query_type == 'filter_nodes':
            return self.batch_filter_nodes(**params)
        elif query_type == 'filter_edges':
            return self.batch_filter_edges(**params)
        elif query_type == 'subgraph':
            return self.create_subgraph_fast(**params)
        elif query_type == 'k_hop':
            return self.get_k_hop_neighborhood(**params)
        elif query_type == 'batch_attrs':
            return self.batch_get_node_attributes(**params)
        else:
            raise ValueError(f"Unknown query type: {query_type}")
