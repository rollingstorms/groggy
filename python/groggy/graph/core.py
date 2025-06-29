"""
Core Graph class implementation
"""

from typing import Dict, List, Any, Optional, Callable, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .subgraph import Subgraph

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
    
    def __init__(self, nodes=None, edges=None, graph_attributes=None, backend=None, max_auto_states: int = 10, directed: bool = True):
        # Store graph properties
        self.directed = directed
        
        # Backend selection - use global setting or override
        if backend is None:
            # Import here to avoid circular import
            from .. import get_current_backend
            backend = get_current_backend()
        
        self.backend = backend
        self.use_rust = (backend == 'rust')
        
        if self.use_rust:
            # Rust handles processing, Python manages state/branches
            self._rust_core = _core.FastGraph(directed=directed)
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
        
        # Check if node already exists
        node_exists = node_id_str in self.nodes
        
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
        
        # Check if edge already exists
        edge_exists = edge_id in self.edges
        
        if self.use_rust:
            # Ensure both nodes exist before adding edge
            if source_str not in self.nodes:
                self.add_node(source_str)
            if target_str not in self.nodes:
                self.add_node(target_str)
            
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

    def has_edge(self, source: NodeID, target: NodeID) -> bool:
        """Check if an edge exists between source and target nodes"""
        source_str = str(source)
        target_str = str(target)
        if self.use_rust:
            return self._rust_core.has_edge(source_str, target_str)
        else:
            _, effective_edges, _ = self._get_effective_data()
            edge_id = f"{source_str}->{target_str}"
            return edge_id in effective_edges

    def get_node_ids(self) -> List[str]:
        """Get all node IDs"""
        if self.use_rust:
            return self._rust_core.get_node_ids()
        else:
            effective_nodes, _, _ = self._get_effective_data()
            return list(effective_nodes.keys())

    def get_neighbors(self, node_id: NodeID, direction: str = 'both') -> List[str]:
        """Get neighbors of a node
        
        Args:
            node_id: The node to get neighbors for
            direction: 'out' for outgoing, 'in' for incoming, 'both' for all neighbors
        """
        node_id_str = str(node_id)
        if self.use_rust:
            # Use the appropriate Rust method based on direction
            if direction == 'out':
                return self._rust_core.get_outgoing_neighbors(node_id_str)
            elif direction == 'in':
                return self._rust_core.get_incoming_neighbors(node_id_str)
            else:  # 'both'
                if self.directed:
                    return self._rust_core.get_all_neighbors(node_id_str)
                else:
                    # For undirected graphs, neighbors() already gives all connected nodes
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

    def get_outgoing_neighbors(self, node_id: NodeID) -> List[str]:
        """Get outgoing neighbors of a node (for directed graphs)"""
        node_id_str = str(node_id)
        if self.use_rust:
            return self._rust_core.get_outgoing_neighbors(node_id_str)
        else:
            neighbors = set()
            _, effective_edges, _ = self._get_effective_data()
            
            for edge in effective_edges.values():
                if edge.source == node_id_str:
                    neighbors.add(edge.target)
            
            return list(neighbors)
    
    def get_incoming_neighbors(self, node_id: NodeID) -> List[str]:
        """Get incoming neighbors of a node (for directed graphs)"""
        node_id_str = str(node_id)
        if self.use_rust:
            return self._rust_core.get_incoming_neighbors(node_id_str)
        else:
            neighbors = set()
            _, effective_edges, _ = self._get_effective_data()
            
            for edge in effective_edges.values():
                if edge.target == node_id_str:
                    neighbors.add(edge.source)
            
            return list(neighbors)
    
    def get_all_neighbors(self, node_id: NodeID) -> List[str]:
        """Get all neighbors of a node (both incoming and outgoing)"""
        node_id_str = str(node_id)
        if self.use_rust:
            return self._rust_core.get_all_neighbors(node_id_str)
        else:
            neighbors = set()
            _, effective_edges, _ = self._get_effective_data()
            
            for edge in effective_edges.values():
                if edge.source == node_id_str:
                    neighbors.add(edge.target)
                if edge.target == node_id_str:
                    neighbors.add(edge.source)
            
            return list(neighbors)

    # High-level wrapper methods to avoid direct _rust_core access
    def filter_nodes(self, filter_func: Union[Callable[[NodeID, Dict[str, Any]], bool], Dict[str, Any], str] = None, return_graph: bool = False, **kwargs) -> Union[List[str], 'Subgraph']:
        """Filter nodes by lambda function, attribute values, string query, or keyword arguments
        
        Args:
            filter_func: Either a callable that takes (node_id, attributes) and returns bool,
                        a dictionary of attribute filters, or a string query like "role == 'Manager'"
            return_graph: If True, return a new Subgraph with filtered nodes and their edges.
                         If False, return a list of node IDs (default)
            **kwargs: Keyword arguments for attribute filtering (e.g., role='engineer', age=30)
            
        Returns:
            List of node IDs that match the filter, or a new Subgraph instance if return_subgraph=True
            
        Examples:
            # Function predicate
            filtered_ids = graph.filter_nodes(lambda node_id, attrs: attrs.get('age', 0) > 30)
            
            # Dictionary predicate
            filtered_ids = graph.filter_nodes({'role': 'Manager', 'department': 'Engineering'})
            
            # String query
            filtered_ids = graph.filter_nodes("role == 'Manager' and salary > 100000")
            
            # Keyword arguments (most convenient)
            filtered_ids = graph.filter_nodes(role='engineer')
            filtered_ids = graph.filter_nodes(role='Manager', department='Engineering')
            
            # Return subgraph
            subgraph = graph.filter_nodes(role='Manager', return_graph=True)
        """
        # Handle keyword arguments - convert to dictionary filter
        if kwargs:
            if filter_func is not None:
                raise ValueError("Cannot specify both filter_func and keyword arguments")
            filter_func = kwargs
        
        # Store original filter criteria for metadata
        original_filter = filter_func
        
        # Handle string query
        if isinstance(filter_func, str):
            from .filtering import QueryCompiler
            filter_func = QueryCompiler.compile_node_query(filter_func)
        elif isinstance(filter_func, dict):
            # Dictionary-based filtering - convert to function
            filter_dict = filter_func  # Store reference to dict before reassigning
            def dict_filter(node_id: str, attributes: Dict[str, Any]) -> bool:
                for attr, value in filter_dict.items():
                    if attr not in attributes or attributes[attr] != value:
                        return False
                return True
            filter_func = dict_filter
        
        # Apply the filter
        if filter_func is None:
            filtered_node_ids = list(self.get_node_ids())
        else:
            # Apply filter function
            if self.use_rust:
                # For Rust backend, we need to iterate through all nodes
                filtered_node_ids = []
                for node_id in self.nodes:
                    node = self.get_node(node_id)
                    if node and filter_func(node_id, node.attributes):
                        filtered_node_ids.append(node_id)
            else:
                # Python fallback implementation
                effective_nodes, _, _ = self._get_effective_data()
                filtered_node_ids = []
                for node_id, node in effective_nodes.items():
                    if filter_func(node_id, node.attributes):
                        filtered_node_ids.append(node_id)
        
        if return_graph:
            return self._create_subgraph_from_nodes(filtered_node_ids, original_filter)
        else:
            return filtered_node_ids

    def filter_edges(self, filter_func: Union[Callable[[str, NodeID, NodeID, Dict[str, Any]], bool], Dict[str, Any], str] = None, return_graph: bool = False, **kwargs) -> Union[List[str], 'Subgraph']:
        """Filter edges by lambda function, attribute values, string query, or keyword arguments
        
        Args:
            filter_func: Either a callable that takes (edge_id, source, target, attributes) and returns bool,
                        a dictionary of attribute filters, or a string query like "weight > 0.5"
            return_graph: If True, return a new Subgraph with filtered edges and connected nodes.
                         If False, return a list of edge IDs (default)
            **kwargs: Keyword arguments for attribute filtering (e.g., relationship='friend', weight=0.8)
            
        Returns:
            List of edge IDs that match the filter, or a new Subgraph instance if return_subgraph=True
            
        Examples:
            # Function predicate
            filtered_ids = graph.filter_edges(lambda edge_id, src, tgt, attrs: attrs.get('weight', 0) > 0.5)
            
            # Dictionary predicate
            filtered_ids = graph.filter_edges({'relationship': 'friend', 'active': True})
            
            # String query
            filtered_ids = graph.filter_edges("weight > 0.5 and relationship == 'colleague'")
            
            # Keyword arguments (most convenient)
            filtered_ids = graph.filter_edges(relationship='friend')
            filtered_ids = graph.filter_edges(relationship='colleague', weight=0.8)
            
            # Return subgraph
            subgraph = graph.filter_edges(relationship='friend', return_graph=True)
        """
        # Handle keyword arguments - convert to dictionary filter
        if kwargs:
            if filter_func is not None:
                raise ValueError("Cannot specify both filter_func and keyword arguments")
            filter_func = kwargs
        
        # Store original filter criteria for metadata
        original_filter = filter_func
        
        # Handle string query
        if isinstance(filter_func, str):
            from .filtering import QueryCompiler
            filter_func = QueryCompiler.compile_edge_query(filter_func)
        elif isinstance(filter_func, dict):
            # Dictionary-based filtering - convert to function
            filter_dict = filter_func  # Store reference to dict before reassigning
            def dict_filter(edge_id: str, source: str, target: str, attributes: Dict[str, Any]) -> bool:
                for attr, value in filter_dict.items():
                    if attr not in attributes or attributes[attr] != value:
                        return False
                return True
            filter_func = dict_filter
        
        # Apply the filter
        if filter_func is None:
            filtered_edge_ids = list(self.edges.keys())
        else:
            # Apply filter function
            if self.use_rust:
                # For Rust backend, we need to iterate through all edges
                filtered_edge_ids = []
                for edge_id in self.edges:
                    edge = self.edges[edge_id]  # Use the EdgeView instead of get_edge
                    if edge and filter_func(edge_id, edge.source, edge.target, edge.attributes):
                        filtered_edge_ids.append(edge_id)
            else:
                # Python fallback implementation
                _, effective_edges, _ = self._get_effective_data()
                filtered_edge_ids = []
                for edge_id, edge in effective_edges.items():
                    if filter_func(edge_id, edge.source, edge.target, edge.attributes):
                        filtered_edge_ids.append(edge_id)
        
        if return_graph:
            return self._create_subgraph_from_edges(filtered_edge_ids, original_filter)
        else:
            return filtered_edge_ids

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

    # Helper methods for enhanced filtering
    def _compile_query_predicate(self, query_str: str, is_node: bool = True):
        """Compile a string query into a filter predicate function"""
        from .filtering import QueryCompiler
        
        if is_node:
            return QueryCompiler.compile_node_query(query_str)
        else:
            return QueryCompiler.compile_edge_query(query_str)
    
    def _create_subgraph_from_nodes(self, node_ids: List[str], filter_criteria=None) -> 'Subgraph':
        """Create a subgraph containing only the specified nodes and edges between them"""
        from .filtering import SubgraphCreator
        return SubgraphCreator.create_node_subgraph(self, node_ids, filter_criteria)
    
    def _create_subgraph_from_edges(self, edge_ids: List[str], filter_criteria=None) -> 'Subgraph':
        """Create a subgraph containing only the specified edges and their connected nodes"""
        from .filtering import SubgraphCreator
        return SubgraphCreator.create_edge_subgraph(self, edge_ids, filter_criteria)

    def update_node(self, node_id: NodeID, attributes: Dict[str, Any] = None, **kwargs):
        """Update node attributes with a user-friendly interface
        
        Args:
            node_id: ID of the node to update
            attributes: Dictionary of attributes to set (optional)
            **kwargs: Attributes as keyword arguments
            
        Examples:
            g.update_node("alice", {"age": 31, "role": "senior_engineer"})
            g.update_node("alice", age=31, role="senior_engineer")
            g.update_node("alice", {"age": 31}, role="senior_engineer")  # Both work together
        """
        # Combine dictionary and keyword attributes
        combined_attrs = attributes or {}
        combined_attrs.update(kwargs)
        
        node_id_str = str(node_id)
        if self.use_rust:
            # Rust backend - direct update using set_node_attribute for each attribute
            for attr_name, attr_value in combined_attrs.items():
                self._rust_core.set_node_attribute(node_id_str, attr_name, attr_value)
            self._invalidate_cache()
        else:
            # Python fallback - use delta
            self._init_delta()
            
            if node_id_str in self._pending_delta.added_nodes:
                # Update existing pending node
                current_node = self._pending_delta.added_nodes[node_id_str]
                current_node.attributes.update(combined_attrs)
            else:
                # Create new node with combined attributes
                new_node = Node(node_id_str, combined_attrs)
                self._pending_delta.added_nodes[node_id_str] = new_node
                self._update_cache_for_node_add(node_id_str, new_node)

    def update_edge(self, source: NodeID, target: NodeID, attributes: Dict[str, Any] = None, **kwargs):
        """Update edge attributes with a user-friendly interface
        
        Args:
            source: Source node ID of the edge to update
            target: Target node ID of the edge to update
            attributes: Dictionary of attributes to set (optional)
            **kwargs: Attributes as keyword arguments
            
        Examples:
            g.update_edge("alice", "bob", {"relationship": "colleague", "since": 2020})
            g.update_edge("alice", "bob", relationship="colleague", since=2020)
            g.update_edge("alice", "bob", {"relationship": "colleague"}, since=2020)  # Both work together
        """
        # Combine dictionary and keyword attributes
        combined_attrs = attributes or {}
        combined_attrs.update(kwargs)
        
        source_str = str(source)
        target_str = str(target)
        edge_id = f"{source_str}->{target_str}"
        
        if self.use_rust:
            # Rust backend - direct update
            self._rust_core.set_edge_attributes(source_str, target_str, combined_attrs)
            self._invalidate_cache()
        else:
            # Python fallback - use delta
            self._init_delta()
            
            if edge_id in self._pending_delta.added_edges:
                # Update existing pending edge
                current_edge = self._pending_delta.added_edges[edge_id]
                current_edge.attributes.update(combined_attrs)
            else:
                # Create new edge with combined attributes
                new_edge = Edge(source, target, combined_attrs)
                self._pending_delta.added_edges[edge_id] = new_edge
                self._update_cache_for_edge_add(edge_id, new_edge)

    def add_nodes(self, nodes_data: List[Dict[str, Any]]):
        """Add multiple nodes efficiently in a single operation
        
        Args:
            nodes_data: List of node dictionaries, each containing 'id' and optional attributes
            
        Example:
            nodes = [
                {'id': 'user_1', 'age': 25, 'role': 'engineer'},
                {'id': 'user_2', 'age': 30, 'role': 'manager'}
            ]
            g.add_nodes(nodes)
        """
        if self.use_rust:
            # Prepare data for Rust backend
            rust_nodes = []
            for node_data in nodes_data:
                node_id = str(node_data['id'])
                attributes = {k: v for k, v in node_data.items() if k != 'id'}
                rust_nodes.append((node_id, attributes))
            
            self._rust_core.add_nodes(rust_nodes)
            self._invalidate_cache()
        else:
            # Python fallback - batch add with single delta
            self._init_delta()
            
            for node_data in nodes_data:
                node_id = str(node_data['id'])
                attributes = {k: v for k, v in node_data.items() if k != 'id'}
                
                if node_id not in self._pending_delta.added_nodes:
                    new_node = Node(node_id, attributes)
                    self._pending_delta.added_nodes[node_id] = new_node
                    self._update_cache_for_node_add(node_id, new_node)

    def add_edges(self, edges_data: List[Dict[str, Any]]):
        """Add multiple edges efficiently in a single operation
        
        Args:
            edges_data: List of edge dictionaries, each containing 'source', 'target' and optional attributes
            
        Example:
            edges = [
                {'source': 'user_1', 'target': 'user_2', 'relationship': 'manages'},
                {'source': 'user_2', 'target': 'user_3', 'relationship': 'collaborates'}
            ]
            g.add_edges(edges)
        """
        if self.use_rust:
            # Prepare data for Rust backend
            rust_edges = []
            for edge_data in edges_data:
                source = str(edge_data['source'])
                target = str(edge_data['target'])
                attributes = {k: v for k, v in edge_data.items() if k not in ['source', 'target']}
                rust_edges.append((source, target, attributes))
            
            self._rust_core.add_edges(rust_edges)
            self._invalidate_cache()
        else:
            # Python fallback - batch add with single delta
            self._init_delta()
            effective_nodes, _, _ = self._get_effective_data()
            
            for edge_data in edges_data:
                source = str(edge_data['source'])
                target = str(edge_data['target'])
                attributes = {k: v for k, v in edge_data.items() if k not in ['source', 'target']}
                edge_id = f"{source}->{target}"
                
                # Auto-create nodes if they don't exist
                if source not in effective_nodes and source not in self._pending_delta.added_nodes:
                    self._pending_delta.added_nodes[source] = Node(source)
                if target not in effective_nodes and target not in self._pending_delta.added_nodes:
                    self._pending_delta.added_nodes[target] = Node(target)
                
                if edge_id not in self._pending_delta.added_edges:
                    new_edge = Edge(source, target, attributes)
                    self._pending_delta.added_edges[edge_id] = new_edge
                    self._update_cache_for_edge_add(edge_id, new_edge)

    def update_nodes(self, updates: Dict[NodeID, Dict[str, Any]]):
        """Update multiple nodes efficiently in a single operation
        
        Args:
            updates: Dictionary mapping node IDs to their attribute updates
            
        Example:
            updates = {
                'user_1': {'salary': 80000, 'role': 'senior_engineer'},
                'user_2': {'salary': 90000, 'department': 'engineering'}
            }
            g.update_nodes(updates)
        """
        if self.use_rust:
            # Use Rust backend for efficient bulk updates - expects dict format
            rust_updates = {}
            for node_id, attrs in updates.items():
                node_id_str = str(node_id)
                rust_updates[node_id_str] = attrs
            
            self._rust_core.set_nodes_attributes_batch(rust_updates)
            self._invalidate_cache()
        else:
            # Python fallback - batch update with single delta
            self._init_delta()
            effective_nodes, _, _ = self._get_effective_data()
            
            for node_id, attrs in updates.items():
                node_id_str = str(node_id)
                
                # Get current node or create if it doesn't exist
                if node_id_str in effective_nodes:
                    current_node = effective_nodes[node_id_str]
                    updated_attrs = {**current_node.attributes, **attrs}
                elif node_id_str in self._pending_delta.added_nodes:
                    current_node = self._pending_delta.added_nodes[node_id_str]
                    updated_attrs = {**current_node.attributes, **attrs}
                else:
                    # Create new node with the attributes
                    updated_attrs = attrs
                
                updated_node = Node(node_id_str, updated_attrs)
                self._pending_delta.added_nodes[node_id_str] = updated_node
                self._update_cache_for_node_add(node_id_str, updated_node)

    def remove_node(self, node_id: NodeID) -> bool:
        """Remove a node from the graph
        
        Args:
            node_id: The ID of the node to remove
            
        Returns:
            bool: True if the node was removed, False if it didn't exist
        """
        node_id_str = str(node_id)
        
        if self.use_rust:
            result = self._rust_core.remove_node(node_id_str)
            self._invalidate_cache()
            return result
        else:
            # Python fallback implementation
            self._init_delta()
            effective_nodes, _, _ = self._get_effective_data()
            
            if node_id_str in effective_nodes or node_id_str in self._pending_delta.added_nodes:
                # Mark for removal and invalidate cache
                self._pending_delta.removed_nodes.add(node_id_str)
                
                # Also remove any edges connected to this node
                effective_edges, _, _ = self._get_effective_data()
                edges_to_remove = []
                for edge_id, edge in effective_edges.items():
                    if edge.source == node_id_str or edge.target == node_id_str:
                        edges_to_remove.append(edge_id)
                
                for edge_id in edges_to_remove:
                    self._pending_delta.removed_edges.add(edge_id)
                
                self._invalidate_cache()
                return True
            
            return False

    def remove_edge(self, source: NodeID, target: NodeID) -> bool:
        """Remove an edge from the graph
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            bool: True if the edge was removed, False if it didn't exist
        """
        source_str = str(source)
        target_str = str(target)
        edge_id = f"{source_str}->{target_str}"
        
        if self.use_rust:
            result = self._rust_core.remove_edge(source_str, target_str)
            self._invalidate_cache()
            return result
        else:
            # Python fallback implementation
            self._init_delta()
            _, effective_edges, _ = self._get_effective_data()
            
            if edge_id in effective_edges or edge_id in self._pending_delta.added_edges:
                self._pending_delta.removed_edges.add(edge_id)
                self._invalidate_cache()
                return True
            
            return False

    def remove_nodes(self, node_ids: List[NodeID]) -> int:
        """Remove multiple nodes efficiently
        
        Args:
            node_ids: List of node IDs to remove
            
        Returns:
            int: Number of nodes actually removed
        """
        node_ids_str = [str(node_id) for node_id in node_ids]
        
        if self.use_rust:
            removed_count = self._rust_core.remove_nodes(node_ids_str)
            self._invalidate_cache()
            return removed_count
        else:
            # Python fallback implementation
            self._init_delta()
            effective_nodes, effective_edges, _ = self._get_effective_data()
            removed_count = 0
            
            for node_id_str in node_ids_str:
                if node_id_str in effective_nodes or node_id_str in self._pending_delta.added_nodes:
                    self._pending_delta.removed_nodes.add(node_id_str)
                    removed_count += 1
                    
                    # Also remove connected edges
                    edges_to_remove = []
                    for edge_id, edge in effective_edges.items():
                        if edge.source == node_id_str or edge.target == node_id_str:
                            edges_to_remove.append(edge_id)
                    
                    for edge_id in edges_to_remove:
                        self._pending_delta.removed_edges.add(edge_id)
            
            if removed_count > 0:
                self._invalidate_cache()
            
            return removed_count

    def remove_edges(self, edge_pairs: List[tuple]) -> int:
        """Remove multiple edges efficiently
        
        Args:
            edge_pairs: List of (source, target) tuples
            
        Returns:
            int: Number of edges actually removed
        """
        edge_pairs_str = [(str(source), str(target)) for source, target in edge_pairs]
        
        if self.use_rust:
            removed_count = self._rust_core.remove_edges(edge_pairs_str)
            self._invalidate_cache()
            return removed_count
        else:
            # Python fallback implementation
            self._init_delta()
            _, effective_edges, _ = self._get_effective_data()
            removed_count = 0
            
            for source_str, target_str in edge_pairs_str:
                edge_id = f"{source_str}->{target_str}"
                if edge_id in effective_edges or edge_id in self._pending_delta.added_edges:
                    self._pending_delta.removed_edges.add(edge_id)
                    removed_count += 1
            
            if removed_count > 0:
                self._invalidate_cache()
            
            return removed_count

    def subgraph(self, node_ids: List[NodeID] = None, edge_ids: List[str] = None, include_edges: bool = True) -> 'Subgraph':
        """Create a single subgraph from specified nodes or edges
        
        Args:
            node_ids: List of node IDs to include in the subgraph
            edge_ids: List of edge IDs to include in the subgraph  
            include_edges: If True and node_ids provided, include edges between specified nodes
            
        Returns:
            Subgraph containing the specified nodes/edges
            
        Examples:
            # Create subgraph from specific nodes
            sub = g.subgraph(node_ids=['alice', 'bob', 'charlie'])
            
            # Create subgraph from specific edges (includes connected nodes)
            sub = g.subgraph(edge_ids=['alice->bob', 'bob->charlie'])
            
            # Create subgraph from nodes without their edges
            sub = g.subgraph(node_ids=['alice', 'bob'], include_edges=False)
        """
        if node_ids is not None and edge_ids is not None:
            raise ValueError("Cannot specify both node_ids and edge_ids")
        
        if node_ids is None and edge_ids is None:
            raise ValueError("Must specify either node_ids or edge_ids")
        
        if node_ids is not None:
            # Convert to strings for consistency
            node_ids_str = [str(node_id) for node_id in node_ids]
            return self._create_subgraph_from_nodes(node_ids_str, f"nodes: {node_ids_str}")
        else:
            # Convert to strings for consistency
            edge_ids_str = [str(edge_id) for edge_id in edge_ids]
            return self._create_subgraph_from_edges(edge_ids_str, f"edges: {edge_ids_str}")

    def subgraphs(self, group_by: Union[str, Dict[str, List[Any]]] = None, **filters) -> Dict[str, 'Subgraph']:
        """Create multiple subgraphs grouped by attribute values
        
        Args:
            group_by: Either:
                - String: attribute name to group nodes by (like SQL GROUP BY)
                - Dict: mapping of group names to lists of attribute values
            **filters: Additional filters as keyword arguments
            
        Returns:
            Dictionary mapping attribute values to Subgraph objects
            
        Examples:
            # Group by single attribute (discovers unique values)
            subs = g.subgraphs('department')  # Returns {'engineering': subgraph, 'design': subgraph, ...}
            
            # Group by specific attribute values  
            subs = g.subgraphs(role=['engineer', 'manager'])  # Returns {'engineer': subgraph, 'manager': subgraph}
            
            # Group by custom mapping
            groups = {
                'technical': ['engineer', 'architect'],
                'leadership': ['manager', 'director']
            }
            subs = g.subgraphs(group_by={'role': groups})  # Returns {'technical': subgraph, 'leadership': subgraph}
            
            # Multiple filters
            subs = g.subgraphs('department', active=True, salary_min=50000)
            
            # Access subgraphs by attribute value:
            designer_sub = subs['designer']  # Gets subgraph with filter_criteria="role=designer"
        """
        result = {}
        
        if isinstance(group_by, str):
            # Group by attribute name - discover unique values
            attr_name = group_by
            unique_values = set()
            
            # Collect unique values for the attribute
            for node_id in self.nodes:
                node = self.get_node(node_id)
                if node and attr_name in node.attributes:
                    unique_values.add(node.attributes[attr_name])
            
            # Create subgraph for each unique value
            for value in unique_values:
                # Build filter criteria
                filter_criteria = {attr_name: value}
                filter_criteria.update(filters)
                
                # Filter nodes matching this value
                matching_nodes = self.filter_nodes(filter_criteria)
                
                if matching_nodes:
                    group_name = f"{attr_name}={value}"
                    subgraph = self._create_subgraph_from_nodes(matching_nodes, group_name)
                    result[str(value)] = subgraph  # Use attribute value as key
        
        elif isinstance(group_by, dict):
            # Group by custom mapping
            for attr_name, groups in group_by.items():
                if isinstance(groups, dict):
                    # groups is a mapping of group_name -> list of values
                    for group_name, values in groups.items():
                        # Find nodes with any of these values
                        matching_nodes = []
                        for node_id in self.nodes:
                            node = self.get_node(node_id)
                            if node and attr_name in node.attributes:
                                if node.attributes[attr_name] in values:
                                    # Check additional filters
                                    matches_filters = True
                                    for filter_attr, filter_value in filters.items():
                                        if filter_attr not in node.attributes or node.attributes[filter_attr] != filter_value:
                                            matches_filters = False
                                            break
                                    
                                    if matches_filters:
                                        matching_nodes.append(node_id)
                        
                        if matching_nodes:
                            subgraph = self._create_subgraph_from_nodes(matching_nodes, group_name)
                            result[group_name] = subgraph  # Use custom group name as key
                else:
                    # groups is a list of values
                    for value in groups:
                        filter_criteria = {attr_name: value}
                        filter_criteria.update(filters)
                        
                        matching_nodes = self.filter_nodes(filter_criteria)
                        if matching_nodes:
                            group_name = f"{attr_name}={value}"
                            subgraph = self._create_subgraph_from_nodes(matching_nodes, group_name)
                            result[str(value)] = subgraph  # Use attribute value as key
        
        elif filters:
            # No group_by specified, but filters provided - create single subgraph
            if len(filters) == 1:
                # Single filter - group by its values
                attr_name, values = next(iter(filters.items()))
                if isinstance(values, list):
                    for value in values:
                        matching_nodes = self.filter_nodes({attr_name: value})
                        if matching_nodes:
                            group_name = f"{attr_name}={value}"
                            subgraph = self._create_subgraph_from_nodes(matching_nodes, group_name)
                            result[str(value)] = subgraph  # Use attribute value as key
                else:
                    # Single value
                    matching_nodes = self.filter_nodes(filters)
                    if matching_nodes:
                        group_name = f"{attr_name}={values}"
                        subgraph = self._create_subgraph_from_nodes(matching_nodes, group_name)
                        result[str(values)] = subgraph  # Use attribute value as key
            else:
                # Multiple filters - create one subgraph
                matching_nodes = self.filter_nodes(filters)
                if matching_nodes:
                    filter_desc = "_".join(f"{k}={v}" for k, v in filters.items())
                    subgraph = self._create_subgraph_from_nodes(matching_nodes, filter_desc)
                    result[filter_desc] = subgraph  # Use combined filter as key
        
        else:
            raise ValueError("Must specify either group_by or filter criteria")
        
        return result


