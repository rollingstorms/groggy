"""
GLI - Graph Language Interface
Core library for graph manipulation with git-like state management and dynamic module loading.
"""

import hashlib
import json
import time
import importlib.util
import yaml
import weakref
from collections import OrderedDict, deque
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path

# Try to use faster xxhash if available, fallback to hashlib
try:
    import xxhash
    def fast_hash(data: str) -> str:
        return xxhash.xxh64(data.encode()).hexdigest()[:16]
except ImportError:
    def fast_hash(data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class ContentPool:
    """Content-addressed storage for nodes and edges to avoid duplication"""
    
    def __init__(self):
        self.nodes = {}  # content_hash -> Node
        self.edges = {}  # content_hash -> Edge
        self.node_refs = {}  # track references
        self.edge_refs = {}  # track references
    
    def _node_content_hash(self, node: 'Node') -> str:
        content = f"{node.id}:{json.dumps(node.attributes, sort_keys=True)}"
        return fast_hash(content)
    
    def _edge_content_hash(self, edge: 'Edge') -> str:
        content = f"{edge.source}->{edge.target}:{json.dumps(edge.attributes, sort_keys=True)}"
        return fast_hash(content)
    
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
class GraphState:
    """Optimized graph state using deltas and content addressing"""
    hash: str
    parent_hash: Optional[str] = None
    operation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    # For root states: full content
    nodes: Optional[Dict[str, str]] = None  # node_id -> content_hash
    edges: Optional[Dict[str, str]] = None  # edge_id -> content_hash
    graph_attributes: Optional[Dict[str, Any]] = None
    
    # For incremental states: delta only
    delta: Optional[CompactGraphDelta] = None
    
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
    """Main graph interface - immutable operations return new Graph instances with copy-on-write"""
    
    def __init__(self, nodes=None, edges=None, graph_attributes=None, graph_store=None):
        self.nodes = OrderedDict(nodes or {})
        self.edges = OrderedDict(edges or {})
        self.graph_attributes = graph_attributes or {}
        self.graph_store = graph_store
        self._current_time = 0
        
        # Track insertion order for dynamic behavior
        self.node_order = {}
        self.edge_order = {}
        
        # Change tracking for copy-on-write
        self._pending_delta: Optional[GraphDelta] = None
        self._is_modified = False
    
    @classmethod
    def empty(cls, graph_store=None):
        """Create empty graph"""
        return cls(graph_store=graph_store)
    
    def _next_time(self):
        self._current_time += 1
        return self._current_time
    
    def _init_delta(self):
        """Initialize delta tracking for copy-on-write"""
        if self._is_modified:
            return
            
        self._is_modified = True
        self._pending_delta = GraphDelta()
        
        # Copy-on-write: shallow copy the collections
        self.nodes = self.nodes.copy()
        self.edges = self.edges.copy()
        self.graph_attributes = self.graph_attributes.copy()
        self.node_order = self.node_order.copy()
        self.edge_order = self.edge_order.copy()
    
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
    
    def _get_effective_data(self):
        """Get effective nodes/edges including pending changes"""
        if not self._pending_delta:
            return self.nodes, self.edges, self.graph_attributes
        
        # Build effective state without modifying original
        effective_nodes = self.nodes.copy()
        effective_edges = self.edges.copy()
        effective_attrs = self.graph_attributes.copy()
        
        delta = self._pending_delta
        
        # Apply changes
        effective_nodes.update(delta.added_nodes)
        effective_nodes.update(delta.modified_nodes)
        for node_id in delta.removed_nodes:
            effective_nodes.pop(node_id, None)
        
        effective_edges.update(delta.added_edges)
        effective_edges.update(delta.modified_edges)
        for edge_id in delta.removed_edges:
            effective_edges.pop(edge_id, None)
        
        # Remove edges connected to removed nodes
        for node_id in delta.removed_nodes:
            edges_to_remove = [eid for eid, edge in effective_edges.items() 
                             if edge.source == node_id or edge.target == node_id]
            for eid in edges_to_remove:
                effective_edges.pop(eid, None)
        
        effective_attrs.update(delta.modified_graph_attrs)
        
        return effective_nodes, effective_edges, effective_attrs
    
    def add_node(self, node_id: str, **attributes) -> 'Graph':
        """Add node using copy-on-write"""
        if not self._is_modified:
            # Check if node already exists
            effective_nodes, _, _ = self._get_effective_data()
            if node_id in effective_nodes:
                return self
        
        self._init_delta()
        
        new_node = Node(node_id, attributes)
        self._pending_delta.added_nodes[node_id] = new_node
        
        return self
    
    def add_edge(self, source: str, target: str, **attributes) -> 'Graph':
        """Add edge using copy-on-write"""
        edge_id = f"{source}->{target}"
        
        # Ensure nodes exist
        effective_nodes, effective_edges, _ = self._get_effective_data()
        
        if edge_id in effective_edges:
            return self
        
        self._init_delta()
        
        # Add missing nodes
        if source not in effective_nodes and source not in self._pending_delta.added_nodes:
            self._pending_delta.added_nodes[source] = Node(source)
        if target not in effective_nodes and target not in self._pending_delta.added_nodes:
            self._pending_delta.added_nodes[target] = Node(target)
        
        new_edge = Edge(source, target, attributes)
        self._pending_delta.added_edges[edge_id] = new_edge
        
        return self
    
    def remove_node(self, node_id: str) -> 'Graph':
        """Remove node using copy-on-write"""
        effective_nodes, _, _ = self._get_effective_data()
        if node_id not in effective_nodes:
            return self
        
        self._init_delta()
        self._pending_delta.removed_nodes.add(node_id)
        
        return self
    
    def remove_edge(self, source: str, target: str) -> 'Graph':
        """Remove edge using copy-on-write"""
        edge_id = f"{source}->{target}"
        effective_nodes, effective_edges, _ = self._get_effective_data()
        
        if edge_id not in effective_edges:
            return self
        
        self._init_delta()
        self._pending_delta.removed_edges.add(edge_id)
        
        return self
    
    def set_node_attribute(self, node_id: str, key: str, value: Any) -> 'Graph':
        """Set node attribute using copy-on-write"""
        effective_nodes, _, _ = self._get_effective_data()
        if node_id not in effective_nodes:
            raise ValueError(f"Node {node_id} not found")
        
        self._init_delta()
        
        # Get current node (from effective state)
        current_node = effective_nodes[node_id]
        updated_node = current_node.set_attribute(key, value)
        self._pending_delta.modified_nodes[node_id] = updated_node
        
        return self
    
    def set_edge_attribute(self, source: str, target: str, key: str, value: Any) -> 'Graph':
        """Set edge attribute using copy-on-write"""
        edge_id = f"{source}->{target}"
        effective_nodes, effective_edges, _ = self._get_effective_data()
        
        if edge_id not in effective_edges:
            raise ValueError(f"Edge {edge_id} not found")
        
        self._init_delta()
        
        # Get current edge (from effective state)
        current_edge = effective_edges[edge_id]
        updated_edge = current_edge.set_attribute(key, value)
        self._pending_delta.modified_edges[edge_id] = updated_edge
        
        return self
    
    def snapshot(self) -> 'Graph':
        """Create immutable snapshot by applying all pending changes"""
        if not self._pending_delta:
            # No changes, can share data
            new_graph = Graph(self.nodes, self.edges, self.graph_attributes, self.graph_store)
            new_graph.node_order = self.node_order
            new_graph.edge_order = self.edge_order
            new_graph._current_time = self._current_time
            return new_graph
        
        # Apply changes and create new graph
        self._apply_pending_changes()
        
        new_graph = Graph(
            OrderedDict(self.nodes), 
            OrderedDict(self.edges), 
            self.graph_attributes.copy(), 
            self.graph_store
        )
        new_graph.node_order = self.node_order.copy()
        new_graph.edge_order = self.edge_order.copy()
        new_graph._current_time = self._current_time
        
        return new_graph
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbor node IDs"""
        effective_nodes, effective_edges, _ = self._get_effective_data()
        neighbors = []
        for edge in effective_edges.values():
            if edge.source == node_id:
                neighbors.append(edge.target)
            elif edge.target == node_id:
                neighbors.append(edge.source)
        return neighbors
    
    def get_nodes_by_time(self, start_time: float = 0, end_time: Optional[float] = None) -> List[Node]:
        """Get nodes added within time range"""
        if end_time is None:
            end_time = self._current_time
        
        effective_nodes, _, _ = self._get_effective_data()
        return [effective_nodes[node_id] for node_id, add_time in self.node_order.items()
                if start_time <= add_time <= end_time and node_id in effective_nodes]
    
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
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for module_config in config.get('modules', []):
            self.load_module(module_config)
    
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


class Graph:
    """Main graph interface - immutable operations return new Graph instances with copy-on-write"""
    
    def __init__(self, nodes=None, edges=None, graph_attributes=None, graph_store=None):
        self.nodes = OrderedDict(nodes or {})
        self.edges = OrderedDict(edges or {})
        self.graph_attributes = graph_attributes or {}
        self.graph_store = graph_store
        self._current_time = 0
        
        # Track insertion order for dynamic behavior
        self.node_order = {}
        self.edge_order = {}
        
        # Change tracking for copy-on-write
        self._pending_delta: Optional[GraphDelta] = None
        self._is_modified = False
    
    @classmethod
    def empty(cls, graph_store=None):
        """Create empty graph"""
        return cls(graph_store=graph_store)
    
    def _next_time(self):
        self._current_time += 1
        return self._current_time
    
    def _init_delta(self):
        """Initialize delta tracking for copy-on-write"""
        if self._is_modified:
            return
            
        self._is_modified = True
        self._pending_delta = GraphDelta()
        
        # Copy-on-write: shallow copy the collections
        self.nodes = self.nodes.copy()
        self.edges = self.edges.copy()
        self.graph_attributes = self.graph_attributes.copy()
        self.node_order = self.node_order.copy()
        self.edge_order = self.edge_order.copy()
    
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
    
    def _get_effective_data(self):
        """Get effective nodes/edges including pending changes"""
        if not self._pending_delta:
            return self.nodes, self.edges, self.graph_attributes
        
        # Build effective state without modifying original
        effective_nodes = self.nodes.copy()
        effective_edges = self.edges.copy()
        effective_attrs = self.graph_attributes.copy()
        
        delta = self._pending_delta
        
        # Apply changes
        effective_nodes.update(delta.added_nodes)
        effective_nodes.update(delta.modified_nodes)
        for node_id in delta.removed_nodes:
            effective_nodes.pop(node_id, None)
        
        effective_edges.update(delta.added_edges)
        effective_edges.update(delta.modified_edges)
        for edge_id in delta.removed_edges:
            effective_edges.pop(edge_id, None)
        
        # Remove edges connected to removed nodes
        for node_id in delta.removed_nodes:
            edges_to_remove = [eid for eid, edge in effective_edges.items() 
                             if edge.source == node_id or edge.target == node_id]
            for eid in edges_to_remove:
                effective_edges.pop(eid, None)
        
        effective_attrs.update(delta.modified_graph_attrs)
        
        return effective_nodes, effective_edges, effective_attrs
    
    def add_node(self, node_id: str, **attributes) -> 'Graph':
        """Add node using copy-on-write"""
        if not self._is_modified:
            # Check if node already exists
            effective_nodes, _, _ = self._get_effective_data()
            if node_id in effective_nodes:
                return self
        
        self._init_delta()
        
        new_node = Node(node_id, attributes)
        self._pending_delta.added_nodes[node_id] = new_node
        
        return self
    
    def add_edge(self, source: str, target: str, **attributes) -> 'Graph':
        """Add edge using copy-on-write"""
        edge_id = f"{source}->{target}"
        
        # Ensure nodes exist
        effective_nodes, effective_edges, _ = self._get_effective_data()
        
        if edge_id in effective_edges:
            return self
        
        self._init_delta()
        
        # Add missing nodes
        if source not in effective_nodes and source not in self._pending_delta.added_nodes:
            self._pending_delta.added_nodes[source] = Node(source)
        if target not in effective_nodes and target not in self._pending_delta.added_nodes:
            self._pending_delta.added_nodes[target] = Node(target)
        
        new_edge = Edge(source, target, attributes)
        self._pending_delta.added_edges[edge_id] = new_edge
        
        return self
    
    def remove_node(self, node_id: str) -> 'Graph':
        """Remove node using copy-on-write"""
        effective_nodes, _, _ = self._get_effective_data()
        if node_id not in effective_nodes:
            return self
        
        self._init_delta()
        self._pending_delta.removed_nodes.add(node_id)
        
        return self
    
    def remove_edge(self, source: str, target: str) -> 'Graph':
        """Remove edge using copy-on-write"""
        edge_id = f"{source}->{target}"
        effective_nodes, effective_edges, _ = self._get_effective_data()
        
        if edge_id not in effective_edges:
            return self
        
        self._init_delta()
        self._pending_delta.removed_edges.add(edge_id)
        
        return self
    
    def set_node_attribute(self, node_id: str, key: str, value: Any) -> 'Graph':
        """Set node attribute using copy-on-write"""
        effective_nodes, _, _ = self._get_effective_data()
        if node_id not in effective_nodes:
            raise ValueError(f"Node {node_id} not found")
        
        self._init_delta()
        
        # Get current node (from effective state)
        current_node = effective_nodes[node_id]
        updated_node = current_node.set_attribute(key, value)
        self._pending_delta.modified_nodes[node_id] = updated_node
        
        return self
    
    def set_edge_attribute(self, source: str, target: str, key: str, value: Any) -> 'Graph':
        """Set edge attribute using copy-on-write"""
        edge_id = f"{source}->{target}"
        effective_nodes, effective_edges, _ = self._get_effective_data()
        
        if edge_id not in effective_edges:
            raise ValueError(f"Edge {edge_id} not found")
        
        self._init_delta()
        
        # Get current edge (from effective state)
        current_edge = effective_edges[edge_id]
        updated_edge = current_edge.set_attribute(key, value)
        self._pending_delta.modified_edges[edge_id] = updated_edge
        
        return self
    
    def snapshot(self) -> 'Graph':
        """Create immutable snapshot by applying all pending changes"""
        if not self._pending_delta:
            # No changes, can share data
            new_graph = Graph(self.nodes, self.edges, self.graph_attributes, self.graph_store)
            new_graph.node_order = self.node_order
            new_graph.edge_order = self.edge_order
            new_graph._current_time = self._current_time
            return new_graph
        
        # Apply changes and create new graph
        self._apply_pending_changes()
        
        new_graph = Graph(
            OrderedDict(self.nodes), 
            OrderedDict(self.edges), 
            self.graph_attributes.copy(), 
            self.graph_store
        )
        new_graph.node_order = self.node_order.copy()
        new_graph.edge_order = self.edge_order.copy()
        new_graph._current_time = self._current_time
        
        return new_graph
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbor node IDs"""
        effective_nodes, effective_edges, _ = self._get_effective_data()
        neighbors = []
        for edge in effective_edges.values():
            if edge.source == node_id:
                neighbors.append(edge.target)
            elif edge.target == node_id:
                neighbors.append(edge.source)
        return neighbors
    
    def get_nodes_by_time(self, start_time: float = 0, end_time: Optional[float] = None) -> List[Node]:
        """Get nodes added within time range"""
        if end_time is None:
            end_time = self._current_time
        
        effective_nodes, _, _ = self._get_effective_data()
        return [effective_nodes[node_id] for node_id, add_time in self.node_order.items()
                if start_time <= add_time <= end_time and node_id in effective_nodes]
    
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
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for module_config in config.get('modules', []):
            self.load_module(module_config)
    
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
    """Main Graph Store with optimized state management and git-like versioning"""
    
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
        
        # Initialize with empty state
        self._create_initial_state()
    
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
        nodes = OrderedDict()
        edges = OrderedDict()
        
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
    
    def update_graph(self, new_graph: Graph, operation: str = "update"):
        """Update current graph state using optimized delta storage"""
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


# Convenience functions for creating graphs
def create_random_graph(n_nodes: int = 10, edge_probability: float = 0.3) -> Graph:
    """Create a random graph for testing"""
    import random
    
    graph = Graph.empty()
    
    # Add nodes
    for i in range(n_nodes):
        graph = graph.add_node(f"node_{i}")
    
    # Add random edges  
    effective_nodes, _, _ = graph._get_effective_data()
    nodes = list(effective_nodes.keys())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if random.random() < edge_probability:
                graph = graph.add_edge(nodes[i], nodes[j])
    
    # Return snapshot to ensure all changes are applied
    return graph.snapshot()




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