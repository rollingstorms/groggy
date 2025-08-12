from typing import Dict, List, Optional, Tuple, Union
from .types import NodeId, EdgeId, AttrName, AttrValue, StateId, BranchName
from .errors import GroggyError, NodeNotFoundError, EdgeNotFoundError, NotImplementedError
from ._groggy import Graph as _RustGraph, AttrValue as _RustAttrValue

class Graph:
    """
    Main Graph interface - Python wrapper around Rust Graph implementation.
    
    This class provides a Pythonic interface to the high-performance Rust graph library,
    with memory optimization, Git-like version control, and advanced query capabilities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Create a new empty graph.
        
        Args:
            config: Optional configuration dictionary
        """
        self._rust_graph = _RustGraph(config)
    
    @classmethod  
    def load_from_path(cls, path: str) -> 'Graph':
        """
        Load an existing graph from storage.
        
        Args:
            path: Path to the saved graph file
            
        Returns:
            Graph instance loaded from file
            
        Raises:
            NotImplementedError: Feature not yet implemented
        """
        raise NotImplementedError("load_from_path")
    
    # === CORE GRAPH OPERATIONS ===
    
    def add_node(self) -> NodeId:
        """
        Add a new node to the graph.
        
        Returns:
            ID of the newly created node
        """
        return self._rust_graph.add_node()
    
    def add_nodes(self, count: int) -> List[NodeId]:
        """
        Add multiple nodes efficiently.
        
        Args:
            count: Number of nodes to create
            
        Returns:
            List of newly created node IDs
        """
        return self._rust_graph.add_nodes(count)
    
    def add_edge(self, source: NodeId, target: NodeId) -> EdgeId:
        """
        Add an edge between two existing nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            ID of the newly created edge
            
        Raises:
            NodeNotFoundError: If either node doesn't exist
        """
        try:
            return self._rust_graph.add_edge(source, target)
        except ValueError as e:
            # Convert generic ValueError to more specific errors
            error_msg = str(e)
            if "Node" in error_msg and "not found" in error_msg:
                raise NodeNotFoundError(source if "source" in error_msg else target, "add_edge") from e
            raise GroggyError(error_msg) from e
    
    def add_edges(self, edges: List[Tuple[NodeId, NodeId]]) -> List[EdgeId]:
        """
        Add multiple edges efficiently.
        
        Args:
            edges: List of (source, target) node ID pairs
            
        Returns:
            List of newly created edge IDs
        """
        try:
            return self._rust_graph.add_edges(edges)
        except ValueError as e:
            raise GroggyError(str(e)) from e
    
    def remove_node(self, node: NodeId) -> None:
        """
        Remove a node and all its incident edges.
        
        Args:
            node: Node ID to remove
            
        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        try:
            self._rust_graph.remove_node(node)
        except ValueError as e:
            error_msg = str(e)
            if "Node" in error_msg and "not found" in error_msg:
                raise NodeNotFoundError(node, "remove_node") from e
            raise GroggyError(error_msg) from e
    
    def remove_edge(self, edge: EdgeId) -> None:
        """
        Remove an edge from the graph.
        
        Args:
            edge: Edge ID to remove
            
        Raises:
            EdgeNotFoundError: If edge doesn't exist
        """
        try:
            self._rust_graph.remove_edge(edge)
        except ValueError as e:
            error_msg = str(e)
            if "Edge" in error_msg and "not found" in error_msg:
                raise EdgeNotFoundError(edge, "remove_edge") from e
            raise GroggyError(error_msg) from e
    
    def remove_nodes(self, nodes: List[NodeId]) -> None:
        """
        Remove multiple nodes efficiently.
        
        Args:
            nodes: List of node IDs to remove
        """
        try:
            self._rust_graph.remove_nodes(nodes)
        except ValueError as e:
            raise GroggyError(str(e)) from e
    
    def remove_edges(self, edges: List[EdgeId]) -> None:
        """
        Remove multiple edges efficiently.
        
        Args:
            edges: List of edge IDs to remove
        """
        try:
            self._rust_graph.remove_edges(edges)
        except ValueError as e:
            raise GroggyError(str(e)) from e
    
    # === ATTRIBUTE OPERATIONS ===
    
    def set_node_attribute(self, node: NodeId, attr: AttrName, value: AttrValue) -> None:
        """
        Set an attribute value on a node.
        
        Args:
            node: Node ID
            attr: Attribute name
            value: Attribute value
            
        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        # Convert Python AttrValue to Rust AttrValue
        rust_value = _RustAttrValue(value.value)
        try:
            self._rust_graph.set_node_attribute(node, attr, rust_value)
        except ValueError as e:
            error_msg = str(e)
            if "Node" in error_msg and "not found" in error_msg:
                raise NodeNotFoundError(node, "set_node_attribute") from e
            raise GroggyError(error_msg) from e
    
    def set_edge_attribute(self, edge: EdgeId, attr: AttrName, value: AttrValue) -> None:
        """
        Set an attribute value on an edge.
        
        Args:
            edge: Edge ID
            attr: Attribute name
            value: Attribute value
            
        Raises:
            EdgeNotFoundError: If edge doesn't exist
        """
        # Convert Python AttrValue to Rust AttrValue
        rust_value = _RustAttrValue(value.value)
        try:
            self._rust_graph.set_edge_attribute(edge, attr, rust_value)
        except ValueError as e:
            error_msg = str(e)
            if "Edge" in error_msg and "not found" in error_msg:
                raise EdgeNotFoundError(edge, "set_edge_attribute") from e
            raise GroggyError(error_msg) from e
    
    def get_node_attribute(self, node: NodeId, attr: AttrName) -> Optional[AttrValue]:
        """
        Get an attribute value from a node.
        
        Args:
            node: Node ID
            attr: Attribute name
            
        Returns:
            Attribute value if it exists, None otherwise
            
        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        try:
            rust_value = self._rust_graph.get_node_attribute(node, attr)
            if rust_value is None:
                return None
            # Convert Rust AttrValue back to Python AttrValue
            return AttrValue(rust_value.value)
        except ValueError as e:
            error_msg = str(e)
            if "Node" in error_msg and "not found" in error_msg:
                raise NodeNotFoundError(node, "get_node_attribute") from e
            raise GroggyError(error_msg) from e
    
    def get_edge_attribute(self, edge: EdgeId, attr: AttrName) -> Optional[AttrValue]:
        """
        Get an attribute value from an edge.
        
        Args:
            edge: Edge ID
            attr: Attribute name
            
        Returns:
            Attribute value if it exists, None otherwise
            
        Raises:
            EdgeNotFoundError: If edge doesn't exist
        """
        try:
            rust_value = self._rust_graph.get_edge_attribute(edge, attr)
            if rust_value is None:
                return None
            # Convert Rust AttrValue back to Python AttrValue
            return AttrValue(rust_value.value)
        except ValueError as e:
            error_msg = str(e)
            if "Edge" in error_msg and "not found" in error_msg:
                raise EdgeNotFoundError(edge, "get_edge_attribute") from e
            raise GroggyError(error_msg) from e
    
    def get_node_attributes(self, node: NodeId) -> Dict[AttrName, AttrValue]:
        """
        Get all attributes for a node.
        
        Args:
            node: Node ID
            
        Returns:
            Dictionary mapping attribute names to values
            
        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        try:
            rust_attrs = self._rust_graph.get_node_attributes(node)
            # Convert Rust AttrValue objects to Python AttrValue objects
            python_attrs = {}
            for attr_name, rust_value in rust_attrs.items():
                python_attrs[attr_name] = AttrValue(rust_value.value)
            return python_attrs
        except ValueError as e:
            error_msg = str(e)
            if "Node" in error_msg and "not found" in error_msg:
                raise NodeNotFoundError(node, "get_node_attributes") from e
            raise GroggyError(error_msg) from e
    
    def get_edge_attributes(self, edge: EdgeId) -> Dict[AttrName, AttrValue]:
        """
        Get all attributes for an edge.
        
        Args:
            edge: Edge ID
            
        Returns:
            Dictionary mapping attribute names to values
            
        Raises:
            EdgeNotFoundError: If edge doesn't exist
        """
        try:
            rust_attrs = self._rust_graph.get_edge_attributes(edge)
            # Convert Rust AttrValue objects to Python AttrValue objects
            python_attrs = {}
            for attr_name, rust_value in rust_attrs.items():
                python_attrs[attr_name] = AttrValue(rust_value.value)
            return python_attrs
        except ValueError as e:
            error_msg = str(e)
            if "Edge" in error_msg and "not found" in error_msg:
                raise EdgeNotFoundError(edge, "get_edge_attributes") from e
            raise GroggyError(error_msg) from e
    
    # === BULK ATTRIBUTE OPERATIONS (Phase 2) ===
    
    def set_node_attributes(self, attrs: Dict[AttrName, List[Tuple[NodeId, AttrValue]]]) -> None:
        """
        Set multiple attributes on multiple nodes efficiently.
        
        Args:
            attrs: Dictionary mapping attribute names to lists of (node_id, value) pairs
            
        Example:
            graph.set_node_attributes({
                "name": [(1, AttrValue("Alice")), (2, AttrValue("Bob"))],
                "age": [(1, AttrValue(25)), (2, AttrValue(30))]
            })
            
        Raises:
            NodeNotFoundError: If any node doesn't exist
        """
        # Convert Python AttrValue objects to Rust AttrValue objects
        rust_attrs = {}
        for attr_name, pairs in attrs.items():
            rust_pairs = []
            for node_id, attr_value in pairs:
                rust_value = _RustAttrValue(attr_value.value)
                rust_pairs.append((node_id, rust_value))
            rust_attrs[attr_name] = rust_pairs
        
        try:
            self._rust_graph.set_node_attributes(rust_attrs)
        except ValueError as e:
            error_msg = str(e)
            if "Node" in error_msg and "not found" in error_msg:
                raise NodeNotFoundError(-1, "set_node_attributes") from e
            raise GroggyError(error_msg) from e
    
    def set_edge_attributes(self, attrs: Dict[AttrName, List[Tuple[EdgeId, AttrValue]]]) -> None:
        """
        Set multiple attributes on multiple edges efficiently.
        
        Args:
            attrs: Dictionary mapping attribute names to lists of (edge_id, value) pairs
            
        Example:
            graph.set_edge_attributes({
                "weight": [(1, AttrValue(0.9)), (2, AttrValue(0.8))],
                "type": [(1, AttrValue("friend")), (2, AttrValue("colleague"))]
            })
            
        Raises:
            EdgeNotFoundError: If any edge doesn't exist
        """
        # Convert Python AttrValue objects to Rust AttrValue objects
        rust_attrs = {}
        for attr_name, pairs in attrs.items():
            rust_pairs = []
            for edge_id, attr_value in pairs:
                rust_value = _RustAttrValue(attr_value.value)
                rust_pairs.append((edge_id, rust_value))
            rust_attrs[attr_name] = rust_pairs
        
        try:
            self._rust_graph.set_edge_attributes(rust_attrs)
        except ValueError as e:
            error_msg = str(e)
            if "Edge" in error_msg and "not found" in error_msg:
                raise EdgeNotFoundError(-1, "set_edge_attributes") from e
            raise GroggyError(error_msg) from e
    
    def get_nodes_attributes(self, attr: AttrName, nodes: List[NodeId]) -> List[Optional[AttrValue]]:
        """
        Get a single attribute from multiple nodes efficiently.
        
        Args:
            attr: Attribute name to retrieve
            nodes: List of node IDs to get the attribute from
            
        Returns:
            List of attribute values (None if attribute doesn't exist for a node)
            
        Example:
            names = graph.get_nodes_attributes("name", [1, 2, 3])
            # Returns [AttrValue("Alice"), None, AttrValue("Charlie")]
        """
        try:
            rust_values = self._rust_graph.get_nodes_attributes(attr, nodes)
            # Convert Rust AttrValue objects to Python AttrValue objects
            python_values = []
            for rust_value in rust_values:
                if rust_value is None:
                    python_values.append(None)
                else:
                    python_values.append(AttrValue(rust_value.value))
            return python_values
        except ValueError as e:
            raise GroggyError(str(e)) from e
    
    def get_edges_attributes(self, attr: AttrName, edges: List[EdgeId]) -> List[Optional[AttrValue]]:
        """
        Get a single attribute from multiple edges efficiently.
        
        Args:
            attr: Attribute name to retrieve
            edges: List of edge IDs to get the attribute from
            
        Returns:
            List of attribute values (None if attribute doesn't exist for an edge)
            
        Example:
            weights = graph.get_edges_attributes("weight", [1, 2, 3])
            # Returns [AttrValue(0.9), AttrValue(0.8), None]
        """
        try:
            rust_values = self._rust_graph.get_edges_attributes(attr, edges)
            # Convert Rust AttrValue objects to Python AttrValue objects
            python_values = []
            for rust_value in rust_values:
                if rust_value is None:
                    python_values.append(None)
                else:
                    python_values.append(AttrValue(rust_value.value))
            return python_values
        except ValueError as e:
            raise GroggyError(str(e)) from e
    
    # === TOPOLOGY OPERATIONS ===
    
    def contains_node(self, node: NodeId) -> bool:
        """Check if a node exists in the graph."""
        return self._rust_graph.contains_node(node)
    
    def contains_edge(self, edge: EdgeId) -> bool:
        """Check if an edge exists in the graph."""
        return self._rust_graph.contains_edge(edge)
    
    def node_ids(self) -> List[NodeId]:
        """Get all active node IDs."""
        return self._rust_graph.node_ids()
    
    def edge_ids(self) -> List[EdgeId]:
        """Get all active edge IDs."""
        return self._rust_graph.edge_ids()
    
    def edge_endpoints(self, edge: EdgeId) -> Tuple[NodeId, NodeId]:
        """
        Get the endpoints of an edge.
        
        Args:
            edge: Edge ID
            
        Returns:
            Tuple of (source, target) node IDs
            
        Raises:
            EdgeNotFoundError: If edge doesn't exist
        """
        try:
            return self._rust_graph.edge_endpoints(edge)
        except ValueError as e:
            error_msg = str(e)
            if "Edge" in error_msg and "not found" in error_msg:
                raise EdgeNotFoundError(edge, "edge_endpoints") from e
            raise GroggyError(error_msg) from e
    
    def neighbors(self, node: NodeId) -> List[NodeId]:
        """
        Get all neighbors of a node.
        
        Args:
            node: Node ID
            
        Returns:
            List of neighboring node IDs
            
        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        try:
            return self._rust_graph.neighbors(node)
        except ValueError as e:
            error_msg = str(e)
            if "Node" in error_msg and "not found" in error_msg:
                raise NodeNotFoundError(node, "neighbors") from e
            raise GroggyError(error_msg) from e
    
    def degree(self, node: NodeId) -> int:
        """
        Get the degree (number of incident edges) of a node.
        
        Args:
            node: Node ID
            
        Returns:
            Number of incident edges
            
        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        try:
            return self._rust_graph.degree(node)
        except ValueError as e:
            error_msg = str(e)
            if "Node" in error_msg and "not found" in error_msg:
                raise NodeNotFoundError(node, "degree") from e
            raise GroggyError(error_msg) from e
    
    # === STATISTICS AND ANALYSIS ===
    
    def statistics(self) -> Dict:
        """
        Get comprehensive graph statistics.
        
        Returns:
            Dictionary containing graph statistics
        """
        return self._rust_graph.statistics()
    
    def memory_statistics(self) -> Dict:
        """
        Get detailed memory usage statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        return self._rust_graph.memory_statistics()
    
    def __repr__(self) -> str:
        return str(self._rust_graph)
    
    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.node_ids())
    
    # === FUTURE PHASES - TO BE IMPLEMENTED ===
    
    def commit(self, message: str, author: str) -> StateId:
        """Commit current changes to create a new state."""
        raise NotImplementedError("commit")
    
    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        raise NotImplementedError("has_uncommitted_changes")
    
    def create_branch(self, branch_name: BranchName) -> None:
        """Create a new branch."""
        raise NotImplementedError("create_branch")
    
    def checkout_branch(self, branch_name: BranchName) -> None:
        """Switch to a different branch."""
        raise NotImplementedError("checkout_branch")
