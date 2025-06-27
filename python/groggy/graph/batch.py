"""
Batch operations context for efficient bulk operations
"""

from typing import TYPE_CHECKING
from ..data_structures import Node, Edge

if TYPE_CHECKING:
    from .core import Graph


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
    
    def set_node_attributes(self, node_attr_dict):
        """Queue bulk node attribute updates
        
        Args:
            node_attr_dict: Dict mapping node_id -> {attr: value}
        """
        for node_id, attributes in node_attr_dict.items():
            # Convert node_id to string for consistency
            node_id_str = str(node_id)
            
            if node_id_str in self.graph.nodes:
                # Update existing node attributes
                current_node = self.graph.nodes[node_id_str]
                new_attrs = current_node.attributes.copy()
                new_attrs.update(attributes)
                self.batch_nodes[node_id_str] = Node(node_id_str, new_attrs)
            elif node_id_str in self.batch_nodes:
                # Update pending batch node
                current_attrs = self.batch_nodes[node_id_str].attributes.copy()
                current_attrs.update(attributes)
                self.batch_nodes[node_id_str] = Node(node_id_str, current_attrs)
            else:
                # Create new node with attributes
                self.batch_nodes[node_id_str] = Node(node_id_str, attributes)
    
    def set_edge_attributes(self, edge_attr_dict):
        """Queue bulk edge attribute updates
        
        Args:
            edge_attr_dict: Dict mapping (source, target) -> {attr: value}
        """
        for (source, target), attributes in edge_attr_dict.items():
            source_str = str(source)
            target_str = str(target)
            edge_id = f"{source_str}->{target_str}"
            
            if edge_id in self.graph.edges:
                # Update existing edge attributes
                current_edge = self.graph.edges[edge_id]
                new_attrs = current_edge.attributes.copy()
                new_attrs.update(attributes)
                self.batch_edges[edge_id] = Edge(source_str, target_str, new_attrs)
            elif edge_id in self.batch_edges:
                # Update pending batch edge
                current_attrs = self.batch_edges[edge_id].attributes.copy()
                current_attrs.update(attributes)
                self.batch_edges[edge_id] = Edge(source_str, target_str, current_attrs)
            else:
                # Create new edge with attributes
                self.batch_edges[edge_id] = Edge(source_str, target_str, attributes)
