"""
Delta tracking for graph modifications
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Any
from .data_structures import Node, Edge


@dataclass
class GraphDelta:
    """Tracks pending changes to avoid full graph copies"""
    added_nodes: Dict[str, Node] = field(default_factory=dict)
    removed_nodes: Set[str] = field(default_factory=set)
    modified_nodes: Dict[str, Node] = field(default_factory=dict)
    added_edges: Dict[str, Edge] = field(default_factory=dict)
    removed_edges: Set[str] = field(default_factory=set)
    modified_edges: Dict[str, Edge] = field(default_factory=dict)
    modified_graph_attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompactGraphDelta:
    """Compact delta using content hashes instead of full objects"""
    added_nodes: Dict[str, str] = field(default_factory=dict)  # node_id -> content_hash
    removed_nodes: Set[str] = field(default_factory=set)  # node_ids
    modified_nodes: Dict[str, str] = field(default_factory=dict)  # node_id -> content_hash
    added_edges: Dict[str, str] = field(default_factory=dict)  # edge_id -> content_hash
    removed_edges: Set[str] = field(default_factory=set)  # edge_ids
    modified_edges: Dict[str, str] = field(default_factory=dict)  # edge_id -> content_hash
    modified_graph_attrs: Dict[str, Any] = field(default_factory=dict)
    
    def is_empty(self) -> bool:
        """Check if delta has any changes"""
        return not any([
            self.added_nodes, self.removed_nodes, self.modified_nodes,
            self.added_edges, self.removed_edges, self.modified_edges,
            self.modified_graph_attrs
        ])
