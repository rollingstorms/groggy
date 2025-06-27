"""
Subgraph module for GLI.

A simple Subgraph class that extends Graph with metadata about its origin.
"""

from typing import Optional, Dict, Any
from .core import Graph


class Subgraph(Graph):
    """
    A Subgraph is a Graph with additional metadata about its origin.
    
    This is a simple extension of Graph that tracks:
    - The parent graph it was created from
    - The filter criteria used to create it
    - Any additional metadata
    """
    
    def __init__(self, parent_graph: Optional[Graph] = None, 
                 filter_criteria: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize a Subgraph.
        
        Args:
            parent_graph: The Graph this subgraph was created from
            filter_criteria: The filter string/criteria used to create this subgraph
            metadata: Additional metadata about this subgraph
            **kwargs: Arguments passed to the parent Graph constructor
        """
        super().__init__(**kwargs)
        
        self.parent_graph = parent_graph
        self.filter_criteria = filter_criteria
        self.metadata = metadata or {}
    
    def __repr__(self):
        """String representation of the subgraph."""
        base_repr = super().__repr__()
        if self.filter_criteria:
            return f"Subgraph(filter='{self.filter_criteria}', {base_repr})"
        return f"Subgraph({base_repr})"
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get all metadata about this subgraph."""
        return {
            'parent_graph': self.parent_graph,
            'filter_criteria': self.filter_criteria,
            'metadata': self.metadata
        }
