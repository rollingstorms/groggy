"""
Table Extensions - Add table() methods to Graph and Subgraph classes

This module adds DataFrame-like table() methods to existing graph classes.
"""

from .graph_table import GraphTable

def add_table_methods():
    """Add table() methods to Graph and related classes."""
    
    # Try to import the graph classes
    try:
        from ._groggy import Graph
        
        def graph_table(self):
            """Return a GraphTable view of all nodes."""
            return GraphTable(self, "nodes")
        
        # Add table method to Graph
        Graph.table = graph_table
        
    except ImportError:
        pass
    
    # Add table method to enhanced subgraphs
    try:
        from .enhanced_query import EnhancedSubgraph
        
        def enhanced_subgraph_table(self):
            """Return a GraphTable view of subgraph nodes."""
            return GraphTable(self, "nodes")
        
        EnhancedSubgraph.table = enhanced_subgraph_table
        
    except ImportError:
        pass

# Add edges table access
class EdgesTableAccessor:
    """Accessor for edges table functionality."""
    
    def __init__(self, graph_or_subgraph):
        self.graph_or_subgraph = graph_or_subgraph
    
    def table(self):
        """Return a GraphTable view of edges."""
        return GraphTable(self.graph_or_subgraph, "edges")

def add_edges_table_accessor():
    """Add edges.table() accessor to Graph classes."""
    
    try:
        from ._groggy import Graph
        
        # Store original edges property if it exists
        original_edges = getattr(Graph, 'edges', None)
        
        def enhanced_edges(self):
            """Enhanced edges accessor with table functionality."""
            accessor = EdgesTableAccessor(self)
            # If there's an original edges implementation, add it as an attribute
            if original_edges:
                accessor.original = original_edges.__get__(self, Graph)
            return accessor
        
        Graph.edges_table = enhanced_edges
        
    except ImportError:
        pass

# Initialize table functionality when module is imported
add_table_methods()
add_edges_table_accessor()