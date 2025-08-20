"""
Table Extensions - Add table() methods to Graph and Subgraph classes

This module adds DataFrame-like table() methods to existing graph classes.
"""

# GraphTable now comes from Rust FFI via __init__.py

def add_table_methods():
    """Add table() methods to Graph and related classes."""
    
    # NOTE: Graph classes already have table() methods implemented in Rust FFI
    # This extension is no longer needed as of the Rust migration
    pass  # No additional table methods needed

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
    
    # NOTE: Graph classes already have edges_table() methods implemented in Rust FFI
    # This extension is no longer needed as of the Rust migration
    pass

# Initialize table functionality when module is imported
add_table_methods()
add_edges_table_accessor()