# python_new/groggy/graph/subgraph.py

class Subgraph:
    """
    A Subgraph is a Graph with additional metadata about its origin.
    
    Represents a filtered or derived view of a parent graph, with explicit provenance and filter criteria stored as metadata.
    Supports composable subgraph creation and metadata inspection.
    """

    def __init__(self, parent_graph, filter_criteria, metadata):
        """
        Initialize a Subgraph with parent graph, filter criteria, and metadata.
        
        Args:
            parent_graph (Graph): The parent graph instance.
            filter_criteria (dict or callable): Criteria used to derive the subgraph.
            metadata (dict): Provenance and creation metadata.
        """
        self.parent_graph = parent_graph
        self.filter_criteria = filter_criteria
        self.metadata = metadata or {}
        # Filtered collections: assume parent_graph.nodes/edges support filter()
        self.nodes = parent_graph.nodes.filter(**filter_criteria) if filter_criteria else parent_graph.nodes
        self.edges = parent_graph.edges.filter(**filter_criteria) if filter_criteria else parent_graph.edges

    def __repr__(self):
        """
        Return a string representation of the subgraph, including metadata summary.
        
        Useful for diagnostics and provenance inspection.
        Returns:
            str: Summary string.
        """
        meta = {k: v for k, v in self.metadata.items()}
        return f"<Subgraph nodes={len(self.nodes)} edges={len(self.edges)} meta={meta}>"

    def get_metadata(self):
        """
        Get all provenance and creation metadata for this subgraph.
        
        Returns:
            dict: Metadata dictionary.
        """
        return self.metadata
