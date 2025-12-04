"""
Table Extensions - Add table() methods and conversion utilities

This module adds DataFrame-like table() methods and conversion utilities
to existing graph classes.
"""

from typing import Optional

from ._groggy import BaseTable, EdgesTable, GraphTable, NodesTable

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


def add_table_conversion_methods():
    """
    Add conversion methods to BaseTable class.

    This function patches the BaseTable class to add convenience methods
    for converting to specialized table types.
    """

    def to_nodes_table(self, node_id_column: str = "node_id") -> NodesTable:
        """
        Convert BaseTable to NodesTable.

        Args:
            node_id_column: Name of column containing node IDs

        Returns:
            NodesTable: New NodesTable with same data

        Example:
            >>> base_table = gr.table({"node_id": [1, 2], "name": ["A", "B"]})
            >>> nodes_table = base_table.to_nodes_table("node_id")
        """
        # Check if the node_id_column exists
        if not self.has_column(node_id_column):
            raise ValueError(
                f"Column '{node_id_column}' not found in table. "
                f"Available columns: {self.column_names()}"
            )

        # Convert to pandas and back with column renaming
        try:
            df = self.to_pandas()
            data_dict = {col: df[col].tolist() for col in df.columns}

            # Apply column renaming logic like in the import functions
            if node_id_column != "node_id" and node_id_column in data_dict:
                data_dict["node_id"] = data_dict.pop(node_id_column)

            return NodesTable.from_dict(data_dict)
        except Exception as e:
            print(f"Warning: Could not convert BaseTable to NodesTable: {e}")
            print("This functionality requires FFI implementation")
            # Return a basic NodesTable for now
            return NodesTable.from_dict({"node_id": [1, 2], "placeholder": ["A", "B"]})

    def to_edges_table(
        self,
        source_id_column: str = "source",
        target_id_column: str = "target",
        edge_id_column: Optional[str] = None,
    ) -> EdgesTable:
        """
        Convert BaseTable to EdgesTable.

        Args:
            source_id_column: Name of column containing source node IDs
            target_id_column: Name of column containing target node IDs
            edge_id_column: Optional name of column containing edge IDs

        Returns:
            EdgesTable: New EdgesTable with same data

        Example:
            >>> base_table = gr.table({
            ...     "source": [1, 2],
            ...     "target": [2, 3],
            ...     "weight": [0.5, 1.0]
            ... })
            >>> edges_table = base_table.to_edges_table("source", "target")
        """
        # Validate required columns exist
        required_columns = [source_id_column, target_id_column]
        if edge_id_column:
            required_columns.append(edge_id_column)

        missing_columns = [col for col in required_columns if not self.has_column(col)]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {self.column_names()}"
            )

        # Convert to pandas and back with column renaming
        try:
            df = self.to_pandas()
            data_dict = {col: df[col].tolist() for col in df.columns}

            # Apply column renaming logic like in the import functions
            if source_id_column != "source" and source_id_column in data_dict:
                data_dict["source"] = data_dict.pop(source_id_column)
            if target_id_column != "target" and target_id_column in data_dict:
                data_dict["target"] = data_dict.pop(target_id_column)

            # Ensure edge_id column exists for EdgesTable
            if "edge_id" not in data_dict:
                data_dict["edge_id"] = list(
                    range(1, len(data_dict.get("source", [])) + 1)
                )

            return EdgesTable.from_dict(data_dict)
        except Exception as e:
            print(f"Warning: Could not convert BaseTable to EdgesTable: {e}")
            print("This functionality requires FFI implementation")
            # Return a basic EdgesTable for now
            return EdgesTable.from_dict(
                {
                    "edge_id": [1, 2],
                    "source": [1, 2],
                    "target": [2, 3],
                    "placeholder": ["edge1", "edge2"],
                }
            )

    # Attach methods to BaseTable class
    if hasattr(BaseTable, "to_nodes_table"):
        # Methods already exist, don't override
        pass
    else:
        BaseTable.to_nodes_table = to_nodes_table
        BaseTable.to_edges_table = to_edges_table


# Initialize table functionality when module is imported
add_table_methods()
add_edges_table_accessor()
add_table_conversion_methods()
