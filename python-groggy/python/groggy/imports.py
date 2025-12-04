"""
Data Import Functionality for Groggy

This module provides comprehensive data import capabilities from various formats
including CSV, pandas DataFrames, numpy arrays, and more.
"""

import os
from typing import Any, Dict, List, Optional, Union

from ._groggy import (BaseTable, EdgesTable, Graph, GraphTable, NodesTable,
                      array, matrix, num_array, table)

# Import pandas for NaN checking - optional import
try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

    def _isna_fallback(x):
        """Fallback NaN checking without pandas"""
        return x is None or str(x).lower() in ["nan", "none", ""]


def _handle_unknown_result_types(
    nodes_dict: Dict[str, List],
    edges_dict: Dict[str, List],
    node_mapping: Dict[str, int],
    node_id_column: str,
    source_id_column: str,
    target_id_column: str,
):
    """
    Handle methods with unknown return types (NaN) from comprehensive_library_testing.py.

    Creates synthetic nodes for both missing result types AND unknown return types
    so we don't lose method information.
    """
    # Get all target values from edges, including handling NaN specially
    target_values = set()
    unknown_count = 0

    if target_id_column in edges_dict:
        for x in edges_dict[target_id_column]:
            if x is not None and not (
                pd.isna(x) if _PANDAS_AVAILABLE else _isna_fallback(x)
            ):
                str_x = str(x).lower()
                if str_x not in ["nan", "none", ""]:
                    target_values.add(str(x))
            else:
                # Count NaN/unknown values
                unknown_count += 1

    # Add a special node for unknown return types
    if unknown_count > 0:
        target_values.add("unknown_return_type")

    # Find missing target values that aren't in the node mapping
    missing_targets = target_values - set(node_mapping.keys())

    if len(missing_targets) > 0:
        print(
            f"🔧 Auto-expanding nodes to include {len(missing_targets)} result types (including {unknown_count} unknown)"
        )
        print(
            f"   Added: {sorted(list(missing_targets))[:5]}{'...' if len(missing_targets) > 5 else ''}"
        )

        # Create expanded node mapping
        expanded_mapping = node_mapping.copy()
        next_id = len(node_mapping)

        for target in sorted(missing_targets):
            expanded_mapping[target] = next_id
            next_id += 1

        # Create expanded nodes dict with synthetic nodes for missing types
        expanded_nodes_dict = nodes_dict.copy()

        # Add the new synthetic nodes to the nodes data
        for col in expanded_nodes_dict:
            if col == node_id_column:
                # Add the missing target values as new node IDs
                expanded_nodes_dict[col].extend(list(missing_targets))
            else:
                # Infer the column type from existing values to maintain type consistency
                existing_values = expanded_nodes_dict[col]

                # Try to detect if this is a numeric column
                is_numeric = False
                if existing_values:
                    # Check first non-null value
                    for val in existing_values:
                        if val is not None and not (
                            pd.isna(val) if _PANDAS_AVAILABLE else _isna_fallback(val)
                        ):
                            # Check if it's numeric
                            try:
                                float(val)
                                is_numeric = True
                                break
                            except (ValueError, TypeError):
                                break

                # Add appropriate default values based on column type
                if is_numeric:
                    # For numeric columns, use 0 as default for synthetic nodes
                    expanded_nodes_dict[col].extend([0 for _ in missing_targets])
                else:
                    # For text columns, add descriptive values
                    expanded_nodes_dict[col].extend(
                        [
                            (
                                "Unknown return type"
                                if target == "unknown_return_type"
                                else f"Type: {target}"
                            )
                            for target in missing_targets
                        ]
                    )

        # Update edges to use "unknown_return_type" for NaN values
        if target_id_column in edges_dict:
            updated_targets = []
            for x in edges_dict[target_id_column]:
                if (
                    x is None
                    or (pd.isna(x) if _PANDAS_AVAILABLE else _isna_fallback(x))
                    or str(x).lower() in ["nan", "none", ""]
                ):
                    updated_targets.append("unknown_return_type")
                else:
                    updated_targets.append(x)
            edges_dict[target_id_column] = updated_targets

        return expanded_nodes_dict, expanded_mapping

    return nodes_dict, node_mapping


def _suggest_column_mapping(
    mapped_dict: Dict[str, List],
    source_column: str,
    target_column: str,
    node_mapping: Dict[str, int],
    filtered_count: int,
):
    """
    Provide concise suggestions when many edges are being filtered due to unmapped values.
    """
    # Find the best alternative column
    available_columns = [
        col for col in mapped_dict.keys() if col not in ["_node_mapping"]
    ]
    best_suggestion = None

    for col in available_columns:
        if col != source_column and col in mapped_dict:
            col_values = set(str(x) for x in mapped_dict[col][:50] if x is not None)
            overlap = col_values & set(node_mapping.keys())
            if len(overlap) > 0:
                best_suggestion = col
                break

    if best_suggestion:
        print(
            f"💡 Did you mean: source_id_column='{best_suggestion}'? ('{source_column}' values don't exist as nodes)"
        )


def _apply_node_id_mapping(
    data_dict: Dict[str, List],
    table_type: str,
    node_id_column: str = "node_id",
    source_id_column: str = "source",
    target_id_column: str = "target",
) -> Dict[str, List]:
    """
    Apply node ID mapping to convert string identifiers to integers.

    This is the core mapping function used by all import functions to ensure
    compatibility with Groggy's integer-based graph representation.

    Args:
        data_dict: Dictionary with column names as keys, lists as values
        table_type: "base", "nodes", "edges", or "graph"
        node_id_column: Column name containing node identifiers
        source_id_column: Column name containing source node identifiers
        target_id_column: Column name containing target node identifiers

    Returns:
        Dictionary with mapped integer node IDs and standardized column names
    """
    # Make a copy to avoid modifying original data
    mapped_dict = data_dict.copy()

    if table_type == "nodes":
        # Create mapping from string node IDs to integers
        if node_id_column in mapped_dict:
            # IMPORTANT: Sort unique nodes to ensure deterministic ID assignment
            unique_nodes = sorted(set(mapped_dict[node_id_column]))
            node_mapping = {str(node): i for i, node in enumerate(unique_nodes)}

            # Add new mapped column alongside original
            mapped_dict["node_id"] = [
                node_mapping[str(node)] for node in mapped_dict[node_id_column]
            ]

            # Keep original column - don't delete it!
            # The original column (like 'object_name') stays for reference

            # Store mapping for potential use in edges
            mapped_dict["_node_mapping"] = node_mapping

    elif table_type == "edges":
        # For edges, we need to map source and target columns
        node_mapping = mapped_dict.pop("_node_mapping", None)

        if not node_mapping:
            # Create mapping from unique values in source and target columns
            all_node_ids = set()
            if source_id_column in mapped_dict:
                all_node_ids.update(
                    str(x)
                    for x in mapped_dict[source_id_column]
                    if x is not None
                    and str(x).lower() not in ["nan", "none", ""]
                    and not (pd.isna(x) if _PANDAS_AVAILABLE else _isna_fallback(x))
                )
            if target_id_column in mapped_dict:
                all_node_ids.update(
                    str(x)
                    for x in mapped_dict[target_id_column]
                    if x is not None
                    and str(x).lower() not in ["nan", "none", ""]
                    and not (pd.isna(x) if _PANDAS_AVAILABLE else _isna_fallback(x))
                )

            node_mapping = {str(node): i for i, node in enumerate(sorted(all_node_ids))}

        # Apply mapping to source column - add new column alongside original
        if source_id_column in mapped_dict:
            mapped_dict["source"] = [
                (
                    node_mapping.get(str(x), None)
                    if x is not None
                    and str(x).lower() not in ["nan", "none", ""]
                    and not (pd.isna(x) if _PANDAS_AVAILABLE else _isna_fallback(x))
                    else None
                )
                for x in mapped_dict[source_id_column]
            ]
            # Keep original source column (like 'object_name') for reference

        # Apply mapping to target column - add new column alongside original
        if target_id_column in mapped_dict:
            mapped_dict["target"] = [
                (
                    node_mapping.get(str(x), None)
                    if x is not None
                    and str(x).lower() not in ["nan", "none", ""]
                    and not (pd.isna(x) if _PANDAS_AVAILABLE else _isna_fallback(x))
                    else None
                )
                for x in mapped_dict[target_id_column]
            ]
            # Keep original target column (like 'result_type') for reference

    # Remove any null values that would cause validation errors
    if table_type == "edges":
        # Filter out edges with null source or target BEFORE creating edge_id
        source_list = mapped_dict.get("source", [])
        target_list = mapped_dict.get("target", [])

        valid_indices = [
            i
            for i, (src, tgt) in enumerate(zip(source_list, target_list))
            if src is not None and tgt is not None
        ]

        if len(valid_indices) < len(source_list):
            filtered_count = len(source_list) - len(valid_indices)
            print(f"⚠️  Filtered out {filtered_count} edges with null source/target")

            # Provide intelligent suggestions if most/all edges are being filtered
            if filtered_count > len(source_list) * 0.5:  # More than 50% filtered
                _suggest_column_mapping(
                    mapped_dict,
                    source_id_column,
                    target_id_column,
                    node_mapping,
                    filtered_count,
                )

            # Keep only valid entries for ALL columns
            filtered_dict = {}
            for key, values in mapped_dict.items():
                if isinstance(values, list) and len(values) == len(source_list):
                    filtered_dict[key] = [values[i] for i in valid_indices]
                else:
                    # Keep non-list values as-is
                    filtered_dict[key] = values

            mapped_dict = filtered_dict

        # Ensure edge_id column exists AFTER filtering
        if "edge_id" not in mapped_dict:
            mapped_dict["edge_id"] = list(range(len(mapped_dict.get("source", []))))
    else:
        # For non-edge tables, ensure edge_id exists if needed
        if table_type == "edges" and "edge_id" not in mapped_dict:
            mapped_dict["edge_id"] = list(range(len(mapped_dict.get("source", []))))

    return mapped_dict


def _create_table_with_mapping(
    data_dict: Dict[str, List],
    table_type: str,
    node_id_column: str = "node_id",
    source_id_column: str = "source",
    target_id_column: str = "target",
) -> Union[BaseTable, NodesTable, EdgesTable]:
    """
    Create a Groggy table with automatic node ID mapping applied.

    This function handles the common pattern of:
    1. Apply node ID mapping to convert strings to integers
    2. Validate column requirements
    3. Create appropriate table type

    Args:
        data_dict: Raw data dictionary
        table_type: "base", "nodes", or "edges"
        node_id_column: Column name for node IDs
        source_id_column: Column name for source node IDs
        target_id_column: Column name for target node IDs

    Returns:
        Appropriate table type with integer node IDs
    """
    # Apply mapping if needed
    if table_type in ["nodes", "edges"]:
        mapped_dict = _apply_node_id_mapping(
            data_dict, table_type, node_id_column, source_id_column, target_id_column
        )
    else:
        mapped_dict = data_dict.copy()

    # Remove internal mapping data
    mapped_dict.pop("_node_mapping", None)

    # Create appropriate table type
    if table_type == "base":
        return BaseTable.from_dict(mapped_dict)
    elif table_type == "nodes":
        return NodesTable.from_dict(mapped_dict)
    elif table_type == "edges":
        return EdgesTable.from_dict(mapped_dict)
    else:
        raise ValueError(f"Unknown table_type: {table_type}")


def from_csv(
    filepath: Optional[str] = None,
    *,
    # Graph creation options
    nodes_filepath: Optional[str] = None,
    edges_filepath: Optional[str] = None,
    node_id_column: str = "node_id",
    source_id_column: str = "source",
    target_id_column: str = "target",
    # CSV parsing options
    **kwargs,
) -> Union[BaseTable, GraphTable]:
    """
    Load data from CSV file(s) into Groggy tables or graphs.

    This function provides flexible CSV import with automatic graph construction
    when node and edge files are provided.

    Args:
        filepath: Path to single CSV file (for single table import)
        nodes_filepath: Path to nodes CSV file for graph creation
        edges_filepath: Path to edges CSV file for graph creation
        node_id_column: Column name for node IDs (default: "node_id")
        source_id_column: Column name for source node IDs (default: "source")
        target_id_column: Column name for target node IDs (default: "target")
        **kwargs: Additional CSV parsing options

    Returns:
        BaseTable: If loading single CSV file
        GraphTable: If loading nodes + edges CSV files

    Examples:
        # Load single table
        >>> table = gr.from_csv("data.csv")

        # Load graph from separate node/edge files
        >>> graph_table = gr.from_csv(
        ...     nodes_filepath="nodes.csv",
        ...     edges_filepath="edges.csv",
        ...     node_id_column="id",
        ...     source_id_column="from",
        ...     target_id_column="to"
        ... )

        # Alternative: specify single file as filepath
        >>> table = gr.from_csv(filepath="single_file.csv")
    """

    # Handle different calling patterns
    if nodes_filepath is None and edges_filepath is None:
        # Single file mode - load as BaseTable
        if filepath is None:
            raise ValueError(
                "Must provide either 'filepath' for single file or 'nodes_filepath'/'edges_filepath' for graph creation"
            )

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        return BaseTable.from_csv(filepath)

    elif edges_filepath is not None:
        # Check if we have both nodes and edges, or just edges
        if nodes_filepath is None:
            # Only edges file specified - load as EdgesTable with column mapping
            if not os.path.exists(edges_filepath):
                raise FileNotFoundError(f"Edges CSV file not found: {edges_filepath}")

            # Load as BaseTable first, then convert with column mapping
            edges_base = BaseTable.from_csv(edges_filepath)
            return from_pandas(
                edges_base.to_pandas(),
                table_type="edges",
                source_id_column=source_id_column,
                target_id_column=target_id_column,
            )

        # Both nodes and edges files specified - create GraphTable

        nodes_file = nodes_filepath
        edges_file = edges_filepath

        if not os.path.exists(nodes_file):
            raise FileNotFoundError(f"Nodes CSV file not found: {nodes_file}")
        if not os.path.exists(edges_file):
            raise FileNotFoundError(f"Edges CSV file not found: {edges_file}")

        # Load the tables and apply unified mapping system
        nodes_base = BaseTable.from_csv(nodes_file)
        edges_base = BaseTable.from_csv(edges_file)

        # Convert to pandas for processing
        nodes_df = nodes_base.to_pandas()
        edges_df = edges_base.to_pandas()

        # Convert to dict format for unified processing
        nodes_dict = {col: nodes_df[col].tolist() for col in nodes_df.columns}
        edges_dict = {col: edges_df[col].tolist() for col in edges_df.columns}

        # Create nodes table with mapping - this generates the node mapping
        nodes_mapped_dict = _apply_node_id_mapping(
            nodes_dict, "nodes", node_id_column, source_id_column, target_id_column
        )
        node_mapping = nodes_mapped_dict.pop("_node_mapping", {})

        # Handle methods with unknown return types by creating synthetic target nodes
        expanded_nodes_dict, expanded_node_mapping = _handle_unknown_result_types(
            nodes_dict,
            edges_dict,
            node_mapping,
            node_id_column,
            source_id_column,
            target_id_column,
        )

        # Apply the expanded mapping to edges
        edges_dict["_node_mapping"] = expanded_node_mapping
        edges_mapped_dict = _apply_node_id_mapping(
            edges_dict, "edges", node_id_column, source_id_column, target_id_column
        )

        # Manually apply the expanded_node_mapping to the expanded nodes
        # DO NOT call _apply_node_id_mapping again as it would re-create the mapping!
        expanded_nodes_mapped_dict = expanded_nodes_dict.copy()
        if node_id_column in expanded_nodes_mapped_dict:
            expanded_nodes_mapped_dict["node_id"] = [
                expanded_node_mapping[str(node)]
                for node in expanded_nodes_mapped_dict[node_id_column]
            ]

        nodes_table = NodesTable.from_dict(
            {
                k: v
                for k, v in expanded_nodes_mapped_dict.items()
                if k != "_node_mapping"
            }
        )
        edges_table = EdgesTable.from_dict(
            {k: v for k, v in edges_mapped_dict.items() if k != "_node_mapping"}
        )

        # Create GraphTable from NodesTable and EdgesTable
        try:
            from ._groggy import GraphTable

            graph_table = GraphTable(nodes_table, edges_table)
            print(f"✅ Created GraphTable from {nodes_file} and {edges_file}")
            print(
                f"📊 Mapped {len(node_mapping)} unique {node_id_column} values to integer node IDs"
            )
            print(f"🔗 Edge columns: {source_id_column} -> {target_id_column}")
            print(
                f"📈 Nodes: {len(nodes_dict[list(nodes_dict.keys())[0]])}, Edges: {len(edges_dict[list(edges_dict.keys())[0]])}"
            )

            return graph_table
        except Exception as e:
            print(f"Warning: Could not create GraphTable: {e}")
            print(f"Falling back to NodesTable")
            return nodes_table

    elif nodes_filepath is not None:
        # Only nodes file specified - load as NodesTable with column mapping
        if not os.path.exists(nodes_filepath):
            raise FileNotFoundError(f"Nodes CSV file not found: {nodes_filepath}")

        # Load as BaseTable first, then convert with column mapping
        nodes_base = BaseTable.from_csv(nodes_filepath)
        return from_pandas(
            nodes_base.to_pandas(), table_type="nodes", node_id_column=node_id_column
        )

    else:
        raise ValueError(
            "Must provide either 'filepath' for single file or 'nodes_filepath'/'edges_filepath' for graph creation"
        )


def from_pandas(
    df,
    table_type: str = "base",
    *,
    node_id_column: str = "node_id",
    source_id_column: str = "source",
    target_id_column: str = "target",
) -> Union[BaseTable, NodesTable, EdgesTable]:
    """
    Create Groggy table from pandas DataFrame with automatic node ID mapping.

    Args:
        df: pandas DataFrame
        table_type: Type of table to create ("base", "nodes", "edges")
        node_id_column: Column name for node IDs (automatically mapped to integers)
        source_id_column: Column name for source node IDs (automatically mapped to integers)
        target_id_column: Column name for target node IDs (automatically mapped to integers)

    Returns:
        BaseTable, NodesTable, or EdgesTable

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"id": ["alice", "bob"], "name": ["A", "B"]})
        >>> table = gr.from_pandas(df, table_type="nodes", node_id_column="id")
        # String IDs "alice", "bob" automatically mapped to integers 0, 1

        >>> edges_df = pd.DataFrame({"from": ["alice"], "to": ["bob"], "weight": [1.0]})
        >>> edges_table = gr.from_pandas(edges_df, table_type="edges",
        ...                              source_id_column="from", target_id_column="to")
        # String node references automatically mapped to integers
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for from_pandas(). Install with: pip install pandas"
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Validate column requirements based on table type
    if table_type == "nodes" and node_id_column not in df.columns:
        raise ValueError(
            f"Column '{node_id_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if table_type == "edges":
        missing_columns = []
        if source_id_column not in df.columns:
            missing_columns.append(source_id_column)
        if target_id_column not in df.columns:
            missing_columns.append(target_id_column)

        if missing_columns:
            raise ValueError(
                f"Missing required edge columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

    # Convert DataFrame to dict format - preserve all original columns
    data_dict = {col: df[col].tolist() for col in df.columns}

    # Use the new unified mapping system (preserves original columns)
    return _create_table_with_mapping(
        data_dict, table_type, node_id_column, source_id_column, target_id_column
    )


def from_numpy(
    arr, array_type: str = "auto"
) -> Union["BaseArray", "NumArray", "GraphMatrix"]:
    """
    Create Groggy array or matrix from numpy array.

    Args:
        arr: numpy array
        array_type: Type to create ("auto", "array", "num_array", "matrix")

    Returns:
        BaseArray, NumArray, or GraphMatrix

    Example:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3, 4])
        >>> groggy_array = gr.from_numpy(arr, array_type="num_array")
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is required for from_numpy(). Install with: pip install numpy"
        )

    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array")

    # Convert to Python list/nested list
    if arr.ndim == 1:
        data = arr.tolist()

        if array_type == "auto":
            # Auto-detect based on dtype
            if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(
                arr.dtype, np.floating
            ):
                return num_array(data)
            else:
                return array(data)
        elif array_type == "array":
            return array(data)
        elif array_type == "num_array":
            return num_array(data)
        else:
            raise ValueError("1D arrays cannot be converted to matrix")

    elif arr.ndim == 2:
        data = arr.tolist()

        if array_type == "auto" or array_type == "matrix":
            return matrix(data)
        else:
            raise ValueError("2D arrays can only be converted to matrix")

    else:
        raise ValueError(f"Arrays with {arr.ndim} dimensions not supported")


def from_json(
    filepath: str,
    table_type: str = "base",
    *,
    node_id_column: str = "node_id",
    source_id_column: str = "source",
    target_id_column: str = "target",
) -> Union[BaseTable, NodesTable, EdgesTable]:
    """
    Load table from JSON file.

    Args:
        filepath: Path to JSON file
        table_type: Type of table to create ("base", "nodes", "edges")
        node_id_column: Column name for node IDs (for validation when table_type="nodes")
        source_id_column: Column name for source node IDs (for validation when table_type="edges")
        target_id_column: Column name for target node IDs (for validation when table_type="edges")

    Returns:
        BaseTable, NodesTable, or EdgesTable

    Example:
        >>> table = gr.from_json("nodes.json", table_type="nodes", node_id_column="id")
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    # Load the table first to validate columns
    base_table = BaseTable.from_json(filepath)

    # Validate column requirements based on table type
    if table_type == "nodes" and not base_table.has_column(node_id_column):
        raise ValueError(
            f"Column '{node_id_column}' not found in JSON file. "
            f"Available columns: {base_table.column_names}"
        )

    if table_type == "edges":
        missing_columns = []
        if not base_table.has_column(source_id_column):
            missing_columns.append(source_id_column)
        if not base_table.has_column(target_id_column):
            missing_columns.append(target_id_column)

        if missing_columns:
            raise ValueError(
                f"Missing required edge columns: {missing_columns}. "
                f"Available columns: {base_table.column_names}"
            )

    # Return appropriate table type
    if table_type == "base":
        return base_table
    elif table_type == "nodes":
        return NodesTable.from_json(filepath)
    elif table_type == "edges":
        return EdgesTable.from_json(filepath)
    else:
        raise ValueError(f"Unknown table_type: {table_type}")


def from_parquet(
    filepath: str,
    table_type: str = "base",
    *,
    node_id_column: str = "node_id",
    source_id_column: str = "source",
    target_id_column: str = "target",
) -> Union[BaseTable, NodesTable, EdgesTable]:
    """
    Load table from Parquet file.

    Args:
        filepath: Path to Parquet file
        table_type: Type of table to create ("base", "nodes", "edges")
        node_id_column: Column name for node IDs (for validation when table_type="nodes")
        source_id_column: Column name for source node IDs (for validation when table_type="edges")
        target_id_column: Column name for target node IDs (for validation when table_type="edges")

    Returns:
        BaseTable, NodesTable, or EdgesTable

    Example:
        >>> table = gr.from_parquet("edges.parquet", table_type="edges",
        ...                         source_id_column="from", target_id_column="to")
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parquet file not found: {filepath}")

    # Load the table first to validate columns
    base_table = BaseTable.from_parquet(filepath)

    # Validate column requirements based on table type
    if table_type == "nodes" and not base_table.has_column(node_id_column):
        raise ValueError(
            f"Column '{node_id_column}' not found in Parquet file. "
            f"Available columns: {base_table.column_names}"
        )

    if table_type == "edges":
        missing_columns = []
        if not base_table.has_column(source_id_column):
            missing_columns.append(source_id_column)
        if not base_table.has_column(target_id_column):
            missing_columns.append(target_id_column)

        if missing_columns:
            raise ValueError(
                f"Missing required edge columns: {missing_columns}. "
                f"Available columns: {base_table.column_names}"
            )

    # Return appropriate table type
    if table_type == "base":
        return base_table
    elif table_type == "nodes":
        return NodesTable.from_parquet(filepath)
    elif table_type == "edges":
        return EdgesTable.from_parquet(filepath)
    else:
        raise ValueError(f"Unknown table_type: {table_type}")


# Additional format support
def from_dict(
    data: Dict[str, List],
    table_type: str = "base",
    *,
    node_id_column: str = "node_id",
    source_id_column: str = "source",
    target_id_column: str = "target",
) -> Union[BaseTable, NodesTable, EdgesTable]:
    """
    Create table from dictionary of columns with automatic node ID mapping.

    Args:
        data: Dictionary with column names as keys and lists as values
        table_type: Type of table to create ("base", "nodes", "edges")
        node_id_column: Column name for node IDs (automatically mapped to integers)
        source_id_column: Column name for source node IDs (automatically mapped to integers)
        target_id_column: Column name for target node IDs (automatically mapped to integers)

    Returns:
        BaseTable, NodesTable, or EdgesTable

    Example:
        >>> nodes_data = {"id": ["alice", "bob"], "name": ["A", "B"]}
        >>> table = gr.from_dict(nodes_data, table_type="nodes", node_id_column="id")
        # String IDs "alice", "bob" automatically mapped to integers 0, 1
    """
    # Validate column requirements based on table type
    if table_type == "nodes" and node_id_column not in data:
        raise ValueError(
            f"Column '{node_id_column}' not found in data. "
            f"Available columns: {list(data.keys())}"
        )

    if table_type == "edges":
        missing_columns = []
        if source_id_column not in data:
            missing_columns.append(source_id_column)
        if target_id_column not in data:
            missing_columns.append(target_id_column)

        if missing_columns:
            raise ValueError(
                f"Missing required edge columns: {missing_columns}. "
                f"Available columns: {list(data.keys())}"
            )

    # Use the new unified mapping system
    return _create_table_with_mapping(
        data, table_type, node_id_column, source_id_column, target_id_column
    )
