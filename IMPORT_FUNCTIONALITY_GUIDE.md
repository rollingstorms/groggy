# Groggy Data Import Functionality Guide

This guide covers the comprehensive data import capabilities added to Groggy, making it easy to work with various data formats and convert between table types.

## Overview

Groggy now provides a unified interface for importing data from multiple sources with **flexible column mapping**:

- **CSV files** → `gr.from_csv()` with custom column specifications
- **Pandas DataFrames** → `gr.from_pandas()` with column mapping
- **NumPy arrays** → `gr.from_numpy()` for arrays and matrices
- **JSON files** → `gr.from_json()` with column validation
- **Parquet files** → `gr.from_parquet()` with column validation
- **Python dictionaries** → `gr.from_dict()` with column mapping

Plus table conversion methods:
- **BaseTable** → `table.to_nodes_table()`, `table.to_edges_table()`

### Key Feature: Custom Column Names

All import functions now support specifying custom column names, allowing you to work with data that doesn't follow Groggy's default naming conventions:

```python
# Your data uses 'user_id' instead of 'node_id'? No problem!
df = pd.DataFrame({"user_id": [1, 2], "name": ["A", "B"]})
nodes = gr.from_pandas(df, table_type="nodes", node_id_column="user_id")

# Edge data with custom column names
edges_data = {"from": [1], "to": [2], "weight": [1.0]}
edges = gr.from_dict(edges_data, table_type="edges",
                    source_id_column="from", target_id_column="to")
```

## Quick Start Examples

### Basic Data Import

```python
import groggy as gr

# Load CSV file as table
table = gr.from_csv("data.csv")

# Load from pandas DataFrame
import pandas as pd
df = pd.DataFrame({"name": ["A", "B"], "value": [1, 2]})
table = gr.from_pandas(df)

# Load from numpy array
import numpy as np
arr = np.array([1, 2, 3, 4])
groggy_array = gr.from_numpy(arr)

# Load from dictionary
data = {"node_id": [1, 2], "name": ["Alice", "Bob"]}
table = gr.from_dict(data)
```

### Graph Construction from CSV Files

```python
# Method 1: Single file with automatic table creation
table = gr.from_csv("data.csv")

# Method 2: Separate node and edge files (planned)
graph_table = gr.from_csv(
    nodes_fp="nodes.csv",
    edges_fp="edges.csv",
    node_id_column="id",
    source_node_id_column="from_node",
    target_node_id_column="to_node"
)
```

### Table Type Conversions

```python
# Load as BaseTable, then convert
base_table = gr.from_csv("nodes.csv")
nodes_table = base_table.to_nodes_table("node_id")

# Load edges data and convert
edges_data = gr.from_dict({
    "edge_id": [1, 2, 3],
    "source": [1, 2, 1],
    "target": [2, 3, 3],
    "weight": [0.5, 0.8, 0.3]
})
edges_table = edges_data.to_edges_table("source", "target")
```

## Detailed API Reference

### `gr.from_csv(filepath, **options)`

Import data from CSV files with flexible graph construction options.

**Parameters:**
- `filepath`: Path to CSV file
- `nodes_fp`: Optional path to nodes CSV for graph creation
- `edges_fp`: Optional path to edges CSV for graph creation
- `node_id_column`: Column name for node IDs (default: "node_id")
- `source_node_id_column`: Column name for source nodes (default: "source")
- `target_node_id_column`: Column name for target nodes (default: "target")

**Returns:** `BaseTable` or `GraphTable`

**Examples:**
```python
# Single table
table = gr.from_csv("data.csv")

# Graph from separate files
graph = gr.from_csv(
    nodes_fp="nodes.csv",
    edges_fp="edges.csv",
    node_id_column="id"
)
```

### `gr.from_pandas(df, table_type="base", *, node_id_column="node_id", source_id_column="source", target_id_column="target")`

Convert pandas DataFrame to Groggy table with flexible column mapping.

**Parameters:**
- `df`: pandas DataFrame
- `table_type`: "base", "nodes", or "edges"
- `node_id_column`: Column name for node IDs (for nodes tables)
- `source_id_column`: Column name for source node IDs (for edges tables)
- `target_id_column`: Column name for target node IDs (for edges tables)

**Returns:** `BaseTable`, `NodesTable`, or `EdgesTable`

**Examples:**
```python
import pandas as pd

# Nodes with custom column names
df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "username": ["alice", "bob", "carol"],
    "department": ["eng", "sales", "marketing"]
})
nodes_table = gr.from_pandas(df, table_type="nodes", node_id_column="user_id")

# Edges with custom column names
edges_df = pd.DataFrame({
    "from_user": [1, 2],
    "to_user": [2, 3],
    "weight": [0.8, 0.6]
})
edges_table = gr.from_pandas(edges_df, table_type="edges",
                            source_id_column="from_user",
                            target_id_column="to_user")
```

### `gr.from_numpy(arr, array_type="auto")`

Convert numpy arrays to Groggy arrays or matrices.

**Parameters:**
- `arr`: numpy array (1D or 2D)
- `array_type`: "auto", "array", "num_array", or "matrix"

**Returns:** `BaseArray`, `NumArray`, or `GraphMatrix`

**Example:**
```python
import numpy as np

# 1D array → NumArray
arr_1d = np.array([1, 2, 3, 4])
num_array = gr.from_numpy(arr_1d, array_type="num_array")

# 2D array → GraphMatrix
arr_2d = np.array([[1, 2], [3, 4]])
matrix = gr.from_numpy(arr_2d, array_type="matrix")
```

### `gr.from_dict(data, table_type="base", *, node_id_column="node_id", source_id_column="source", target_id_column="target")`

Create tables from Python dictionaries with flexible column mapping.

**Parameters:**
- `data`: Dictionary with column names as keys, lists as values
- `table_type`: "base", "nodes", or "edges"
- `node_id_column`: Column name for node IDs (for nodes tables)
- `source_id_column`: Column name for source node IDs (for edges tables)
- `target_id_column`: Column name for target node IDs (for edges tables)

**Returns:** `BaseTable`, `NodesTable`, or `EdgesTable`

**Examples:**
```python
# Nodes with custom column names
nodes_data = {
    "employee_id": [101, 102, 103],
    "name": ["Alice", "Bob", "Carol"],
    "department": ["Eng", "Sales", "Marketing"]
}
nodes_table = gr.from_dict(nodes_data, table_type="nodes", node_id_column="employee_id")

# Edges with custom column names
edges_data = {
    "from_employee": [101, 102],
    "to_employee": [102, 103],
    "reports_to": [True, False]
}
edges_table = gr.from_dict(edges_data, table_type="edges",
                          source_id_column="from_employee",
                          target_id_column="to_employee")
```

### Table Conversion Methods

#### `table.to_nodes_table(node_id_column="node_id")`

Convert any BaseTable to NodesTable.

**Parameters:**
- `node_id_column`: Name of column containing node IDs

**Returns:** `NodesTable`

#### `table.to_edges_table(source_column="source", target_column="target", edge_id_column=None)`

Convert any BaseTable to EdgesTable.

**Parameters:**
- `source_column`: Column with source node IDs
- `target_column`: Column with target node IDs
- `edge_id_column`: Optional column with edge IDs

**Returns:** `EdgesTable`

## Common Workflows

### 1. Social Network from CSV

```python
# Load network data
nodes = gr.from_csv("users.csv")  # id, name, age, location
edges = gr.from_csv("friendships.csv")  # user_a, user_b, since

# Convert to specialized tables
nodes_table = nodes.to_nodes_table("id")
edges_table = edges.to_edges_table("user_a", "user_b")

print(f"Network: {nodes_table.nrows} users, {edges_table.nrows} friendships")
```

### 2. Scientific Data from NumPy

```python
import numpy as np

# Adjacency matrix
adj_matrix = np.random.rand(100, 100)
graph_matrix = gr.from_numpy(adj_matrix, array_type="matrix")

# Node features
features = np.random.rand(100, 10)
feature_matrix = gr.from_numpy(features, array_type="matrix")
```

### 3. Analysis Pipeline with Pandas

```python
import pandas as pd

# Load and preprocess with pandas
df = pd.read_csv("raw_data.csv")
df_cleaned = df.dropna().reset_index(drop=True)

# Import to Groggy for graph analysis
table = gr.from_pandas(df_cleaned, table_type="base")
nodes_table = table.to_nodes_table("entity_id")

# Continue with Groggy's graph algorithms...
```

## File Format Support

| Format | Function | Notes |
|--------|----------|-------|
| CSV | `gr.from_csv()` | Full support with graph construction |
| JSON | `gr.from_json()` | Uses existing FFI methods |
| Parquet | `gr.from_parquet()` | Uses existing FFI methods |
| pandas | `gr.from_pandas()` | Requires pandas installation |
| NumPy | `gr.from_numpy()` | Requires numpy installation |
| Dict | `gr.from_dict()` | Native Python support |

## Requirements

- **Core functionality**: No additional dependencies
- **Pandas support**: `pip install pandas`
- **NumPy support**: `pip install numpy`

## Error Handling

The import functions provide clear error messages for common issues:

```python
# Missing required columns
try:
    edges = table.to_edges_table("src", "dst")  # columns don't exist
except ValueError as e:
    print(f"Error: {e}")
    # Error: Missing required columns: ['src', 'dst']. Available columns: ['source', 'target']

# File not found
try:
    table = gr.from_csv("missing.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Error: CSV file not found: missing.csv
```

## Implementation Notes

- **Table conversions** use pandas as an intermediate format when possible
- **Graph construction** from separate node/edge files requires FFI implementation
- **Type detection** is automatic for numpy arrays based on dtype
- **Column validation** ensures required columns exist before conversion

## Future Enhancements

Planned improvements include:

1. **Direct FFI table conversions** (bypass pandas intermediate step)
2. **GraphTable.from_nodes_edges()** for direct graph construction
3. **Additional format support** (HDF5, Arrow, etc.)
4. **Streaming import** for large datasets
5. **Custom column mapping** and data transformations

This import system makes Groggy much more accessible for data science workflows while maintaining its high-performance graph processing capabilities.

## ✨ New in This Version: Enhanced Column Specification

### Universal Column Parameters

All import functions now accept these keyword-only parameters:

- **`node_id_column="node_id"`** - Specify the column containing node IDs
- **`source_id_column="source"`** - Specify the column containing source node IDs
- **`target_id_column="target"`** - Specify the column containing target node IDs

### Automatic Column Mapping

The import system automatically handles column name differences:

```python
# Data with non-standard column names
company_data = {
    "employee_id": [101, 102, 103],  # ← Maps to 'node_id' internally
    "employee_name": ["Alice", "Bob", "Carol"],
    "department": ["Engineering", "Sales", "Marketing"]
}

# Specify the mapping
employees = gr.from_dict(company_data,
                        table_type="nodes",
                        node_id_column="employee_id")  # ← Column mapping

# Groggy handles the rest automatically!
```

### Comprehensive Validation

- **Column existence checking** before table creation
- **Clear error messages** with available column suggestions
- **Type-specific validation** (nodes vs edges requirements)
- **Consistent behavior** across all import functions

This enhancement makes Groggy work seamlessly with real-world datasets that use diverse naming conventions, eliminating the need for manual data preprocessing!