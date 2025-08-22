# Python API Reference

This document provides comprehensive documentation for Groggy's Python API, covering all modules, classes, and methods available to users.

## Table of Contents

1. [Overview](#overview)
2. [Core Graph API](#core-graph-api)
3. [Storage Views](#storage-views)
4. [Query Interface](#query-interface)
5. [Analytics Module](#analytics-module)
6. [Display System](#display-system)
7. [Integration Modules](#integration-modules)
8. [Utility Functions](#utility-functions)

## Overview

Groggy's Python API is designed to be intuitive and familiar to users of pandas, NumPy, and NetworkX while providing high-performance graph processing capabilities.

### Import and Basic Usage

```python
import groggy as gr

# Create a graph
g = gr.Graph()

# Create storage views
array = gr.array([1, 2, 3, 4])
matrix = gr.matrix([[1, 2], [3, 4]])
table = gr.table({"col1": [1, 2], "col2": [3, 4]})
```

### API Design Principles

- **Fluent Interface**: Methods can be chained for complex operations
- **Pandas Compatibility**: Familiar methods like `.head()`, `.tail()`, `.describe()`
- **Type Safety**: Clear error messages for type mismatches
- **Performance**: Batch operations and lazy evaluation by default

## Core Graph API

### Graph Class

The `Graph` class is the primary interface for graph creation and manipulation.

```python
class Graph:
    """High-performance graph with unified storage views."""
    
    def __init__(self, directed: bool = True):
        """Create a new graph.
        
        Args:
            directed: Whether the graph is directed (default: True)
        """
```

#### Node Operations

```python
def add_node(self, node_id: Union[str, int], **attributes) -> None:
    """Add a single node with attributes.
    
    Args:
        node_id: Unique identifier for the node
        **attributes: Node attributes as keyword arguments
        
    Examples:
        >>> g.add_node("alice", age=30, role="engineer")
        >>> g.add_node(1, name="Node 1", active=True)
    """

def add_nodes(self, nodes: List[Dict]) -> None:
    """Add multiple nodes efficiently.
    
    Args:
        nodes: List of node dictionaries with 'id' key and attributes
        
    Examples:
        >>> nodes = [
        ...     {'id': 'alice', 'age': 30, 'role': 'engineer'},
        ...     {'id': 'bob', 'age': 25, 'role': 'designer'}
        ... ]
        >>> g.add_nodes(nodes)
    """

def remove_node(self, node_id: Union[str, int]) -> None:
    """Remove a node and all connected edges.
    
    Args:
        node_id: ID of the node to remove
        
    Raises:
        KeyError: If node doesn't exist
    """

def has_node(self, node_id: Union[str, int]) -> bool:
    """Check if a node exists.
    
    Args:
        node_id: Node ID to check
        
    Returns:
        bool: True if node exists, False otherwise
    """

def get_node(self, node_id: Union[str, int]) -> Dict[str, Any]:
    """Get node attributes.
    
    Args:
        node_id: Node ID
        
    Returns:
        Dict of node attributes
        
    Raises:
        KeyError: If node doesn't exist
    """

def update_node(self, node_id: Union[str, int], attributes: Dict[str, Any]) -> None:
    """Update node attributes.
    
    Args:
        node_id: Node ID
        attributes: Dictionary of attributes to update
    """
```

#### Edge Operations

```python
def add_edge(self, source: Union[str, int], target: Union[str, int], **attributes) -> None:
    """Add a single edge with attributes.
    
    Args:
        source: Source node ID
        target: Target node ID
        **attributes: Edge attributes as keyword arguments
        
    Examples:
        >>> g.add_edge("alice", "bob", weight=0.8, relationship="friend")
    """

def add_edges(self, edges: List[Dict]) -> None:
    """Add multiple edges efficiently.
    
    Args:
        edges: List of edge dictionaries with 'source', 'target', and attributes
        
    Examples:
        >>> edges = [
        ...     {'source': 'alice', 'target': 'bob', 'weight': 0.8},
        ...     {'source': 'bob', 'target': 'charlie', 'weight': 0.6}
        ... ]
        >>> g.add_edges(edges)
    """

def has_edge(self, source: Union[str, int], target: Union[str, int]) -> bool:
    """Check if an edge exists.
    
    Args:
        source: Source node ID
        target: Target node ID
        
    Returns:
        bool: True if edge exists, False otherwise
    """

def get_edge(self, source: Union[str, int], target: Union[str, int]) -> Dict[str, Any]:
    """Get edge attributes.
    
    Args:
        source: Source node ID
        target: Target node ID
        
    Returns:
        Dict of edge attributes
        
    Raises:
        KeyError: If edge doesn't exist
    """
```

#### Graph Properties

```python
@property
def nodes(self) -> NodeView:
    """Access to graph nodes."""

@property  
def edges(self) -> EdgeView:
    """Access to graph edges."""

@property
def directed(self) -> bool:
    """Whether the graph is directed."""

def node_count(self) -> int:
    """Number of nodes in the graph."""

def edge_count(self) -> int:
    """Number of edges in the graph."""

def degree(self, node_id: Optional[Union[str, int]] = None) -> Union[int, Dict[Union[str, int], int]]:
    """Get degree(s) of node(s).
    
    Args:
        node_id: Specific node ID, or None for all nodes
        
    Returns:
        Single degree or dictionary of node_id -> degree
    """
```

#### Storage View Access

```python
def adjacency(self, **kwargs) -> GraphMatrix:
    """Get adjacency matrix representation.
    
    Returns:
        GraphMatrix: Adjacency matrix of the graph
        
    Examples:
        >>> adj = g.adjacency()
        >>> print(adj.shape)
        (100, 100)
        >>> print(adj.is_sparse)
        True
    """

def table(self, entity_type: str = "nodes", attributes: Optional[List[str]] = None) -> GraphTable:
    """Get tabular representation of graph data.
    
    Args:
        entity_type: "nodes" or "edges"
        attributes: Specific attributes to include, or None for all
        
    Returns:
        GraphTable: Tabular view of graph entities
        
    Examples:
        >>> nodes_table = g.table("nodes", ["age", "role"])
        >>> edges_table = g.table("edges", ["weight"])
    """
```

### Node and Edge Views

#### NodeView

```python
class NodeView:
    """View interface for graph nodes."""
    
    def __len__(self) -> int:
        """Number of nodes."""
    
    def __iter__(self) -> Iterator[Union[str, int]]:
        """Iterate over node IDs."""
    
    def __contains__(self, node_id: Union[str, int]) -> bool:
        """Check if node exists."""
    
    def __getitem__(self, node_id: Union[str, int]) -> Dict[str, Any]:
        """Get node attributes."""
    
    def table(self, attributes: Optional[List[str]] = None) -> GraphTable:
        """Convert nodes to table format.
        
        Args:
            attributes: Specific attributes to include
            
        Returns:
            GraphTable with node data
        """
    
    def filter(self, condition: Union[str, Callable]) -> List[Union[str, int]]:
        """Filter nodes by condition.
        
        Args:
            condition: String expression or callable predicate
            
        Returns:
            List of node IDs matching condition
            
        Examples:
            >>> active_nodes = g.nodes.filter("active == True")
            >>> young_nodes = g.nodes.filter(lambda n: n.get('age', 0) < 30)
        """
```

#### EdgeView

```python
class EdgeView:
    """View interface for graph edges."""
    
    def __len__(self) -> int:
        """Number of edges."""
    
    def __iter__(self) -> Iterator[Tuple[Union[str, int], Union[str, int]]]:
        """Iterate over (source, target) pairs."""
    
    def table(self, attributes: Optional[List[str]] = None) -> GraphTable:
        """Convert edges to table format."""
    
    def filter(self, condition: Union[str, Callable]) -> List[Tuple[Union[str, int], Union[str, int]]]:
        """Filter edges by condition."""
```

## Storage Views

### GraphArray

Single-column typed data with statistical operations.

```python
class GraphArray:
    """High-performance array with statistical operations."""
    
    def __init__(self, values: List[Any], name: Optional[str] = None):
        """Create array from values."""
    
    # Properties
    @property
    def values(self) -> List[Any]:
        """Get all values as Python list."""
    
    @property
    def dtype(self) -> str:
        """Data type of the array."""
    
    @property
    def is_sparse(self) -> bool:
        """Whether array is stored sparsely."""
    
    def __len__(self) -> int:
        """Length of the array."""
    
    def __getitem__(self, index: Union[int, slice, List[int], List[bool]]) -> Union[Any, GraphArray]:
        """Advanced indexing support.
        
        Examples:
            >>> arr[5]           # Single element
            >>> arr[1:10:2]      # Slice with step
            >>> arr[[1, 3, 5]]   # Fancy indexing
            >>> arr[mask]        # Boolean indexing
        """
    
    # Statistical operations
    def mean(self) -> Optional[float]:
        """Arithmetic mean."""
    
    def median(self) -> Optional[float]:
        """Median value."""
    
    def std(self) -> Optional[float]:
        """Standard deviation."""
    
    def min(self) -> Optional[Any]:
        """Minimum value."""
    
    def max(self) -> Optional[Any]:
        """Maximum value."""
    
    def sum(self) -> Optional[float]:
        """Sum of values."""
    
    def count(self) -> int:
        """Count of non-null values."""
    
    def unique(self) -> GraphArray:
        """Unique values."""
    
    def value_counts(self) -> Dict[Any, int]:
        """Count of each unique value."""
    
    def describe(self) -> Dict[str, float]:
        """Summary statistics."""
    
    # Operations
    def filter(self, predicate: Callable[[Any], bool]) -> GraphArray:
        """Filter values by predicate."""
    
    def map(self, transform: Callable[[Any], Any]) -> GraphArray:
        """Transform values."""
    
    def sort(self, ascending: bool = True) -> GraphArray:
        """Sort values."""
    
    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
    
    def to_pandas(self) -> pd.Series:
        """Convert to pandas Series."""
```

### GraphMatrix

Collection of homogeneous arrays with linear algebra operations.

```python
class GraphMatrix:
    """Matrix operations on homogeneous data."""
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix dimensions (rows, cols)."""
    
    @property
    def dtype(self) -> str:
        """Data type of matrix elements."""
    
    @property
    def is_square(self) -> bool:
        """Whether matrix is square."""
    
    @property
    def is_sparse(self) -> bool:
        """Whether matrix is stored sparsely."""
    
    def __getitem__(self, key: Union[int, Tuple[int, int], slice]) -> Union[Any, GraphArray, GraphMatrix]:
        """Matrix indexing.
        
        Examples:
            >>> matrix[0]        # First row
            >>> matrix[0, 5]     # Single element
            >>> matrix[:, 2]     # Third column
            >>> matrix[1:5, :]   # Row slice
        """
    
    # Statistical operations
    def sum_axis(self, axis: int) -> GraphArray:
        """Sum along axis (0=columns, 1=rows)."""
    
    def mean_axis(self, axis: int) -> GraphArray:
        """Mean along axis."""
    
    def std_axis(self, axis: int) -> GraphArray:
        """Standard deviation along axis."""
    
    # Matrix operations
    def transpose(self) -> GraphMatrix:
        """Matrix transpose."""
    
    def power(self, n: int) -> GraphMatrix:
        """Matrix power (A^n)."""
    
    def to_dense(self) -> GraphMatrix:
        """Convert to dense representation."""
    
    def to_sparse(self) -> GraphMatrix:
        """Convert to sparse representation."""
    
    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
```

### GraphTable

Collection of heterogeneous arrays with relational operations.

```python
class GraphTable:
    """Pandas-like table operations with graph integration."""
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Table dimensions (rows, cols)."""
    
    @property
    def columns(self) -> List[str]:
        """Column names."""
    
    @property
    def dtypes(self) -> Dict[str, str]:
        """Data types of each column."""
    
    def __len__(self) -> int:
        """Number of rows."""
    
    def __getitem__(self, column: Union[str, List[str]]) -> Union[GraphArray, GraphTable]:
        """Column access.
        
        Examples:
            >>> table['age']           # Single column as GraphArray
            >>> table[['age', 'name']] # Multiple columns as GraphTable
        """
    
    # Data access
    def head(self, n: int = 5) -> GraphTable:
        """First n rows."""
    
    def tail(self, n: int = 5) -> GraphTable:
        """Last n rows."""
    
    def sample(self, n: int) -> GraphTable:
        """Random sample of n rows."""
    
    def iloc(self, index: Union[int, slice, List[int]]) -> Union[Dict[str, Any], GraphTable]:
        """Position-based indexing."""
    
    # Statistical operations
    def describe(self) -> GraphTable:
        """Summary statistics for each column."""
    
    def group_by(self, column: str) -> GroupBy:
        """Group table by column values."""
    
    def agg(self, operations: Dict[str, Union[str, List[str]]]) -> Dict[str, Any]:
        """Aggregate operations.
        
        Args:
            operations: Dict of column -> operation(s)
            
        Examples:
            >>> table.agg({
            ...     'age': ['mean', 'std'],
            ...     'salary': 'sum',
            ...     'active': 'count'
            ... })
        """
    
    # Data manipulation
    def sort_by(self, column: str, ascending: bool = True) -> GraphTable:
        """Sort by column values."""
    
    def filter_rows(self, predicate: Callable[[Dict[str, Any]], bool]) -> GraphTable:
        """Filter rows by predicate."""
    
    def drop_duplicates(self) -> GraphTable:
        """Remove duplicate rows."""
    
    def fillna(self, value: Any) -> GraphTable:
        """Fill null values."""
    
    def dropna(self) -> GraphTable:
        """Drop rows with null values."""
    
    # Multi-table operations
    def join(self, other: GraphTable, on: str, how: str = "inner") -> GraphTable:
        """Join with another table.
        
        Args:
            other: Table to join with
            on: Column name to join on
            how: Join type ('inner', 'left', 'right', 'outer')
        """
    
    def union(self, other: GraphTable) -> GraphTable:
        """Union with another table (concatenate rows)."""
    
    def intersect(self, other: GraphTable) -> GraphTable:
        """Intersection with another table."""
    
    # Graph-aware operations
    @staticmethod
    def neighborhood_table(graph: Graph, node_id: Union[str, int], attributes: List[str]) -> GraphTable:
        """Extract neighborhood as table.
        
        Args:
            graph: Source graph
            node_id: Central node
            attributes: Node attributes to include
            
        Returns:
            Table of neighboring nodes with attributes
        """
    
    @staticmethod
    def k_hop_neighborhood_table(graph: Graph, node_id: Union[str, int], k: int, attributes: List[str]) -> GraphTable:
        """Extract k-hop neighborhood as table."""
    
    def filter_by_degree(self, graph: Graph, node_column: str, min_degree: int = 0, max_degree: Optional[int] = None) -> GraphTable:
        """Filter rows by node degree in graph."""
    
    def filter_by_connectivity(self, graph: Graph, node_column: str, target_nodes: List[Union[str, int]], mode: str = "any") -> GraphTable:
        """Filter rows by connectivity to target nodes."""
    
    # Export/import
    def to_dict(self) -> Dict[str, List[Any]]:
        """Convert to dictionary of lists."""
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
    
    def to_csv(self, path: str) -> None:
        """Export to CSV file."""
    
    def to_json(self) -> str:
        """Export to JSON string."""
```

### GroupBy Operations

```python
class GroupBy:
    """Group-by operations for GraphTable."""
    
    def count(self, column: Optional[str] = None) -> Union[int, GraphTable]:
        """Count of values in each group."""
    
    def sum(self, column: Optional[str] = None) -> Union[float, GraphTable]:
        """Sum of values in each group."""
    
    def mean(self, column: Optional[str] = None) -> Union[float, GraphTable]:
        """Mean of values in each group."""
    
    def min(self, column: Optional[str] = None) -> Union[Any, GraphTable]:
        """Minimum value in each group."""
    
    def max(self, column: Optional[str] = None) -> Union[Any, GraphTable]:
        """Maximum value in each group."""
    
    def std(self, column: Optional[str] = None) -> Union[float, GraphTable]:
        """Standard deviation in each group."""
    
    def agg(self, operations: Dict[str, Union[str, List[str]]]) -> GraphTable:
        """Multiple aggregation operations."""
```

## Query Interface

### Enhanced Query System

```python
# String-based queries
active_users = g.nodes.filter("active == True and age > 25")
high_value_edges = g.edges.filter("weight > 0.8")

# Complex table queries
result = g.table('nodes').query("""
    SELECT name, age, department 
    WHERE age > 30 AND department IN ('engineering', 'research')
    ORDER BY age DESC
""")

# Graph-aware queries  
influential_users = g.table('nodes').query("""
    SELECT name, centrality_score
    WHERE degree > 10 AND centrality_score > 0.5
""")
```

## Analytics Module

### Graph Algorithms

```python
# Centrality measures
betweenness = g.centrality.betweenness()
pagerank = g.centrality.pagerank(alpha=0.85)
closeness = g.centrality.closeness()

# Community detection
communities = g.communities.louvain()
modularity = g.communities.modularity(communities)

# Path algorithms
shortest_path = g.path.shortest("alice", "bob")
all_paths = g.path.all_simple("alice", "bob", max_length=5)

# Connectivity
components = g.connectivity.connected_components()
bridges = g.connectivity.bridges()
articulation_points = g.connectivity.articulation_points()
```

## Display System

### Rich Display Support

Groggy provides beautiful HTML displays for Jupyter notebooks and formatted text output for terminals.

```python
class DisplayMixin:
    """Mixin providing rich display capabilities."""
    
    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
    
    def __repr__(self) -> str:
        """String representation for terminals."""
    
    def preview(self, limit: int = 10) -> str:
        """Preview with limited rows."""
    
    def summary(self) -> str:
        """Summary information."""
```

### Customizable Display

```python
# Configure display options
gr.config.display.max_rows = 100
gr.config.display.max_columns = 20
gr.config.display.precision = 3
gr.config.display.html_table_style = "bootstrap"

# Custom formatters
gr.config.display.formatters['float'] = lambda x: f"{x:.2f}"
gr.config.display.formatters['large_int'] = lambda x: f"{x:,}"
```

## Integration Modules

### NetworkX Compatibility

```python
import groggy as gr
import networkx as nx

# Convert between formats
nx_graph = nx.Graph()
nx_graph.add_edges_from([('a', 'b'), ('b', 'c')])

# NetworkX to Groggy
g = gr.from_networkx(nx_graph)

# Groggy to NetworkX  
nx_graph = g.to_networkx()

# Use NetworkX algorithms with Groggy data
communities = nx.community.greedy_modularity_communities(g.to_networkx())
```

### Pandas Integration

```python
import pandas as pd

# DataFrame to Groggy table
df = pd.DataFrame({'name': ['alice', 'bob'], 'age': [30, 25]})
table = gr.table.from_pandas(df)

# Groggy table to DataFrame
df = table.to_pandas()

# Seamless operations
result = table.head().to_pandas().merge(other_df, on='name')
```

## Utility Functions

### Constructor Functions

```python
def array(values: List[Any], name: Optional[str] = None) -> GraphArray:
    """Create GraphArray from values."""

def matrix(data: Union[List[List[Any]], np.ndarray]) -> GraphMatrix:
    """Create GraphMatrix from 2D data."""

def table(data: Union[Dict[str, List[Any]], pd.DataFrame, List[Dict[str, Any]]]) -> GraphTable:
    """Create GraphTable from various data sources."""

def graph() -> Graph:
    """Create empty graph."""

def from_networkx(nx_graph) -> Graph:
    """Create graph from NetworkX graph."""

def from_pandas(nodes_df: pd.DataFrame, edges_df: Optional[pd.DataFrame] = None) -> Graph:
    """Create graph from pandas DataFrames."""
```

### Configuration

```python
import groggy.config as config

# Performance settings
config.performance.parallel_threshold = 1000
config.performance.cache_size = "1GB"
config.performance.use_sparse_by_default = True

# Display settings
config.display.max_rows = 50
config.display.precision = 3
config.display.show_dtypes = True

# Memory settings  
config.memory.pool_size = "512MB"
config.memory.string_cache_size = 10000
config.memory.cleanup_threshold = 0.8
```

### Error Handling

```python
from groggy.errors import (
    GraphError,
    NodeNotFoundError,
    EdgeNotFoundError, 
    TypeMismatchError,
    InvalidOperationError
)

try:
    node = g.get_node("nonexistent")
except NodeNotFoundError as e:
    print(f"Node not found: {e}")
    
try:
    result = table.join(other_table, on="invalid_column")
except InvalidOperationError as e:
    print(f"Join failed: {e}")
```

This Python API provides a comprehensive, intuitive interface for graph processing while maintaining high performance through the underlying Rust implementation.