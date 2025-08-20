# Groggy Usage Examples

This document contains comprehensive usage examples extracted from the development documentation, showcasing the current and planned API capabilities of the Groggy graph library.

## 🚀 GraphArray - Statistical Arrays with Native Performance

### Basic Statistical Operations
```python
# NEW: GraphTable columns return GraphArray objects with native statistics
table = g.table()
ages = table['age']              # Returns GraphArray (not plain list)
print(ages.mean(), ages.std())   # Native Rust statistical operations

# Create GraphArray from values (renamed from PyArray)
ages = groggy.GraphArray([25, 30, 35, 40, 45])

# Statistical methods (computed in Rust)
print(ages.mean())           # 35.0
print(ages.std())            # 7.91
print(ages.min())            # 25
print(ages.max())            # 45  
print(ages.median())         # 35.0
print(ages.quantile(0.95))   # 44.0

# List compatibility
print(len(ages))             # 5
print(ages[0])               # 25
print(ages[-1])              # 45 (negative indexing works)
for age in ages: print(age)  # Iteration works

# Statistical summary
summary = ages.describe()
print(summary.count, summary.mean, summary.std)

# Convert back to plain list
plain_list = ages.to_list()
```

### Enhanced GraphArray Features (Planned)
```python
# Enhanced: GraphArray with statistical capabilities
node_ids = g.node_ids           # Returns GraphArray([0, 1, 2, 3, 4])
print(node_ids.min(), node_ids.max())  # ID range analysis
print(node_ids.mean())          # Average node ID (useful for sparse graphs)
print(len(node_ids.unique()))   # Count unique IDs (validation)

# Subgraph node analysis
subgraph_nodes = subgraph.node_ids  # GraphArray([1, 3, 5])  
print(f"Subgraph has {len(subgraph_nodes)} nodes")
print(f"Node ID range: {subgraph_nodes.min()}-{subgraph_nodes.max()}")

# Scientific conversions (PLANNED)
ids_numpy = node_ids.to_numpy()        # Convert to numpy array
ages_pandas = ages.to_pandas()          # Convert to pandas Series
sparse_matrix = adjacency.to_scipy_sparse()  # For sparse matrices

# Integration examples
import numpy as np
import pandas as pd
age_stats = np.histogram(ages.to_numpy(), bins=10)  # NumPy integration
age_series = ages.to_pandas()                       # Pandas integration
correlation = np.corrcoef(ages.to_numpy(), salaries.to_numpy())  # Multi-array analysis

# GraphArray repr improvements (PLANNED)
# Current: Uninformative representation
ages = GraphArray([25, 30, 35, 40, 45])
print(ages)  # GraphArray(len=5) - Not helpful!

# Should show: GraphArray(len=5, values=[25, 30, 35, 40, 45])
# For large arrays: GraphArray(len=1000, values=[25, 30, 35, ..., 98, 99, 100])

# Pandas-like .values property (PLANNED)
ages = table['age']              # Returns GraphArray
raw_data = ages.values           # Should return plain Python list
```

## 🏗️ Graph Construction and Basic Operations

### Node and Edge Creation
```python
# Clean API with kwargs and flexible inputs
alice = g.add_node(id="alice", age=30, role="engineer")  
bob = g.add_node(id="bob", age=25, role="engineer")
g.add_edge(alice, bob, relationship="collaborates")

# Bulk operations with dicts
node_data = [
    {"id": "alice", "age": 30, "role": "engineer"}, 
    {"id": "bob", "age": 25, "role": "designer"}
]
edge_data = [
    {"source": "alice", "target": "bob", "relationship": "collaborates"}
]

# Two-step with mapping
node_mapping = g.add_nodes(node_data, uid_key="id")  # Returns {"alice": internal_id_0, "bob": internal_id_1}
g.add_edges(edge_data, node_mapping)

# Direct edge addition
edge_data = [(0, 1, {"relationship": "collaborates"})]
g.add_edges(edge_data)

g.add_edge(0, 1, relationship="collaborates")
g.add_edge("alice", "bob", relationship="collaborates", uid_key="id")
```

### Graph Properties and Basic Access
```python
# FIXED: Subgraph properties now work consistently
subgraph = g.connected_components()[0]
print(subgraph.node_ids)         # ✅ Works - shows component nodes
print(subgraph.edge_ids)         # ✅ Works - shows component edges
```

## 🔍 Querying and Filtering

### String-Based Query Parsing
```python
# Multiple filtering approaches
role_filter = gr.NodeFilter.attribute_equals("role", gr.AttrValue("engineer"))  # original syntax
engineers = g.filter_nodes(role_filter)

# String-based query parsing
engineers = g.filter_nodes("role == 'engineer'")
high_earners = g.filter_nodes("salary > 120000")

# Complex expressions (COMPLETED)
# 3+ term AND/OR
g.filter_nodes("age > 25 AND age < 50 AND salary > 70000 AND active == true")
g.filter_nodes("dept == 'Sales' OR dept == 'Marketing' OR dept == 'HR'")

# Parentheses and mixed operators  
g.filter_nodes("(age < 30 OR age > 50) AND active == true")
g.filter_nodes("dept == 'Engineering' AND (age < 30 OR salary > 100000)")

# NOT with complex expressions
g.filter_nodes("NOT (dept == 'Engineering' OR dept == 'Sales')")

# Nested parentheses
g.filter_nodes("(dept == 'Engineering' OR dept == 'Sales') AND (age > 30 AND salary > 80000)")
```

## 📊 GraphTable - DataFrame-like Operations

### Table Creation and Access
```python
# Create table views of graph data
node_table = g.table()  # All nodes with all attributes
edge_table = g.edges.table()  # All edges with source/target + attributes

# Subgraph table views
engineers = g.filter_nodes('dept == "Engineering"')  
eng_table = engineers.table()  # Only engineering nodes
edge_eng_table = engineers.edges.table()  # Engineering subgraph edges

# Export capabilities
node_table.to_pandas()  # Convert to pandas DataFrame
node_table.to_csv('data.csv')  # Direct export
node_table.to_json('data.json')  # JSON export

# Enhanced: GraphArray columns with statistical methods
table = g.table()
ages = table['age']          # Returns GraphArray([25, 30, 35, 40, 45])

# Native statistical operations on table columns
print(ages.mean())           # 35.0 - computed in Rust
print(ages.std())            # 7.91 - native standard deviation
print(ages.quantile(0.95))   # 44.0 - 95th percentile
print(ages.describe())       # Full statistical summary
```

### Multi-Column Selection (Planned)
```python
# Multi-column selection returns GraphMatrix (collection of GraphArray columns)
result = g.nodes[:][['id','index']]  
print(result)  # GraphMatrix(columns=['id', 'index'], rows=100)

# GraphMatrix is a structured wrapper around GraphArray columns
print(result.shape)              # (100, 2) - rows x columns
print(result.columns)            # ['id', 'index'] - column names
print(result.column_count)       # 2
print(result.row_count)          # 100

# Access individual columns as GraphArray
ids = result['id']               # Returns GraphArray([0, 1, 2, ...])
indices = result['index']        # Returns GraphArray([0, 1, 2, ...])

# Statistical operations work on each column
print(result['id'].mean())       # Mean of ID column
print(result['index'].std())     # Std dev of index column

# Data access with .values property  
raw_data = result.values         # Returns [[id_values], [index_values]] - 2D structure
column_data = result['id'].values # Returns [0, 1, 2, ...] - 1D list

# Convert to different formats
matrix_numpy = result.to_numpy()     # 2D NumPy array (100x2)
matrix_pandas = result.to_pandas()   # Pandas DataFrame with proper columns
matrix_list = result.to_list()       # List of lists [[ids], [indices]]
matrix_table = result.table()        # GraphTable with inherited column names

# Consistent API constructors (planned)
matrix = gr.matrix(data)              # Create GraphMatrix via gr.matrix
table = gr.table(data)                # Create GraphTable via gr.table  
array = gr.array(data)                # Create GraphArray via gr.array

# Bidirectional conversions
table_from_matrix = matrix.table()    # GraphMatrix → GraphTable
matrix_from_table = table.matrix()    # GraphTable → GraphMatrix

# Universal Design Principle: Any operation returning multiple GraphArrays returns GraphMatrix
# This provides structure while maintaining access to individual GraphArray statistical capabilities

# Key Distinction: GraphMatrix vs GraphTable
# GraphMatrix: Lightweight, sparse-friendly column collection (like numpy 2D array)
#   - Minimal metadata, efficient for large sparse datasets  
#   - Scientific computing focus: to_numpy(), to_scipy_sparse()
#   - Use case: Multi-column data extraction, matrix operations, linear algebra
# GraphTable: Rich DataFrame-like structure (like pandas DataFrame)
#   - Full metadata, row/column labels, graph entity relationships
#   - Data analysis focus: to_pandas(), to_csv(), to_json()  
#   - Use case: Data analysis, exports, joins, complex table operations

# Usage examples:
ages_salaries = g.nodes[:][['age', 'salary']]  # → GraphMatrix (column extraction)
full_table = g.table()                         # → GraphTable (complete analysis)

# Direct conversions - both support numpy/pandas
matrix_numpy = ages_salaries.to_numpy()        # GraphMatrix → numpy (2D array)
table_numpy = full_table.to_numpy()            # GraphTable → numpy (2D array) 
sparse_matrix = ages_salaries.to_scipy_sparse() # GraphMatrix → scipy (sparse)
analysis_df = full_table.to_pandas()           # GraphTable → pandas (DataFrame)

# No need for intermediate conversions:
# full_table.to_numpy() works directly (no need for .matrix().to_numpy())

# Consistent API exposure with gr.* constructors (planned)
matrix = gr.matrix(data)              # Create GraphMatrix via gr.matrix
table = gr.table(data)                # Create GraphTable via gr.table  
array = gr.array(data)                # Create GraphArray via gr.array

# Bidirectional conversions between data structures
table = g.table()                     # Original GraphTable
matrix = table.to_graphmatrix()       # Convert to GraphMatrix for column operations
back_to_table = matrix.to_graphtable() # Convert back to GraphTable with inherited metadata

# Universal Design Principle: Any operation returning multiple GraphArrays returns GraphMatrix
# This provides structure while maintaining access to individual GraphArray statistical capabilities

# GraphTable Multi-Column Access (PLANNED)
# Current: This fails with TypeError
table = g.table()
subset = table[['id', 'index']]  
# TypeError: Key must be string (column), int (row), or slice

# Should work: Return GraphTable with only selected columns (for table views)
# Expected: subset.to_pandas() shows DataFrame with 'id', 'index' columns only
```

### Advanced Table Operations
```python
# Create table and analyze columns
engineers = g.filter_nodes('dept == "Engineering"')
table = engineers.table()

# Analyze salary distribution
salaries = table['salary']              # Returns GraphArray
print(f"Mean: {salaries.mean()}")       # Native mean
print(f"Median: {salaries.median()}")   # Native median  
print(f"95th percentile: {salaries.quantile(0.95)}")

# Age analysis
ages = table['age']
print(ages.describe())                  # Full summary statistics

# Multi-column analysis
experience = table['years_experience']
print(f"Salary-Experience correlation: {salaries.correlation(experience)}")
```

## 🔢 Adjacency Matrix and Scientific Computing

### Adjacency Matrix Operations (Planned)
```python
# Full graph adjacency matrix
adj_matrix = g.adjacency()                    # Returns gr.array (sparse/dense matrix)

# Subgraph adjacency matrix with index mapping  
subgraph = g.filter_nodes("dept == 'Engineering'")
adj_matrix = subgraph.adjacency(map_index=True)   # Default: True - compact gr.array
index_mapping = subgraph.index_mapping()           # Maps subgraph indices to original node IDs

# Option: Full-size matrix (rare use case)
adj_matrix_full = subgraph.adjacency(map_index=False)  # Full graph size gr.array, sparse for subgraph

# Laplacian matrix support
laplacian = g.laplacian(epsilon=-0.5, k=1)            # Graph Laplacian as gr.array
laplacian_sub = subgraph.laplacian(epsilon=-0.5, k=1) # Subgraph Laplacian as gr.array

# Fast conversion to scientific computing libraries
adj_numpy = adj_matrix.to_numpy()              # Convert to NumPy array
adj_scipy = adj_matrix.to_scipy_sparse()       # Convert to SciPy sparse matrix  
adj_pandas = adj_matrix.to_pandas()            # Convert to Pandas DataFrame

# GraphArray integration - node/edge IDs as statistical arrays
node_ids = g.node_ids                           # Returns GraphArray([0, 1, 2, 3, 4])
edge_ids = subgraph.edge_ids                    # Returns GraphArray([5, 7, 12])

# Statistical analysis on IDs
print(f"Node ID range: {node_ids.min()}-{node_ids.max()}")
print(f"Average node ID: {node_ids.mean()}")   # Useful for sparse graphs
print(f"Subgraph connectivity: {len(edge_ids)} edges for {len(node_ids)} nodes")

# Scientific computing with ID arrays
nodes_numpy = node_ids.to_numpy()              # NumPy array of node IDs
edges_pandas = edge_ids.to_pandas()            # Pandas Series of edge IDs

# Usage examples
import numpy as np
eigenvals = np.linalg.eigvals(adj_matrix.to_numpy())      # Graph spectral analysis
degrees = adj_matrix.to_numpy().sum(axis=1)               # Node degrees from matrix
laplacian_eigenvals = np.linalg.eigvals(laplacian.to_numpy())  # Laplacian spectrum
```

### Implementation Strategy for Matrices (Planned)
```python
# PyGraph method
def adjacency(self, sparse=True, dtype=np.float64):
    """Generate adjacency matrix for full graph"""
    return self._rust_graph.adjacency_matrix(sparse, dtype)

def laplacian(self, epsilon=-0.5, k=1, sparse=True, dtype=np.float64):
    """Generate Laplacian matrix for full graph"""
    return self._rust_graph.laplacian_matrix(epsilon, k, sparse, dtype)

# PySubgraph method  
def adjacency(self, map_index=True, sparse=True, dtype=np.float64):
    """Generate adjacency matrix for subgraph"""
    if map_index:
        # Compact matrix using subgraph node indices
        return self._build_compact_adjacency_matrix(sparse, dtype)
    else:
        # Full-size matrix with zeros for non-subgraph nodes
        return self._build_full_adjacency_matrix(sparse, dtype)

def laplacian(self, epsilon=-0.5, k=1, map_index=True, sparse=True, dtype=np.float64):
    """Generate Laplacian matrix for subgraph"""
    if map_index:
        return self._build_compact_laplacian_matrix(epsilon, k, sparse, dtype)
    else:
        return self._build_full_laplacian_matrix(epsilon, k, sparse, dtype)

def index_mapping(self):
    """Return mapping from compact indices to original node IDs"""
    return {i: node_id for i, node_id in enumerate(self.node_ids)}
```

## 🔄 Multi-Column Slicing and Advanced Access

### Current Multi-Column Slicing
```python
# Single column access (existing)
ages = g.nodes[:5]['age']                    # Returns GraphArray of age values

# Multi-column access returns GraphMatrix (COMPLETED!)
age_height = g.nodes[:5][['age', 'height']] # Returns GraphMatrix(columns=['age', 'height'], rows=5)
print(age_height.shape)                      # (5, 2) - 5 rows x 2 columns

# Access individual columns from GraphMatrix
ages = age_height['age']         # Returns GraphArray([25, 30, 35, 40, 45])
heights = age_height['height']   # Returns GraphArray([170, 165, 180, 175, 168])

# GraphMatrix provides structured access to multiple GraphArray columns
print(age_height.columns)        # ['age', 'height']
print(age_height.column_count)   # 2
print(age_height.row_count)      # 5

# Convert GraphMatrix to other formats
matrix_pandas = age_height.to_pandas()   # DataFrame with 'age' and 'height' columns
matrix_numpy = age_height.to_numpy()     # 2D NumPy array (5x2)

# Works with any subgraph
filtered = g.filter_nodes("age > 25")
multi_data = filtered[['salary', 'dept', 'active']]  # Returns GraphMatrix with 3 columns
print(multi_data.shape)  # (num_filtered_nodes, 3)
```

### Consistent GraphArray Returns (Planned)
```python
# All list outputs should be GraphArray for consistency and performance
attribute_values = g.attributes.nodes['age']   # Should be GraphArray
filtered_ids = g.filter_nodes('dept == "Engineering"').node_ids  # Should be GraphArray
edge_weights = g.attributes.edges['weight']    # Should be GraphArray
component_sizes = [len(comp.node_ids) for comp in g.connected_components()]  # Each comp.node_ids should be GraphArray
```

## 🧮 Algorithm and Graph Analysis

### Graph Algorithms (Completed)
```python
# All algorithms return Subgraph objects
components = g.connected_components()
for component in components:
    print(f"Component has {len(component.node_ids)} nodes")

# BFS/DFS traversal
visited = g.bfs(start_node=0)
visited = g.dfs(start_node=0)

# Shortest path
path = g.shortest_path(source=0, target=5)

# In-place operations supported
g.connected_components(inplace=True)
g.bfs(start_node=0, inplace=True)
```

### Graph Statistics and Metrics
```python
# Basic graph properties
print(f"Nodes: {g.node_count()}")
print(f"Edges: {g.edge_count()}")
print(f"Is connected: {g.is_connected()}")

# Node and edge statistics
node_degrees = [g.degree(node) for node in g.node_ids]
avg_degree = sum(node_degrees) / len(node_degrees)
```

## 📚 Advanced Features and Patterns

### Version Control and History
```python
# Version control functionality
g.commit("Added initial nodes")
g.create_branch("feature-branch")
g.checkout("feature-branch")

# Historical views
historical = g.get_historical_view(commit_id="abc123")
```

### Performance Patterns
```python
# Bulk operations for performance
node_data = [{"id": f"node_{i}", "value": i} for i in range(1000)]
edge_data = [(i, i+1, {"weight": 1.0}) for i in range(999)]

# Efficient bulk insertion
node_mapping = g.add_nodes(node_data, uid_key="id")
g.add_edges(edge_data)

# Memory-efficient filtering
large_subgraph = g.filter_nodes("value > 500")
```

### Data Export and Integration
```python
# Export to different formats
df = g.table().to_pandas()  # Pandas DataFrame
g.table().to_csv("nodes.csv")  # CSV export
g.table().to_json("nodes.json")  # JSON export

# Convert to NetworkX (if needed)
nx_graph = g.to_networkx()  # Hypothetical conversion
```

## 🔮 Future API Patterns (Planned)

### Unified View API (Planned)
```python
# All views support table() - returns GraphTable
g.table()                                   # All nodes (current ✅)
g.nodes.table()                             # same as above
g.edges.table()                             # All edges (current ✅)  
g.nodes[:5].table()                         # Subgraph nodes (NEW)
g.edges[:10].table()                        # Subgraph edges (NEW)
g.nodes[age > 30].table()                   # Filtered nodes table (NEW)

# Single entity views support dict() - returns dict
node_dict = g.nodes[0].dict()               # Node attributes + metadata (NEW)
g.edges[0].dict()                           # Edge attributes + metadata (NEW)
# Returns: {'id': 0, 'age': 25, 'name': 'Alice', '_graph_id': ...}

# Flexible attribute setting on single entities
g.nodes[0].set('age', 30)                   # Positional syntax (NEW)  
g.nodes[0].set(age=30)                      # Keyword syntax (NEW)
g.nodes[0].set(age=30, name="Alice")        # Multiple kwargs (NEW)
g.edges[0].set('weight', 0.8)               # Edge attributes (NEW)

# Column-subset table views
g.nodes[:5][['age', 'name']].table()        # Only specific columns (NEW)
g.edges[:10][['weight', 'type']].table()    # Edge column subsets (NEW)

# Chain operations naturally
engineers = g.filter_nodes('dept == "Engineering"')
young_eng = engineers[engineers['age'] < 30]  
young_eng_table = young_eng[['name', 'age', 'salary']].table()  # Filtered + selected columns
```

## 📈 Performance Examples

### Benchmark Results (Achieved)
```python
# Performance comparison vs NetworkX
# Graph Creation: 2.0x faster than NetworkX 🚀
# Filter Numeric Range: 1.4x faster 🚀  
# Filter Edges: 3.6x faster 🚀
# BFS Traversal: 11.5x faster 🚀
# Connected Components: 9.0x faster 🚀

# Scaling performance (250K nodes)
# Numeric Range: 74→83ns (Excellent O(n))
# Filter NOT: 141→124ns (Excellent O(n))  
# Connected Components: 348→355ns (Excellent O(n))
```

### Performance Optimization Examples
```python
# Efficient filtering patterns
# Single attribute filtering: ~109ns per operation
engineers = g.filter_nodes("dept == 'Engineering'")

# Complex AND queries: ~134ns per operation  
senior_engineers = g.filter_nodes("dept == 'Engineering' AND experience > 5")

# Memory-efficient large graphs
# 250K nodes: ~370MB memory usage
large_graph = gr.Graph()
# Bulk operations minimize memory fragmentation
```

### Critical Subgraph Accessor Issues (IDENTIFIED)
```python
# Issue 1: Connected components subgraphs lack graph reference
components[0].nodes  # RuntimeError: No graph reference available

# Issue 2: NodesAccessor missing .table() method  
subgraph.nodes.table()  # AttributeError: 'NodesAccessor' object has no attribute 'table'

# Issue 3: Inconsistent subgraph behavior between creation methods
g.filter_nodes('component_id == 0').nodes  # ✅ Works - Returns NodesAccessor
g.connected_components()[0].nodes           # ❌ RuntimeError (no graph reference)

# Expected universal subgraph API:
subgraph.nodes           # NodesAccessor with graph reference
subgraph.edges           # EdgesAccessor with graph reference  
subgraph.nodes.table()   # GraphTable of subgraph nodes
subgraph.edges.table()   # GraphTable of subgraph edges
subgraph.table()         # Combined GraphTable (pending implementation)
```

### Missing Subgraph Features (IDENTIFIED)
```python
# Issue: Subgraphs lack .table() method for DataFrame-like access
subgraph.table()  # AttributeError: 'PySubgraph' object has no attribute 'table'

# Expected: subgraph.table() should work like g.table() but only show subgraph entities
# Use Case: Analyze subgraph data in tabular format, export subgraph data to pandas/CSV

# Issue: GraphTable multi-column selection not implemented
table = g.table()
subset = table[['age','height']]  # TypeError: Key must be string, int, or slice
# Expected: Should return GraphTable with only the selected columns
```

---

## 📝 Notes

- ✅ **COMPLETED**: Features marked as completed are currently implemented
- 🔮 **PLANNED**: Features marked as planned are in development or design phase
- 🚀 **PERFORMANCE**: All statistical operations computed in native Rust for maximum speed
- 📊 **COMPATIBILITY**: GraphArray maintains full list compatibility while adding statistical capabilities
- 🔬 **SCIENTIFIC**: Strong integration planned with NumPy, SciPy, and Pandas ecosystems

This document will be updated as new features are implemented and the API evolves.
