# Appendix A: Glossary

**Complete reference of Groggy terminology and concepts**

---

## Core Concepts

### Graph
The main data structure in Groggy representing a network of nodes and edges. A Graph maintains both the **graph structure** (topology) and **graph signal** (attributes) in a columnar storage system. Implemented as a thin Python wrapper over Rust core data structures.

**See:** [Graph API](../api/graph.md), [Graph Core Guide](../guide/graph-core.md)

### Node
An entity in the graph, also called a vertex. Nodes can have arbitrary attributes stored in the columnar pool. Identified by a unique `NodeId`.

**Related:** Edge, NodeId, Attribute

### Edge
A connection between two nodes. Edges can be directed or undirected and can have arbitrary attributes. Identified by a unique `EdgeId`.

**Related:** Node, EdgeId, Directed Graph, Undirected Graph

### NodeId
Integer identifier for a node. Used to reference nodes when creating edges and querying the graph.

**Type:** Integer (Rust: u32 or u64)

### EdgeId
Integer identifier for an edge. Used to reference edges in queries and operations.

**Type:** Integer (Rust: u32 or u64)

---

## Graph Types

### Directed Graph
A graph where edges have direction - they go from a source node to a target node. The default in Groggy.

**Example:** Social network following relationships, web page links, dependency graphs

### Undirected Graph
A graph where edges have no direction - the connection is bidirectional.

**Example:** Friendships, physical proximity, collaboration networks

### Weighted Graph
A graph where edges (or nodes) have numerical weights representing strength, cost, distance, or other metrics.

**Related:** Attribute, NumArray

### Multigraph
A graph that allows multiple edges between the same pair of nodes. Supported through edge attributes in Groggy.

---

## Views and Substructures

### Subgraph
A view into a Graph containing a subset of nodes and/or edges. Subgraphs are **immutable views** that reference the parent graph without copying data. Created through filtering operations.

**See:** [Subgraph API](../api/subgraph.md), [Subgraphs Guide](../guide/subgraphs.md)

**Related:** View, Induced Subgraph, Filter

### Induced Subgraph
A subgraph containing a specified set of nodes and all edges between those nodes in the parent graph.

**Example:**
```python
nodes_to_keep = [0, 1, 2, 5]
induced = g.nodes[nodes_to_keep]  # Subgraph with all edges between these nodes
```

### View
An immutable, lightweight reference to graph data that doesn't copy the underlying structure. Most Groggy operations return views rather than copies for performance.

**Examples:** Subgraph, Array slice, Table view

**Related:** Materialization, Lazy Evaluation

### SubgraphArray
A collection of Subgraph objects, typically the result of graph algorithms like connected components or community detection. Supports delegation chains for batch operations.

**See:** [SubgraphArray API](../api/subgrapharray.md), [Subgraph Arrays Guide](../guide/subgraph-arrays.md)

---

## Accessors

### NodesAccessor
The `g.nodes` accessor providing node-level operations and queries. Enables filtering nodes, accessing attributes, and creating node-based subgraphs.

**Access Pattern:** `g.nodes[condition]` or `g.nodes["attribute"]`

**See:** [NodesAccessor API](../api/nodesaccessor.md), [Accessors Guide](../guide/accessors.md)

### EdgesAccessor
The `g.edges` accessor providing edge-level operations and queries. Enables filtering edges, accessing attributes, and creating edge-based subgraphs.

**Access Pattern:** `g.edges[condition]` or `g.edges["attribute"]`

**See:** [EdgesAccessor API](../api/edgesaccessor.md), [Accessors Guide](../guide/accessors.md)

---

## Data Structures

### Attribute
Data attached to nodes or edges. Attributes are stored separately from the graph structure in a columnar pool for efficient bulk operations.

**Key Concept:** Nodes and edges only **point to** attributes; they never store them directly.

**Example:**
```python
g.add_node(name="Alice", age=29, role="Engineer")
# name, age, role are attributes
```

### Columnar Storage
Storage architecture where each attribute is stored as a separate column (array) rather than row-wise. Enables efficient bulk operations and SIMD optimization.

**Benefits:**
- Cache-friendly access patterns
- Efficient filtering and aggregation
- Optimal for machine learning workflows

**Related:** GraphPool, Attribute

### GraphSpace
The active state of the graph - which nodes and edges are currently alive. Part of the core Rust implementation.

**Related:** GraphPool, HistoryForest

### GraphPool
The flyweight pool containing all node and edge attributes in columnar format. Part of the core Rust implementation.

**Related:** GraphSpace, Columnar Storage

### HistoryForest
Git-like version control system for graphs, enabling time-travel queries and branching. Part of the core Rust implementation.

**Related:** State, Branch, Version Control

---

## Tables

### GraphTable
Unified tabular view of the entire graph containing both node and edge data. Can be split into NodesTable and EdgesTable.

**See:** [GraphTable API](../api/graphtable.md), [Tables Guide](../guide/tables.md)

### NodesTable
Tabular view of node data with columns for node attributes. Each row represents a node.

**See:** [NodesTable API](../api/nodestable.md)

### EdgesTable
Tabular view of edge data with columns for edge attributes. Each row represents an edge.

**See:** [EdgesTable API](../api/edgestable.md)

### BaseTable
Base table type providing common table operations. Other table types (GraphTable, NodesTable, EdgesTable) inherit from this.

**See:** [BaseTable API](../api/basetable.md)

---

## Arrays

### BaseArray
Base array type for attribute data. Provides common array operations and can be specialized to NumArray for numeric data.

**Related:** NumArray, Columnar Storage

### NumArray
Specialized array for numeric data with statistical and mathematical operations (mean, sum, min, max, etc.).

**See:** [NumArray API](../api/numarray.md), [Arrays Guide](../guide/arrays.md)

### NodesArray
Array of node IDs, typically returned from node queries. Supports node-specific operations.

**See:** [NodesArray API](../api/nodesarray.md)

### EdgesArray
Array of edge IDs, typically returned from edge queries. Supports edge-specific operations.

**See:** [EdgesArray API](../api/edgesarray.md)

---

## Matrices

### GraphMatrix
Matrix representation of graph data, including adjacency matrices, Laplacian matrices, and embeddings.

**See:** [GraphMatrix API](../api/graphmatrix.md), [Matrices Guide](../guide/matrices.md)

### Adjacency Matrix
Matrix A where A[i,j] = 1 (or edge weight) if there's an edge from node i to node j, 0 otherwise.

**Example:**
```python
A = g.adjacency_matrix()
```

### Laplacian Matrix
Matrix L = D - A where D is the degree matrix and A is the adjacency matrix. Used in spectral graph theory.

**Types:**
- **Graph Laplacian:** `L = D - A`
- **Normalized Laplacian:** `L_norm = I - D^(-1/2) A D^(-1/2)`
- **Random Walk Laplacian:** `L_rw = I - D^(-1) A`

**Example:**
```python
L = g.laplacian_matrix()
```

### Degree Matrix
Diagonal matrix D where D[i,i] equals the degree (number of connections) of node i.

### Spectral Embedding
Low-dimensional representation of graph structure derived from eigenvectors of the Laplacian matrix.

**Example:**
```python
embedding = g.spectral().compute(dims=128)
```

**Related:** Laplacian Matrix, Embedding

---

## Algorithms

### Connected Components
Maximal subgraphs where every node is reachable from every other node. For directed graphs, can be strongly or weakly connected.

**Example:**
```python
components = g.connected_components()
# Returns SubgraphArray
```

**See:** [Algorithms Guide](../guide/algorithms.md)

### Shortest Path
Minimum-length path between two nodes, measured by edge count or edge weights.

**Algorithms:**
- Dijkstra's algorithm (weighted)
- Breadth-first search (unweighted)

### Centrality
Measure of a node's importance in the network.

**Types:**
- **Degree Centrality:** Number of connections
- **Betweenness Centrality:** Number of shortest paths through node
- **Closeness Centrality:** Average distance to all other nodes
- **Eigenvector Centrality:** Importance based on neighbor importance

### Community Detection
Algorithms for finding groups of densely connected nodes.

**Algorithms:**
- Modularity optimization
- Label propagation
- Louvain method

---

## Operations

### Delegation
Pattern where objects forward method calls to related objects, enabling chainable operations.

**Example:**
```python
result = (g.connected_components()
          .sample(5)
          .neighborhood(depth=2)
          .table()
          .agg({"weight": "mean"}))
```

**See:** [Connected Views](../concepts/connected-views.md)

### Chaining
Calling multiple methods in sequence where each returns an object supporting the next method. Made possible by delegation.

**Related:** Delegation, Method Forwarding

### Filtering
Selecting a subset of nodes or edges based on conditions. Returns a Subgraph view.

**Example:**
```python
young = g.nodes[g.nodes["age"] < 30]
heavy_edges = g.edges[g.edges["weight"] > 5.0]
```

### Materialization
Converting a lazy view into concrete data. Most operations in Groggy are lazy until materialized.

**Example:**
```python
df = subgraph.table().to_pandas()  # Materializes as DataFrame
```

**Related:** View, Lazy Evaluation

### Aggregation
Computing summary statistics across groups or the entire graph.

**Example:**
```python
result = g.table().agg({"weight": ["mean", "sum", "count"]})
```

---

## Architecture Terms

### Three-Tier Architecture
Groggy's architectural layers:
1. **Rust Core:** High-performance algorithms and storage
2. **FFI Bridge:** PyO3 bindings (pure translation, no logic)
3. **Python API:** User-facing interface

**See:** [Architecture](../concepts/architecture.md)

### FFI (Foreign Function Interface)
The PyO3-based bridge between Rust and Python. Contains pure translation code with no business logic.

**Related:** PyO3, Three-Tier Architecture

### PyO3
Rust framework for creating Python extensions. Used to expose Rust core to Python.

### Method Forwarding
Technique where one object type delegates method calls to another type it can transform into.

**Example:** SubgraphArray forwards `table()` method which returns GraphTable

---

## State and Versioning

### State
A snapshot of the graph at a point in time, including which nodes/edges are alive and their attributes.

**Related:** GraphSpace, StateId

### StateId
Identifier for a specific graph state in the history system.

### Branch
Named sequence of graph states, similar to Git branches. Enables parallel development of graph versions.

**Related:** HistoryForest, Version Control

### BranchName
String identifier for a branch in the history system.

### Commit
Saving the current graph state to history. Creates a new StateId.

### Inplace Operation
Operation that modifies the graph in place rather than returning a new object.

**Example:**
```python
g.connected_components(inplace=True, label="component")
# Writes 'component' attribute to nodes
```

---

## Data Interchange

### Bundle
Serialized graph format containing both structure and attributes. Used for save/load operations.

**Example:**
```python
g.save_bundle("graph.bundle")
g2 = gr.GraphTable.load_bundle("graph.bundle")
```

**Related:** GraphTable, Serialization

### Parquet
Apache Parquet format for efficient columnar storage. Used for table export/import.

**Example:**
```python
g.nodes.table().to_parquet("nodes.parquet")
```

---

## Performance Terms

### Amortized Complexity
Average time complexity over a sequence of operations, accounting for occasional expensive operations.

**Example:** Graph node insertion is O(1) amortized

### Columnar Operation
Bulk operation on an entire attribute column, typically SIMD-optimized for performance.

**Related:** Columnar Storage

### Sparse Matrix
Matrix where most entries are zero. Stored efficiently using sparse formats (CSR, COO).

**Related:** GraphMatrix, Adjacency Matrix

### Dense Matrix
Matrix stored as a complete 2D array. More memory but faster access for dense data.

---

## Integration Terms

### NetworkX Compatibility
Ability to convert between Groggy and NetworkX graph formats.

**See:** [Integration Guide](../guide/integration.md)

### Pandas Integration
Converting Groggy tables to/from pandas DataFrames.

**Example:**
```python
df = g.nodes.table().to_pandas()
```

### NumPy Interop
Converting Groggy arrays and matrices to/from NumPy arrays.

**Example:**
```python
np_array = num_array.to_numpy()
```

---

## Common Abbreviations

| Abbreviation | Full Term |
|--------------|-----------|
| API | Application Programming Interface |
| FFI | Foreign Function Interface |
| CSR | Compressed Sparse Row (matrix format) |
| COO | Coordinate List (matrix format) |
| SIMD | Single Instruction Multiple Data |
| GIL | Global Interpreter Lock (Python) |
| ADR | Architectural Decision Record |
| BFS | Breadth-First Search |
| DFS | Depth-First Search |

---

## Quick Reference

### Common Object Transformations

```
Graph → Subgraph → SubgraphArray → GraphTable → NodesTable/EdgesTable
      → BaseArray → NumArray
      → GraphMatrix
```

### Access Patterns

```python
g.nodes[condition]          # Filter → Subgraph
g.nodes["attribute"]        # Column → BaseArray
g.edges[condition]          # Filter → Subgraph
g["attribute"]              # Alias for g.nodes["attribute"]
g.table()                   # Graph → GraphTable
g.adjacency_matrix()        # Graph → GraphMatrix
```

---

## See Also

- **[Architecture](../concepts/architecture.md)** - System design and structure
- **[Connected Views](../concepts/connected-views.md)** - Object transformation graph
- **[Performance Cookbook](performance-cookbook.md)** - Optimization patterns
- **[Design Decisions](design-decisions.md)** - Architectural rationale
