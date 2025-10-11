# Concepts & Architecture Overview

This section explains the core concepts and architectural decisions that make Groggy unique.

---

## The Big Ideas

### 1. Structure vs. Signal

The foundational insight of Groggy's design: **graph structure** and **graph data** are stored separately.

- **Structure** (topology): Which nodes and edges exist, and how they're connected
- **Signal** (attributes): The data associated with each node and edge

This separation enables:
- Efficient bulk attribute operations
- Version control without duplicating data
- Clear conceptual model for reasoning about graphs

### 2. Everything is a View

When you work with a graph in Groggy, you're usually working with an **immutable view**:

- **Subgraphs** are views into the main graph
- **Tables** are snapshots of graph state
- **Arrays** are columnar views of attributes
- **Matrices** represent graph structure or embeddings

Views are cheap to create and enable powerful composition without copying data.

### 3. Delegation & Transformation

Objects in Groggy know how to transform into other objects. This creates an intuitive **transformation graph**:

```
Graph → Subgraph → SubgraphArray → Table → Array
  ↓         ↓           ↓              ↓       ↓
Matrix    Matrix      Table         Matrix  Table
```

Once you learn the transformation patterns, the entire API becomes predictable.

### 4. Columnar Thinking

Groggy optimizes for **bulk operations** over single-item loops:

```python
# Fast: operate on entire columns
ages = g.nodes["age"]
mean_age = ages.mean()

# Slower: iterate node by node
total = 0
for node in g.nodes:
    total += node["age"]
mean_age = total / len(g.nodes)
```

This design enables SIMD, parallelization, and cache-friendly access patterns.

---

## Three-Tier Architecture

Groggy is built in three layers, each with a specific purpose:

```
┌──────────────────────────────────────┐
│        Python API Layer              │  User-facing
│          (Thin Wrapper)              │  Intuitive, chainable
├──────────────────────────────────────┤
│          FFI Bridge                  │  Translation only
│         (PyO3 bindings)              │  No business logic
├──────────────────────────────────────┤
│         Rust Core                    │  Performance
│  (Storage, State, Algorithms)        │  Memory-safe
└──────────────────────────────────────┘
```

### Rust Core
- All algorithms and data structures
- Columnar storage backend
- State management and history
- Performance-critical operations

### FFI Bridge
- Pure translation layer (PyO3)
- No business logic
- Safe error handling
- GIL management for parallelism

### Python API
- User-facing interface
- Integration with PyData ecosystem
- Notebook-friendly display

**Why this separation?**
- **Performance**: Rust handles compute-intensive operations
- **Safety**: Rust's memory model prevents entire classes of bugs
- **Ergonomics**: Python provides a friendly, familiar API
- **Maintainability**: Clear boundaries make the codebase easier to reason about

---

## Core Objects

Groggy provides five main object types:

### 1. Graph
The foundational object representing a complete graph.

```python
g = gr.Graph()
node_id = g.add_node(name="Alice")
edge_id = g.add_edge(node1, node2)
```

**Internally:**
- **GraphSpace**: Which nodes/edges are alive (topology)
- **GraphPool**: Where attributes are stored (columnar)

### 2. Subgraph
A view into a subset of a graph.

```python
sub = g.nodes[:10]          # First 10 nodes
filtered = g.nodes[mask]    # Boolean filter
```

**Key insight:** Subgraphs don't copy data—they reference the parent graph.

### 3. Table
Tabular view of graph data (nodes, edges, or both).

```python
nodes_table = g.nodes.table()
edges_table = g.edges.table()
graph_table = g.table()  # Both
```

**Types:**
- **NodesTable**: Node attributes + node IDs
- **EdgesTable**: Edge attributes + src/dst
- **GraphTable**: Complete graph as two tables

### 4. Array
Columnar view of a single attribute or collection of values.

```python
ages = g.nodes["age"]        # BaseArray
ids = g.nodes.node_ids()     # NodesArray
components = g.connected_components()  # SubgraphArray
```

**Types:**
- **BaseArray**: Generic array of values
- **NumArray**: Numeric array with statistical operations
- **NodesArray**: Array of node IDs
- **EdgesArray**: Array of edge IDs
- **SubgraphArray**: Array of subgraphs (enables delegation chains)

### 5. Matrix
Matrix representation of graph structure or embeddings.

```python
A = g.to_matrix()                  # Adjacency matrix
L = g.laplacian_matrix()           # Laplacian
embedding = g.spectral().compute(8)  # Spectral embedding
```

**Use cases:**
- Linear algebra operations
- Spectral analysis
- Machine learning features

---

## The Object Transformation Graph

Groggy itself is a graph! Here's how objects transform:

```
                    ┌─────────┐
                    │  Graph  │
                    └────┬────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ↓                ↓                ↓
   ┌────────┐      ┌──────────┐     ┌────────┐
   │Subgraph│      │GraphTable│     │ Array  │
   └────┬───┘      └────┬─────┘     └────┬───┘
        │               │                 │
        ↓               ↓                 ↓
  ┌───────────┐    ┌────────┐       ┌────────┐
  │SubgraphArr│    │ Table  │       │ Matrix │
  └─────┬─────┘    └────────┘       └────────┘
        │
        ↓
   ┌──────────┐
   │ TableArr │
   └──────────┘
```

**Example transformation path:**
```python
result = (
    g                            # Graph
     .connected_components()     # → SubgraphArray
     .sample(5)                  # → SubgraphArray (filtered)
     .table()                    # → GraphTable
     .agg({"weight": "mean"})    # → AggregationResult
)
```

---

## Design Principles

### 1. Attribute-First Optimization
Structure and attributes are separate, with attributes stored in columnar format for efficient bulk operations.

### 2. Immutable Views
Operations return views when possible, avoiding unnecessary data copying.

### 3. Explicit Materialization
You control when data is materialized (e.g., `to_graph()`, `to_pandas()`).

### 4. Composition Over Inheritance
Objects compose via delegation rather than deep inheritance hierarchies.

### 5. Performance by Default
O(1) amortized complexity for core operations, with linear memory scaling.

---

## Key Innovations

### Delegation Chains
Objects forward methods to enable expressive pipelines:

```python
g.connected_components().filter(lambda c: len(c.nodes) > 5).table()
```

### Columnar Attribute Storage
Attributes stored separately from structure in columnar format:
- Fast bulk reads/writes
- Efficient compression
- Cache-friendly access patterns

### Git-Like Versioning
Graph states can be committed, branched, and merged:
- Time-travel queries
- A/B testing on graph structure
- Reproducible experiments

### Zero-Copy Views
Subgraphs and tables are views—no data duplication:
- Low memory overhead
- Fast creation
- Safe (immutable views prevent accidental modification)

---

## Next Steps

Now that you understand the core concepts:

- **[Origins & Design](origins.md)**: Learn how the ultralight example informed Groggy's architecture
- **[Architecture Deep Dive](architecture.md)**: Detailed look at the three-tier system
- **[Connected Views](connected-views.md)**: Master the transformation graph

Or jump straight into the [User Guide](../guide/graph-core.md) to start building!
