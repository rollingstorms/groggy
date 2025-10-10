# Origins & Design: The Ultralight Example

## How Groggy Started

Groggy began with a mission: create a **high-performance Rust-based graph backend** combined with a **quick, intuitive Python frontend** capable of handling:

- Lightweight graph diagrams
- Heavy-duty machine learning algorithms
- Real-time graph analytics

The journey started with the "**ultralight example**" - an attempt to distill Groggy's essence into the smallest possible implementation. This exploration led to the core architectural decisions that define Groggy today.

---

## The Ultralight Example

### The Core Insight: Separation of Structure and Attributes

The ultralight example revealed a fundamental truth:

**Graph structure (topology) and graph data (attributes) should be stored separately.**

```
┌─────────────────┐         ┌──────────────────┐
│  Graph Structure│         │  Graph Attributes│
│   (Topology)    │ ←──────→│     (Signal)     │
├─────────────────┤         ├──────────────────┤
│ Nodes:  0,1,2,3 │         │ name: "Alice"    │
│ Edges:  0→1,1→2 │         │ age: 29          │
│         2→3,3→0 │         │ club: "Blue"     │
└─────────────────┘         └──────────────────┘
```

**Why separate them?**

1. **Efficient Bulk Operations**: Attributes stored in columnar format enable SIMD and cache-friendly operations
2. **Dynamic Graphs**: Track structural changes independently from attribute changes
3. **Version Control**: Store deltas efficiently without duplicating structure
4. **Clear Mental Model**: Reason about topology and data separately

---

## The Key Objects

The ultralight example introduced five core objects that remain central to Groggy:

### 1. AttributeValues

Storage for any data type:

```rust
enum AttributeValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    // ... more types
}
```

### 2. Delta Objects

Track changes over time (essential for dynamic graphs):

```rust
struct Delta {
    timestamp: u64,
    change_type: ChangeType,  // Add, Remove, Modify
    entity_id: usize,
    old_value: Option<AttributeValue>,
    new_value: Option<AttributeValue>,
}
```

**Key insight:** Every change is tracked, nothing is lost. This enables:
- Time-travel queries
- Audit trails
- Reproducible experiments

### 3. GraphSpace

The **active state** of the graph—which nodes and edges are currently "alive":

```rust
struct GraphSpace {
    live_nodes: BitSet,      // Which nodes exist
    live_edges: BitSet,      // Which edges exist
    node_count: usize,
    edge_count: usize,
}
```

**Design principle:** Nodes/edges are never deleted, only marked as inactive. This enables:
- O(1) node/edge queries
- Efficient state restoration
- Version history without data loss

### 4. GraphPool

The **flyweight pool** containing all attributes:

```rust
struct GraphPool {
    node_attrs: ColumnarStorage,  // All node attributes
    edge_attrs: ColumnarStorage,  // All edge attributes
    attr_index: HashMap<String, ColumnId>,
}
```

**Critical design pattern:** Attributes are never stored inside nodes/edges. Nodes and edges only **point to** attributes.

```
Node {id: 0} ──→ GraphPool["name"][0] = "Alice"
             ──→ GraphPool["age"][0] = 29
             ──→ GraphPool["club"][0] = "Blue"
```

### 5. HistoryForest

Git-like version control for graphs:

```rust
struct HistoryForest {
    commits: Vec<Commit>,
    branches: HashMap<String, BranchId>,
    current_branch: BranchId,
}

struct Commit {
    id: CommitId,
    parent: Option<CommitId>,
    deltas: Vec<Delta>,
    message: String,
    timestamp: u64,
}
```

This enables:
- Branching and merging graph states
- Time-travel queries
- A/B testing on graph structure
- Reproducible experiments

---

## The Columnar Architecture Decision

To support both graph operations and machine learning workflows, Groggy needed **rectangular data**. The solution: **columnar storage**.

### Why Columnar?

Traditional graph libraries store attributes like this:

```python
# Node-centric storage (inefficient for bulk ops)
node = {
    "id": 0,
    "name": "Alice",
    "age": 29,
    "club": "Blue"
}
```

Groggy stores them like this:

```python
# Columnar storage (efficient for bulk ops)
node_pool = {
    "name": ["Alice", "Bob", "Carol", ...],  # Contiguous memory
    "age":  [29, 55, 31, ...],               # SIMD-friendly
    "club": ["Blue", "Purple", "Blue", ...]   # Cache-friendly
}
```

**Benefits:**

1. **Vectorized Operations**: Process entire columns at once
2. **Cache Efficiency**: Sequential memory access patterns
3. **Compression**: Columnar data compresses better
4. **Analytics**: Natural fit for data science workflows

### Example: Mean Age Computation

**Traditional (row-wise):**
```python
total = 0
for node in graph.nodes:
    total += node["age"]  # Memory scattered, cache misses
mean = total / len(graph.nodes)
```

**Groggy (columnar):**
```python
ages = graph.nodes["age"]  # Single contiguous array
mean = ages.mean()         # Vectorized, SIMD-optimized
```

The columnar approach is 10-100x faster for bulk operations.

---

## From Ultralight to Full Implementation

The ultralight example established the foundation. The full Groggy implementation expands it with:

### Additional Components

1. **Display System**: Rich formatting for notebooks and terminals
2. **Query Engine**: Pandas-style filtering and selection
3. **Algorithm Library**: Connected components, centrality, etc.
4. **Neural Module**: Graph neural networks with autodiff
5. **Visualization**: Real-time graph rendering
6. **I/O System**: Parquet, CSV, bundles, pandas integration

### Three-Tier Architecture

The ultralight concepts map to the three-tier architecture:

```
Python API Layer
  ├─ Graph, Subgraph, Table, Array, Matrix
  └─ User-facing delegation chains

FFI Bridge (PyO3)
  ├─ Type conversions
  └─ Safe error handling

Rust Core
  ├─ GraphSpace (active state)
  ├─ GraphPool (attribute storage)
  ├─ HistoryForest (version control)
  ├─ Delta tracking
  └─ Algorithms
```

---

## Design Principles from Ultralight

The ultralight example taught us these principles:

### 1. Separation of Concerns
Structure and attributes are independent. This makes each simpler to reason about and optimize.

### 2. Everything is Append-Only
Never delete, only mark as inactive. This enables:
- Efficient version control
- Time-travel queries
- Simpler concurrent access patterns

### 3. Columnar is Fundamental
Not an optimization—it's the core design. Structure is graph, data is columnar.

### 4. Track All Changes
Deltas are first-class citizens. Every change is recorded, enabling reproducibility and audit trails.

### 5. Views, Not Copies
Create views into data rather than copying. Subgraphs, tables, and arrays are all views.

---

## The Philosophy: Everything is a Graph

Even Groggy's architecture is a graph:

- **Nodes** = Core objects (GraphSpace, GraphPool, Delta, etc.)
- **Edges** = Dependencies and transformations

This recursive thinking influenced the API design where objects transform into each other via delegation chains.

---

## Key Takeaways

The ultralight example established these enduring truths:

1. **Structure ≠ Signal**: Separate topology from attributes
2. **Columnar is Key**: Bulk operations are the common case
3. **Track Everything**: Deltas enable time-travel and reproducibility
4. **Views > Copies**: Immutable views are cheap and safe
5. **Graph Thinking**: Apply graph concepts recursively

These principles guide every design decision in Groggy today.

---

## Next Steps

- **[Architecture Deep Dive](architecture.md)**: Detailed look at the three-tier system
- **[Connected Views](connected-views.md)**: Master object transformations
- **[User Guide](../guide/graph-core.md)**: Start building with these concepts
