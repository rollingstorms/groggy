# Architecture Deep Dive

## The Three-Tier System

Groggy is built as a three-layer architecture, with each layer having a specific responsibility and clear boundaries.

```
┌────────────────────────────────────────────────────┐
│              Python API Layer                      │
│                                                    │
│  • User-facing objects (Graph, Table, Array, etc.) │
│  • Delegation chains and method forwarding         │
│  • Integration with PyData ecosystem               │
│  • Notebook-friendly display                       │
│                                                    │
├────────────────────────────────────────────────────┤
│                 FFI Bridge (PyO3)                  │
│                                                    │
│  • Trait-backed explicit methods (v0.5.1+)         │
│  • Type conversions (Python ↔ Rust)                │
│  • Safe error handling and propagation             │
│  • GIL management for parallelism                  │
│  • ~100ns per-call FFI budget maintained           │
│                                                    │
├────────────────────────────────────────────────────┤
│                  Rust Core                         │
│                                                    │
│  • GraphSpace (active state / topology)            │
│  • GraphPool (columnar attribute storage)          │
│  • HistoryForest (version control)                 │
│  • Algorithm implementations                       │
│  • All performance-critical operations             │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## Layer 1: Rust Core

The foundation of Groggy. All algorithms, data structures, and performance-critical code lives here.

### Core Components

#### 1. GraphSpace
The **active state** of the graph—which nodes and edges are alive.

```rust
pub struct GraphSpace {
    live_nodes: BitSet,
    live_edges: BitSet,
    node_count: usize,
    edge_count: usize,
    adjacency: AdjacencyList,
}
```

**Responsibilities:**
- Track which nodes/edges exist
- Maintain adjacency relationships
- Provide O(1) existence checks
- Support efficient iteration over live entities

**Key Operations:**
```rust
// O(1) - Check if node exists
space.contains_node(node_id) -> bool

// O(1) amortized - Add node
space.add_node() -> NodeId

// O(1) - Mark node as inactive (doesn't delete)
space.remove_node(node_id)

// O(deg(node)) - Get neighbors
space.neighbors(node_id) -> Iterator<NodeId>
```

#### 2. GraphPool
The **flyweight pool** storing all attributes in columnar format.

```rust
pub struct GraphPool {
    node_attrs: ColumnarStorage,
    edge_attrs: ColumnarStorage,
    attr_names: HashMap<String, AttrId>,
}

pub struct ColumnarStorage {
    columns: Vec<Column>,
    indices: HashMap<EntityId, RowId>,
}

pub enum Column {
    IntColumn(Vec<i64>),
    FloatColumn(Vec<f64>),
    StringColumn(Vec<String>),
    BoolColumn(Vec<bool>),
    // ... more types
}
```

**Responsibilities:**
- Store attributes separately from structure
- Provide columnar access for bulk operations
- Manage type-specific storage
- Handle sparse attributes efficiently

**Key Operations:**
```rust
// O(1) - Get attribute value
pool.get_attr(entity_id, "name") -> Option<AttrValue>

// O(1) - Set attribute value
pool.set_attr(entity_id, "name", value)

// O(1) - Get entire column
pool.get_column("age") -> &Vec<i64>

// O(n) - Set column
pool.set_column("age", values)
```

#### 3. HistoryForest
Git-like version control for graphs.

```rust
pub struct HistoryForest {
    commits: HashMap<CommitId, Commit>,
    branches: HashMap<BranchName, BranchId>,
    current: StateId,
    deltas: DeltaLog,
}

pub struct Commit {
    id: CommitId,
    parent: Option<CommitId>,
    message: String,
    timestamp: u64,
    state_snapshot: StateId,
}

pub struct Delta {
    change_type: ChangeType,
    entity_id: EntityId,
    attribute: String,
    old_value: Option<AttrValue>,
    new_value: Option<AttrValue>,
}
```

**Responsibilities:**
- Track all graph changes as deltas
- Support branching and merging
- Enable time-travel queries
- Maintain commit history

**Key Operations:**
```rust
// Create a commit
history.commit("message") -> CommitId

// Create a branch
history.branch("feature-x") -> BranchId

// Checkout a state
history.checkout(commit_id)

// Time-travel query
history.at_time(timestamp) -> GraphView
```

#### 4. Algorithm Implementations

All graph algorithms live in the Rust core:

```rust
// Connected components
pub fn connected_components(
    space: &GraphSpace
) -> Vec<ComponentId>

// Shortest paths
pub fn dijkstra(
    space: &GraphSpace,
    pool: &GraphPool,
    source: NodeId,
    weight_attr: &str
) -> HashMap<NodeId, Distance>

// PageRank
pub fn pagerank(
    space: &GraphSpace,
    damping: f64,
    max_iter: usize
) -> Vec<f64>
```

**Design principle:** FFI contains **zero** algorithm logic. Everything is implemented in Rust and exposed via simple interfaces.

---

## Layer 2: FFI Bridge (PyO3)

The translation layer between Python and Rust. Contains **no business logic**.

!!! info "v0.5.1+ Architecture"
    Starting in v0.5.1, the FFI layer uses **explicit trait-backed delegation** instead of dynamic attribute lookups. All methods are written explicitly in `#[pymethods]` blocks for discoverability and maintainability. See [Trait-Backed Delegation](trait-delegation.md) for details.

### Responsibilities

1. **Type Conversion**: Python ↔ Rust type mapping
2. **Error Handling**: Rust `Result<T, E>` → Python exceptions
3. **Memory Safety**: Ensure safe cross-language boundaries
4. **GIL Management**: Release GIL for long-running operations
5. **Trait Delegation**: Route Python calls to shared Rust trait implementations
6. **Builder Pipelines**: Execute JSON step specs through `builder.step_pipeline`

### Example FFI Binding (Modern Trait-Backed Approach)

```rust
#[pyclass]
pub struct PyGraph {
    inner: Arc<RwLock<CoreGraph>>,
    // Cached full view for efficient trait method calls
    full_view_cache: Arc<RwLock<Option<Subgraph>>>,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> Self {
        PyGraph {
            inner: Arc::new(RwLock::new(CoreGraph::new())),
            full_view_cache: Arc::new(RwLock::new(None)),
        }
    }

    fn add_node(&self, py: Python, attrs: HashMap<String, PyObject>) -> PyResult<NodeId> {
        // 1. Convert Python types to Rust
        let rust_attrs = convert_attrs(attrs)?;

        // 2. Release GIL for Rust operation
        let node_id = py.allow_threads(|| {
            let mut graph = self.inner.write().unwrap();
            graph.add_node_with_attrs(rust_attrs)
        });

        // 3. Return result
        Ok(node_id)
    }

    // Modern trait-backed delegation using with_full_view helper
    fn connected_components(slf: PyRef<Self>, py: Python) -> PyResult<PyComponentsArray> {
        // Helper manages cache, GIL release, and error translation
        Self::with_full_view(slf, py, |subgraph, _py| {
            // Call shared trait method (zero business logic in FFI)
            let components = subgraph
                .inner
                .connected_components()  // Trait method from SubgraphOps
                .map_err(graph_error_to_py_err)?;
            
            // Convert result to Python type
            Ok(PyComponentsArray::from_components(components, subgraph.inner.graph().clone()))
        })
    }
}

// Helper for consistent trait delegation
impl PyGraph {
    fn with_full_view<F, R>(
        slf: PyRef<Self>,
        py: Python,
        f: F,
    ) -> PyResult<R>
    where
        F: FnOnce(&PyGraph, Python) -> PyResult<R> + Send,
        R: Send,
    {
        // Get or create cached full view
        let needs_refresh = {
            let cache = slf.full_view_cache.read().unwrap();
            cache.is_none()
        };
        
        if needs_refresh {
            let graph = slf.inner.read().unwrap();
            let full_view = graph.full_subgraph();
            *slf.full_view_cache.write().unwrap() = Some(full_view);
        }
        
        // Release GIL for potentially long operation
        py.allow_threads(|| f(&slf, py))
    }
}
```

### Key Principles

**1. Pure Translation**
```rust
// GOOD: Just translates
fn add_node(&self, attrs: HashMap<String, PyObject>) -> PyResult<NodeId> {
    let rust_attrs = convert_attrs(attrs)?;
    Ok(self.inner.add_node(rust_attrs))
}

// BAD: Contains algorithm logic in FFI
fn add_node_with_validation(&self, attrs: HashMap<String, PyObject>) -> PyResult<NodeId> {
    // ❌ Business logic in FFI layer
    if attrs.contains_key("invalid") {
        return Err(PyValueError::new_err("Invalid attribute"));
    }
    // This validation should be in Rust core!
    ...
}
```

**2. Safe Error Handling**
```rust
fn operation(&self) -> PyResult<T> {
    self.inner
        .rust_operation()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
```

**3. GIL Management**
```rust
// Release GIL for long operations
fn expensive_operation(&self, py: Python) -> PyResult<T> {
    py.allow_threads(|| {
        self.inner.do_expensive_work()
    })
}
```

---
<!--  This isnt true! we have most python methods in the ffi as pyo3 methods. 
## Layer 3: Python API

The user-facing layer. Designed for ergonomics and integration.

### Core Objects

#### 1. Graph
```python
class Graph:
    """Main graph object - wraps PyGraph from FFI"""

    def __init__(self):
        self._graph = _groggy.Graph()  # FFI object

    @property
    def nodes(self):
        """Returns NodesAccessor for node operations"""
        return NodesAccessor(self._graph)

    @property
    def edges(self):
        """Returns EdgesAccessor for edge operations"""
        return EdgesAccessor(self._graph)

    def __getitem__(self, attr_name):
        """Direct attribute access: g["name"]"""
        return self._graph.get_column(attr_name)
```

#### 2. Delegation & Chaining

The Python layer implements method forwarding:

```python
class SubgraphArray:
    """Array of subgraphs with delegation"""

    def __init__(self, subgraphs):
        self._subgraphs = subgraphs

    def sample(self, n):
        """Sample n subgraphs - returns SubgraphArray"""
        sampled = random.sample(self._subgraphs, n)
        return SubgraphArray(sampled)

    def neighborhood(self, depth=1):
        """Expand to neighborhoods - returns SubgraphArray"""
        expanded = [sg.expand_neighbors(depth) for sg in self._subgraphs]
        return SubgraphArray(expanded)

    def table(self):
        """Convert to table - returns GraphTable"""
        return GraphTable.from_subgraphs(self._subgraphs)

    def filter(self, predicate):
        """Filter subgraphs - returns SubgraphArray"""
        filtered = [sg for sg in self._subgraphs if predicate(sg)]
        return SubgraphArray(filtered)
``` 

This enables chaining:
```python
result = (
    g.connected_components()    # → SubgraphArray
     .sample(5)                 # → SubgraphArray
     .neighborhood(depth=2)     # → SubgraphArray
     .table()                   # → GraphTable
)
```
#### 3. Integration Layer

```python
class GraphTable:
    """Integration with pandas, numpy, etc."""

    def to_pandas(self):
        """Convert to pandas DataFrame"""
        return pd.DataFrame(self._data)

    def to_parquet(self, path):
        """Export to Parquet"""
        df = self.to_pandas()
        df.to_parquet(path)

    @classmethod
    def from_pandas(cls, nodes_df, edges_df):
        """Import from pandas DataFrames"""
        # Convert and create GraphTable
        ...
```
-->

---

## Data Flow

### Example: Adding a Node with Attributes

```
1. Python API Layer
   ↓
   g.add_node(name="Alice", age=29)
   ↓

2. FFI Bridge (PyO3)
   ↓
   • Convert Python dict → Rust HashMap
   • Convert Python types → Rust types
   • Release GIL
   ↓

3. Rust Core
   ↓
   • GraphSpace: Add node to bitset, update adjacency
   • GraphPool: Add attributes to columnar storage
   • Return NodeId
   ↓

4. FFI Bridge
   ↓
   • Convert Rust NodeId → Python int
   • Re-acquire GIL
   ↓

5. Python API Layer
   ↓
   Returns: node_id (int)
```

### Example: Running Connected Components

```
import groggy as gr

# Create graph
g = gr.Graph()
g.add_node(name="Alice", age=29)
g.add_node(name="Bob", age=55)
g.add_node(name="Carol", age=31)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)

# Run connected components
components = g.connected_components()
print(components)
```

---

## Performance Characteristics

### Design Goals

| Layer | Optimization Target | Complexity |
|-------|-------------------|------------|
| Rust Core | Algorithm efficiency | O(1) amortized for core ops |
| FFI Bridge | Minimal overhead | <100ns per call |
| Python API | Developer ergonomics | Delegation overhead negligible |

### Bottleneck Analysis

**Where time is spent:**
1. **Rust algorithms**: 90-95% (where it should be)
2. **Type conversion**: 3-5% (unavoidable overhead)
3. **Python delegation**: 1-2% (negligible)
4. **GIL management**: <1% (properly managed)

### Optimization Strategies

**1. Batch Operations**
```python
# Bad: N FFI calls
for node in nodes:
    g.set_attr(node, "value", compute(node))

# Good: 1 FFI call
values = [compute(node) for node in nodes]
g.set_column("value", values)
```

**2. Release GIL**
```rust
// For expensive operations
pub fn expensive_algo(&self, py: Python) -> PyResult<T> {
    py.allow_threads(|| {
        // Rust work here - Python can run in parallel
        self.core_algorithm()
    })
}
```

**3. Zero-Copy Views**
```python
# No data copying
subgraph = g.nodes[:100]  # View into graph
table = subgraph.table()   # View of subgraph
```

---

## Memory Management

### Ownership Model

**Rust Core:**
- Owns all data
- Arc<RwLock<T>> for shared mutable access
- Automatic cleanup via Drop trait

**Python Layer:**
- Holds references to Rust objects
- Python GC manages Python wrappers
- Rust data lives as long as Python references exist

### Lifetime Safety

```rust
// Rust enforces lifetime safety
#[pyclass]
pub struct PyGraph {
    inner: Arc<RwLock<CoreGraph>>,  // Thread-safe reference
}

#[pyclass]
pub struct PySubgraph {
    graph: Arc<RwLock<CoreGraph>>,  // Keeps graph alive
    node_ids: Vec<NodeId>,          // Just IDs, not copies
}
```

The subgraph holds a reference to the graph, so the graph can't be dropped while the subgraph exists.

---

## Key Design Decisions

### 1. FFI Contains Zero Logic

**Why?**
- Easier to maintain (logic in one place)
- Easier to test (test Rust directly)
- Easier to optimize (profile Rust)

**Enforcement:**
- Code reviews enforce this
- FFI functions should be <10 lines
- Any logic moves to Rust core

### 2. Columnar Storage in Core

**Why?**
- Performance is core to identity
- Enables SIMD and parallelization
- Natural fit for analytics

**Trade-off:**
- More complex than row-wise
- But 10-100x faster for bulk ops

### 3. Views Over Copies

**Why?**
- Memory efficiency
- Immutability prevents bugs
- Enables cheap composition

**Implementation:**
- Subgraphs hold node/edge IDs, not data
- Tables are snapshots (cheap to create)
- Arrays reference columns in pool

---

## Next Steps

- **[Connected Views](connected-views.md)**: Master object transformations and delegation
- **[User Guide](../guide/graph-core.md)**: Start building with this architecture
- **[API Reference](../api/graph.md)**: Detailed documentation for each component
