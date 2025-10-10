# Appendix B: Design Decisions (ADRs)

**Architectural Decision Records for Groggy**

This document captures key architectural decisions made during Groggy's development, including the rationale, alternatives considered, and consequences.

---

## ADR-001: Three-Tier Architecture

**Status:** Accepted
**Date:** Early development
**Context:** Need to balance Python usability with high-performance graph operations

### Decision

Implement a three-tier architecture:

1. **Rust Core** - High-performance algorithms, storage, state management
2. **FFI Bridge (PyO3)** - Pure translation layer with no business logic
3. **Python API** - User-facing interface with delegation and chaining

### Rationale

**Why this approach:**
- Rust provides memory safety and performance for core operations
- Python provides usability and ecosystem integration
- Clear separation prevents logic duplication
- FFI layer stays thin and maintainable

**Alternatives considered:**
- **Pure Python:** Too slow for large graphs
- **Python with NumPy:** Still bottlenecked on graph operations
- **Cython:** Less memory safe, harder to maintain than Rust
- **Logic in FFI:** Would create duplication and maintenance burden

### Consequences

**Positive:**
- ✅ Excellent performance for core operations
- ✅ Memory-safe implementation
- ✅ Python's ease of use maintained
- ✅ Clear architectural boundaries

**Negative:**
- ⚠️ FFI overhead for frequent small operations
- ⚠️ Two languages to maintain
- ⚠️ Build complexity (Rust + Python toolchain)

**Mitigation:**
- Batch operations to minimize FFI crossings
- Keep FFI interface minimal and well-documented
- Use maturin for simplified builds

---

## ADR-002: Separation of Structure and Attributes

**Status:** Accepted
**Date:** Ultralight example phase
**Context:** Need to support both graph topology operations and attribute-based queries efficiently

### Decision

Store graph structure (nodes, edges) completely separately from attributes (node/edge data):

```
Graph Structure     Attribute Data
(Topology)          (Signal)
    │                   │
    ├─ GraphSpace  ←→  GraphPool
    │  (which alive)    (columnar attrs)
    │
    └─ HistoryForest
       (versions)
```

### Rationale

**Why separate storage:**
- Graph topology and data have different access patterns
- Topology queries don't need attribute data
- Attribute queries benefit from columnar storage
- Enables independent optimization of each

**Key insight from ultralight example:**
> "Nodes and edges only **point to** attributes; they never store them."

**Alternatives considered:**
- **Row-wise storage:** Nodes/edges contain their attributes directly
  - Poor cache locality for attribute queries
  - Harder to implement columnar ML workflows
- **Hybrid approach:** Some attributes embedded, others separate
  - Complexity without clear benefits

### Consequences

**Positive:**
- ✅ Optimal performance for both graph and attribute operations
- ✅ Natural fit for machine learning (columnar data)
- ✅ Efficient bulk attribute operations
- ✅ Minimal memory overhead

**Negative:**
- ⚠️ Indirection when accessing attributes
- ⚠️ More complex internal implementation

**Impact:**
- Fundamental to Groggy's performance characteristics
- Enables the columnar storage architecture
- Allows graph signal/structure distinction

---

## ADR-003: Columnar Attribute Storage

**Status:** Accepted
**Date:** Ultralight example phase
**Context:** Need efficient attribute operations for ML workflows

### Decision

Store attributes in columnar format where each attribute is a separate array:

```python
# Instead of:
nodes = [
    {"id": 0, "name": "Alice", "age": 29},
    {"id": 1, "name": "Bob", "age": 35}
]

# Use:
node_ids = [0, 1]
names = ["Alice", "Bob"]    # Column
ages = [29, 35]             # Column
```

### Rationale

**Why columnar:**
- Cache-friendly access when querying single attributes
- SIMD optimization opportunities
- Efficient filtering and aggregation
- Natural fit for NumPy/Pandas/Arrow ecosystem
- Minimal memory overhead for sparse attributes

**Alternatives considered:**
- **Row-wise (struct of arrays):** Poor locality for attribute scans
- **Hybrid:** Complexity without proportional benefits
- **Dictionary per node:** Memory overhead, poor cache behavior

### Consequences

**Positive:**
- ✅ 10-100x faster attribute queries vs row-wise
- ✅ Efficient pandas/numpy integration
- ✅ Bulk operations are trivial
- ✅ Sparse attribute support is natural

**Negative:**
- ⚠️ Reconstructing full node/edge data requires joins
- ⚠️ More complex than naive implementation

**Performance impact:**
- Filter by attribute: O(n) columnar scan (cache-friendly)
- Get all attributes of node: O(k) where k = number of attributes
- Aggregate across nodes: SIMD-optimized bulk operation

---

## ADR-004: Immutable Views over Copies

**Status:** Accepted
**Date:** Early API design
**Context:** Need efficient subgraph operations without excessive memory use

### Decision

Return immutable **views** (lightweight references) rather than copying data:

```python
subgraph = g.nodes[g.nodes["age"] < 30]  # View, not copy
table = g.table()                         # View, not copy
array = g["attribute"]                    # View, not copy
```

### Rationale

**Why views:**
- Avoid expensive data copying
- Enable lazy evaluation
- Chain operations efficiently
- Memory efficient for large graphs

**Alternatives considered:**
- **Eager copying:** Simple but wasteful
- **Copy-on-write:** Complex, still requires copying
- **Explicit copy flag:** Burdens user with decisions

### Consequences

**Positive:**
- ✅ O(1) subgraph creation (vs O(n+m) for copying)
- ✅ Constant memory overhead for views
- ✅ Chaining operations is cheap
- ✅ User doesn't think about copying

**Negative:**
- ⚠️ Views become invalid if parent graph changes
- ⚠️ Requires clear documentation of lifetime

**API Impact:**
- Most operations return views
- Explicit `to_graph()` when copy needed
- `to_pandas()` materializes data

---

## ADR-005: Delegation-Based Chaining

**Status:** Accepted
**Date:** API design phase
**Context:** Need expressive, composable API without method explosion

### Decision

Objects delegate methods to their natural transformations, enabling chaining:

```python
result = (g.connected_components()    # → SubgraphArray
          .sample(5)                  # → SubgraphArray
          .neighborhood(depth=2)      # → SubgraphArray
          .table()                    # → GraphTable
          .agg({"weight": "mean"}))   # → AggregationResult
```

### Rationale

**Why delegation:**
- Each object has focused responsibility
- Methods live on the "right" object
- Chaining emerges naturally
- Discovery through IDE autocomplete

**Key principle:**
> "Groggy itself is a graph where objects are nodes and methods are edges"

**Alternatives considered:**
- **Monolithic Graph class:** 100+ methods, hard to discover
- **Explicit conversions:** Verbose, breaks flow
- **Operator overloading:** Confusing, non-obvious

### Consequences

**Positive:**
- ✅ Intuitive, expressive API
- ✅ Methods are discoverable
- ✅ Clear object responsibilities
- ✅ Chainable workflows

**Negative:**
- ⚠️ User must understand transformations
- ⚠️ More objects to document

**API Impact:**
- Core to Groggy's user experience
- Documented in "Connected Views" concept
- Each object knows what it can become

---

## ADR-006: FFI Contains No Business Logic

**Status:** Accepted
**Date:** Architecture definition
**Context:** Prevent logic duplication between Rust and Python

### Decision

The FFI layer (PyO3 bindings) is **pure translation only**:

- ✅ Type conversion (Rust ↔ Python)
- ✅ Error handling and conversion
- ✅ GIL management
- ❌ No algorithms
- ❌ No business logic
- ❌ No data transformations

### Rationale

**Why pure translation:**
- Prevents duplication of logic
- Single source of truth (Rust core)
- Easier testing (test Rust, trust FFI)
- Clearer responsibilities

**Principle:**
> "Bridge translates; it doesn't think."

**Alternatives considered:**
- **Python-side optimization:** Creates divergence
- **Logic in FFI:** Hard to test, duplicates code
- **Hybrid approach:** Unclear boundaries

### Consequences

**Positive:**
- ✅ Single implementation of each algorithm
- ✅ FFI stays thin and maintainable
- ✅ Clear where logic lives
- ✅ Easy to reason about

**Negative:**
- ⚠️ Some Python-only features harder to add
- ⚠️ Must go through Rust for new algorithms

**Development impact:**
- New features: implement in Rust, expose via FFI
- Bug fixes: fix in Rust, FFI just translates
- Testing: focus on Rust core

---

## ADR-007: Git-Like Version Control (HistoryForest)

**Status:** Accepted
**Date:** Core design phase
**Context:** Need to support temporal queries and state management

### Decision

Implement git-like version control system for graphs:

- States identified by `StateId`
- Named branches for parallel development
- Commit/checkout operations
- Time-travel queries

### Rationale

**Why version control:**
- Temporal network analysis requires history
- Branching enables "what-if" scenarios
- Familiar mental model (git)
- Enables reproducible analysis

**Alternatives considered:**
- **Snapshot copying:** Memory expensive
- **Event sourcing:** Complex to query
- **No versioning:** Limits temporal analysis

### Consequences

**Positive:**
- ✅ Temporal queries and analysis
- ✅ Reproducible research workflows
- ✅ Branching for experimentation
- ✅ Familiar git-like interface

**Negative:**
- ⚠️ Memory overhead for history
- ⚠️ Complexity in implementation
- ⚠️ Learning curve for users

**Impact:**
- Enables unique temporal analysis features
- Must document history management
- Memory management considerations

---

## ADR-008: Attribute-First Optimization

**Status:** Accepted
**Date:** Performance optimization phase
**Context:** Most queries filter by attributes, not topology

### Decision

Optimize for attribute-based queries:

```python
# This should be fast:
young = g.nodes[g.nodes["age"] < 30]

# Not just this:
neighbors = g.neighbors(node_id)
```

### Rationale

**Why attribute-first:**
- Real-world queries often filter by properties
- ML workflows need fast attribute access
- Columnar storage enables this
- Topology queries can still be fast

**Observation:**
> "Users query 'nodes with age < 30' more than 'nodes 3 hops away'"

**Alternatives considered:**
- **Topology-first:** Traditional graph databases
- **Balanced:** Neither optimized specifically
- **Index-heavy:** Memory overhead

### Consequences

**Positive:**
- ✅ Fast attribute filtering
- ✅ Efficient ML feature extraction
- ✅ Good pandas integration
- ✅ Bulk operations optimized

**Negative:**
- ⚠️ Some topology queries less optimized
- ⚠️ Assumes attribute-heavy use cases

**Performance characteristics:**
- Attribute filter: O(n) columnar scan (SIMD)
- Get neighbors: O(degree) with index
- Multi-hop: Standard graph complexity

---

## ADR-009: Python API as Primary Interface

**Status:** Accepted
**Date:** Project inception
**Context:** Choose primary user-facing language

### Decision

Python is the primary interface, Rust is the implementation:

- Public API is Python
- Documentation targets Python users
- Examples and tutorials in Python
- Rust core is internal implementation detail

### Rationale

**Why Python:**
- Largest data science ecosystem
- Easy to learn and use
- Rich integration options (pandas, numpy, etc.)
- Interactive workflows (Jupyter)

**Alternatives considered:**
- **Rust as primary:** Smaller audience, harder to use
- **Both equal:** Fragmented effort
- **CLI-first:** Less discoverable, harder to script

### Consequences

**Positive:**
- ✅ Accessible to data scientists
- ✅ Rich ecosystem integration
- ✅ Interactive analysis workflows
- ✅ Easy prototyping

**Negative:**
- ⚠️ Rust experts can't use directly
- ⚠️ Python overhead for some operations

**Impact:**
- All documentation in Python
- Performance-critical users need to learn Python
- Rust API is internal and unstable

---

## ADR-010: Method Naming: Explicit Over Implicit

**Status:** Accepted
**Date:** API design
**Context:** Choose naming conventions for clarity

### Decision

Use explicit, descriptive method names:

```python
# Preferred:
g.connected_components()
g.to_pandas()
g.laplacian_matrix()

# Not:
g.cc()           # Abbreviation unclear
g.df()           # What kind of dataframe?
g.laplacian()    # Laplacian what?
```

### Rationale

**Why explicit:**
- Self-documenting code
- IDE autocomplete more helpful
- Less ambiguity
- New users can guess correctly

**Alternatives considered:**
- **Short names:** Faster to type, harder to remember
- **Very long names:** Too verbose
- **Mixed approach:** Inconsistent

### Consequences

**Positive:**
- ✅ Clear, self-documenting API
- ✅ Easy to discover features
- ✅ Reduces need for documentation lookup
- ✅ Consistent across library

**Negative:**
- ⚠️ More typing (mitigated by autocomplete)
- ⚠️ Longer names in chains

**Guidelines:**
- Full words over abbreviations
- Specific over general (`to_pandas()` not `to_df()`)
- Action verbs for operations (`compute()`, `calculate()`)

---

## ADR-011: Builders Are Lowercase Functions

**Status:** Accepted
**Date:** API design
**Context:** Distinguish constructors from classes

### Decision

Builder/constructor functions use lowercase:

```python
import groggy as gr

# Builders (lowercase):
g = gr.graph()
arr = gr.array([1, 2, 3])
mat = gr.matrix([[1, 0], [0, 1]])

# Classes (TitleCase):
from groggy import Graph
g = Graph()  # Also valid
```

### Rationale

**Why lowercase builders:**
- Follows NumPy convention (`np.array()`)
- Clear distinction from classes
- More natural in chains
- Consistent with Python ecosystem

**Alternatives considered:**
- **TitleCase only:** Less friendly for quick construction
- **All lowercase:** Confuses classes and functions
- **Mixed by type:** Inconsistent

### Consequences

**Positive:**
- ✅ Familiar to NumPy/SciPy users
- ✅ Natural in interactive use
- ✅ Clear constructor vs class distinction

**Negative:**
- ⚠️ Two ways to create objects (builder vs class)
- ⚠️ Must document both patterns

**Usage:**
- Interactive/scripting: use builders (`gr.graph()`)
- Library code: use classes (`Graph()`)
- Both are equivalent

---

## Decision Summary Table

| ADR | Decision | Status | Impact |
|-----|----------|--------|--------|
| 001 | Three-Tier Architecture | Accepted | High - Fundamental structure |
| 002 | Separate Structure & Attributes | Accepted | High - Core data model |
| 003 | Columnar Storage | Accepted | High - Performance critical |
| 004 | Immutable Views | Accepted | High - Memory & API |
| 005 | Delegation Chaining | Accepted | High - User experience |
| 006 | FFI Pure Translation | Accepted | Medium - Maintainability |
| 007 | Git-Like Versioning | Accepted | Medium - Advanced features |
| 008 | Attribute-First Optimization | Accepted | High - Performance model |
| 009 | Python Primary Interface | Accepted | High - User audience |
| 010 | Explicit Method Names | Accepted | Medium - API clarity |
| 011 | Lowercase Builders | Accepted | Low - Convenience |

---

## Design Principles

These decisions reflect core principles:

1. **Performance First, Usability Close Second**
   - Rust for speed, Python for ease

2. **Separation of Concerns**
   - Structure vs attributes, FFI vs logic, views vs copies

3. **Columnar Thinking**
   - Optimize for bulk operations over single-item

4. **Chainable Workflows**
   - Objects transform naturally via delegation

5. **Explicit Over Implicit**
   - Clear names, clear transformations, clear behavior

---

## See Also

- **[Architecture](../concepts/architecture.md)** - How these decisions manifest in the system
- **[Origins](../concepts/origins.md)** - Historical context and ultralight example
- **[Performance Cookbook](performance-cookbook.md)** - Practical implications of these decisions
- **[Glossary](glossary.md)** - Terms used in these decisions
