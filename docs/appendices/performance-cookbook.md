# Appendix C: Performance Cookbook

**Practical optimization patterns for Groggy**

This cookbook provides actionable performance guidance based on Groggy's architecture. Each recipe includes the pattern, why it works, and when to use it.

---

## Understanding Groggy's Performance Model

### Core Performance Characteristics

Groggy's performance is based on three architectural decisions:

1. **Columnar Storage** - Attributes stored as separate arrays
2. **Immutable Views** - No copying unless explicitly requested
3. **Rust Core** - High-performance algorithms and data structures

**Key Insight:**
> Bulk operations on columns are 10-100x faster than iteration over individual items

---

## Recipe 1: Batch Operations Over Loops

### ❌ Anti-Pattern: Item-by-Item Iteration

```python
# SLOW: Python loop, FFI call per node
for node_id in g.node_ids():
    age = g.nodes[node_id]["age"]  # FFI call
    if age < 30:
        young_nodes.append(node_id)
```

**Why it's slow:**
- Python loop overhead
- FFI crossing per iteration
- No SIMD optimization

### ✅ Best Practice: Columnar Operations

```python
# FAST: Single columnar filter
young = g.nodes[g.nodes["age"] < 30]
```

**Why it's fast:**
- Single FFI crossing
- Bulk columnar scan (cache-friendly)
- SIMD-optimized filtering
- Returns view (no copy)

**Performance:** ~100x faster for large graphs

### When to Use

- ✅ Filtering nodes/edges by attributes
- ✅ Aggregating statistics
- ✅ Transforming attribute values
- ❌ Complex per-node logic that can't be vectorized

---

## Recipe 2: Minimize FFI Crossings

### Understanding FFI Overhead

Each call from Python to Rust has overhead (~50-100ns). For tight loops, this adds up.

### ❌ Anti-Pattern: Frequent Small Calls

```python
# SLOW: Many small FFI calls
total = 0
for node_id in g.node_ids():  # FFI call
    degree = g.degree(node_id)  # FFI call per node
    total += degree
```

### ✅ Best Practice: Bulk Retrieval

```python
# FAST: Single bulk operation
degrees = g.degrees()  # One FFI call, returns NumArray
total = degrees.sum()  # Computed in Rust
```

**Performance:** O(n) FFI calls → O(1) FFI call

### Strategy

1. **Identify the operation** you need
2. **Check if bulk version exists** (often does)
3. **Retrieve all at once**, process in Rust
4. **Return result** to Python

### Common Bulk Operations

```python
# Instead of loops, use bulk operations:
g.degrees()              # All degrees at once
g.node_ids()             # All node IDs
g.nodes["attribute"]     # Entire attribute column
g.table()                # All data as table
```

---

## Recipe 3: Use Views, Materialize Only When Needed

### Understanding Views

Most Groggy operations return **views** (lightweight references), not copies.

### ✅ Views Are Cheap

```python
# All O(1) - no data copying:
subgraph = g.nodes[g.nodes["age"] < 30]     # View
table = g.table()                            # View
array = g["age"]                             # View
```

### ⚠️ Materialization Is Expensive

```python
# These create new data structures:
df = g.nodes.table().to_pandas()    # Materializes to DataFrame
g_copy = subgraph.to_graph()        # Copies to new Graph
np_array = matrix.to_numpy()        # Converts to NumPy array
```

### Best Practice: Delay Materialization

```python
# Good: Chain views, materialize at end
result = (g.nodes[g.nodes["age"] < 30]    # View
           .table()                        # View
           .select(["name", "age"])        # View
           .to_pandas())                   # Materialize once

# Bad: Materialize early
df = g.nodes.table().to_pandas()           # Materialize
young_df = df[df["age"] < 30]              # Python filtering
result_df = young_df[["name", "age"]]      # Python selection
```

**Performance Impact:**
- Views: O(1) time, O(1) memory
- Materialization: O(n) time, O(n) memory

### When to Materialize

- ✅ Final output for external libraries (pandas, numpy)
- ✅ When you need a persistent copy
- ✅ For complex Python-side processing
- ❌ In the middle of a chain

---

## Recipe 4: Leverage Columnar Filtering

### How Columnar Filtering Works

Groggy scans attribute columns using SIMD-optimized operations.

### ✅ Fast Attribute Filters

```python
# FAST: Columnar scan with SIMD
active_users = g.nodes[g.nodes["active"] == True]
heavy_edges = g.edges[g.edges["weight"] > 5.0]
young_adults = g.nodes[
    (g.nodes["age"] >= 18) & (g.nodes["age"] < 30)
]
```

**Complexity:** O(n) with SIMD optimization, cache-friendly

### ⚠️ Slower Topology Filters

```python
# SLOWER: Topology traversal
two_hop_neighbors = g.neighbors(node, depth=2)
```

**Complexity:** O(degree^depth)

### Best Practice: Filter Attributes First

```python
# Good: Filter 100k nodes to 1k, then traverse
young = g.nodes[g.nodes["age"] < 25]  # Fast columnar filter
neighbors = young.neighbors(depth=2)   # Smaller graph to traverse

# Bad: Traverse first, then filter
all_neighbors = g.neighbors(some_node, depth=2)  # Large graph
young_neighbors = all_neighbors[all_neighbors["age"] < 25]
```

**Why it's better:**
- Columnar filter is O(n) but cache-friendly
- Reduces graph size before expensive traversal
- SIMD optimization on attributes

---

## Recipe 5: Choose the Right Data Structure

### Graph vs Table vs Array vs Matrix

Each structure is optimized for different operations:

```python
# Graph - topology operations:
g.neighbors(node)                # Fast: adjacency list lookup
g.add_edge(n1, n2)              # Fast: O(1) amortized

# Table - columnar queries:
g.nodes.table().select(["age", "name"])  # Fast: column selection
df = g.table().to_pandas()               # Fast: bulk conversion

# Array - numeric operations:
ages = g.nodes["age"]            # Fast: direct column access
mean_age = ages.mean()           # Fast: SIMD aggregation

# Matrix - linear algebra:
A = g.adjacency_matrix()         # Fast: CSR sparse matrix
eigvals = A.eigenvalues()        # Fast: optimized linalg
```

### Performance by Operation

| Operation | Best Structure | Complexity | Notes |
|-----------|---------------|------------|-------|
| Get neighbors | Graph | O(degree) | Adjacency list |
| Filter by attribute | Graph → Subgraph | O(n) | SIMD columnar |
| Aggregate statistics | Array | O(n) | SIMD optimized |
| Column selection | Table | O(1) | View creation |
| Matrix operations | Matrix | Varies | Sparse-optimized |
| Export to pandas | Table | O(n) | Bulk conversion |

### Recipe: Choose Based on Next Operation

```python
# If filtering → use Graph/Subgraph:
subset = g.nodes[condition]

# If aggregating → use Array:
ages = g["age"]
stats = {"mean": ages.mean(), "std": ages.std()}

# If exporting → use Table:
df = g.table().to_pandas()

# If linear algebra → use Matrix:
L = g.laplacian_matrix()
embedding = L.eigenvalues()[:10]
```

---

## Recipe 6: Attribute Sparsity Optimization

### Understanding Sparse Attributes

Not all nodes have all attributes. Groggy handles this efficiently.

### ✅ Sparse Storage Is Free

```python
# Only 10% of nodes have "email" attribute - that's fine:
g.add_node(id=0, name="Alice", email="alice@example.com")
g.add_node(id=1, name="Bob")  # No email
g.add_node(id=2, name="Carol")  # No email

# Sparse attributes don't waste memory:
emails = g["email"]  # Only stores 1 value, not 3
```

**Memory:** O(non-null values), not O(total nodes)

### ⚠️ Avoid Dense Computation on Sparse Data

```python
# Bad: Materializes all values (including nulls):
df = g.nodes.table().to_pandas()
email_df = df[["email"]]  # Creates full column with NaN

# Good: Filter first:
has_email = g.nodes[g.nodes["email"].is_not_null()]
emails = has_email["email"].to_list()  # Only non-null values
```

### Best Practice: Query Before Materializing

```python
# Check sparsity first:
total_nodes = g.node_count()
has_attr = g.nodes[g.nodes["attr"].is_not_null()].node_count()
sparsity = has_attr / total_nodes

if sparsity < 0.1:
    # Sparse - filter first:
    subset = g.nodes[g.nodes["attr"].is_not_null()]
    values = subset["attr"]
else:
    # Dense - direct access:
    values = g["attr"]
```

---

## Recipe 7: Efficient Subgraph Operations

### Subgraphs Are Views

Creating a subgraph is O(1) - it's just a view.

### ✅ Chain Subgraph Operations

```python
# Fast: All views, no copying
result = (g.nodes[g.nodes["age"] < 30]           # View
          .edges[g.edges["weight"] > 5.0]        # View on view
          .connected_components()                 # Algorithm
          .largest()                              # View
          .table())                               # View
```

**Total copies:** 0 (until `to_pandas()` or similar)

### ⚠️ Converting to Graph Copies Data

```python
# Expensive: Creates new graph
subgraph = g.nodes[condition]
new_graph = subgraph.to_graph()  # O(n + m) copy
```

**Only convert when:**
- Need to modify the subgraph
- Persisting for later use
- Passing to external library

### Best Practice: Stay in Subgraph

```python
# Good: Work with subgraph directly
subgraph = g.nodes[g.nodes["active"] == True]
components = subgraph.connected_components()
largest = components.largest()
stats = largest.table().agg({"degree": "mean"})

# Bad: Convert to graph unnecessarily
subgraph = g.nodes[g.nodes["active"] == True]
new_g = subgraph.to_graph()  # Expensive copy
components = new_g.connected_components()
# ... rest of operations
```

---

## Recipe 8: Matrix Operation Optimization

### Sparse vs Dense Matrices

Groggy automatically chooses the right format based on graph density.

### Understanding Sparsity

```python
# Check sparsity:
n = g.node_count()
m = g.edge_count()
density = m / (n * n)

# Sparse: density < 0.01 (1%)
# Dense: density > 0.1 (10%)
```

### ✅ Sparse Matrix Operations

```python
# For sparse graphs (most real networks):
A = g.adjacency_matrix()  # Returns sparse CSR matrix

# These are optimized for sparse:
degrees = A.sum(axis=1)        # O(m) not O(n²)
product = A @ A                # Only non-zero entries
eigvals = A.eigenvalues(k=10)  # Sparse eigensolver
```

### ⚠️ When to Densify

```python
# Only densify if:
# 1. Matrix is already dense (density > 0.1)
# 2. Algorithm requires dense format
# 3. Performance testing shows benefit

# Explicit densification:
A_dense = A.to_dense()  # Memory: O(n²) vs O(m)
```

### Best Practice: Check Sparsity First

```python
# Adaptive approach:
A = g.adjacency_matrix()

if A.density() < 0.01:
    # Sparse operations:
    result = A.sparse_operation()
else:
    # Dense operations:
    A_dense = A.to_dense()
    result = A_dense.dense_operation()
```

---

## Recipe 9: Memory Management

### Understanding Memory Usage

Groggy's memory footprint:

```
Total Memory = Structure + Attributes + Views + History

Structure: O(n + m) - nodes and edges
Attributes: O(total non-null values)
Views: O(view metadata) - negligible
History: O(states × changes)
```

### ✅ Minimize History Overhead

```python
# If you don't need history:
g = gr.Graph(track_history=False)  # Saves memory

# Or clear history periodically:
g.clear_history()  # Frees old states
g.commit()         # Start fresh
```

### ✅ Efficient Bulk Loading

```python
# Good: Batch loading
nodes = [{"id": i, "name": f"Node{i}"} for i in range(10000)]
g.add_nodes(nodes)  # Single bulk operation

# Bad: One at a time
for i in range(10000):
    g.add_node(name=f"Node{i}")  # Repeated overhead
```

**Performance:** Bulk is ~10x faster

### ✅ Release Large Objects

```python
# After expensive operation:
large_matrix = g.adjacency_matrix()
result = compute_something(large_matrix)

# Release when done:
del large_matrix  # Frees memory
```

### Monitor Memory

```python
import psutil
import os

# Check memory usage:
process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024**2:.1f} MB")

# Profile operations:
mem_before = process.memory_info().rss
result = expensive_operation()
mem_after = process.memory_info().rss
print(f"Operation used: {(mem_after - mem_before) / 1024**2:.1f} MB")
```

---

## Recipe 10: Algorithm Selection

### Choose the Right Algorithm

Different algorithms have different complexity for the same task.

### Shortest Paths

```python
# Unweighted graph → BFS:
path = g.shortest_path(source, target)  # O(n + m)

# Weighted graph → Dijkstra:
path = g.shortest_path(source, target, weighted=True)  # O((n + m) log n)

# All pairs → Floyd-Warshall (small graphs only):
all_paths = g.all_pairs_shortest_paths()  # O(n³)
```

### Connected Components

```python
# Undirected → Union-Find:
components = g.connected_components()  # O(n + m)

# Directed → Tarjan's:
strong_components = g.strongly_connected_components()  # O(n + m)
```

### Centrality Measures

```python
# Exact centrality (small graphs):
bc = g.betweenness_centrality()  # O(nm)

# Approximate centrality (large graphs):
bc_approx = g.betweenness_centrality(approximate=True, samples=100)  # O(sm)
```

### Rule of Thumb

| Graph Size | Algorithm Choice |
|------------|------------------|
| n < 1,000 | Exact algorithms fine |
| 1,000 < n < 100,000 | Efficient exact algorithms |
| n > 100,000 | Approximate or sampling-based |

---

## Recipe 11: Parallel Processing

### When Groggy Uses Parallelism

Groggy automatically parallelizes certain operations:

```python
# These use multiple cores:
g.connected_components()  # Parallel component finding
A.eigenvalues()          # Parallel linear algebra
g.spectral_embedding()   # Parallel SVD
```

### ✅ Batch Independent Operations

```python
# Process multiple subgraphs in parallel:
from concurrent.futures import ProcessPoolExecutor

def process_component(subgraph):
    return subgraph.table().agg({"weight": "mean"})

components = g.connected_components()

with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_component, components))
```

### ⚠️ GIL Considerations

Python's GIL limits pure Python parallelism, but Groggy's Rust core releases it:

```python
# Good: Rust operations release GIL
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(g.connected_components)
        for g in graphs
    ]
    results = [f.result() for f in futures]

# Bad: Python operations hold GIL
# (use ProcessPoolExecutor instead)
```

---

## Performance Checklist

### Before Optimization

- [ ] **Profile first** - measure, don't guess
- [ ] **Identify bottleneck** - where is time spent?
- [ ] **Check if it matters** - is this code on critical path?

### Optimization Strategies

1. **Use bulk operations** instead of loops
2. **Minimize FFI crossings** - batch calls
3. **Delay materialization** - work with views
4. **Filter attributes first** - reduce data size
5. **Choose right structure** - graph/table/array/matrix
6. **Leverage sparsity** - don't materialize nulls
7. **Chain subgraph ops** - avoid unnecessary copies
8. **Use sparse matrices** - for low-density graphs
9. **Clear history** - if not needed
10. **Pick right algorithm** - complexity matters

### Common Pitfalls

| Anti-Pattern | Better Approach | Speedup |
|--------------|-----------------|---------|
| Python loops | Bulk operations | ~100x |
| Many small FFI calls | Single bulk call | ~10x |
| Early materialization | Delay to end of chain | ~5x |
| Dense on sparse | Keep sparse | ~10x memory |
| Full graph traversal | Filter first | Varies |

---

## Profiling Tools

### Timing Operations

```python
import time

start = time.perf_counter()
result = expensive_operation()
elapsed = time.perf_counter() - start
print(f"Operation took: {elapsed:.3f}s")
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def my_function():
    g = gr.Graph()
    # ... operations
    return g.table()
```

### Visual Profiling

```python
import cProfile
import pstats

# Profile code:
cProfile.run('expensive_operation()', 'profile_stats')

# Analyze:
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

---

## Performance By Graph Size

### Small Graphs (n < 1,000)

- ✅ All algorithms work fine
- ✅ Dense operations acceptable
- ✅ No special optimization needed
- Focus on code clarity

### Medium Graphs (1,000 < n < 100,000)

- ✅ Use columnar operations
- ✅ Sparse matrices for low density
- ✅ Efficient algorithms (Dijkstra not Floyd-Warshall)
- Watch memory usage

### Large Graphs (n > 100,000)

- ✅ Bulk operations essential
- ✅ Sparse matrices mandatory
- ✅ Approximate algorithms when possible
- ✅ Consider sampling for analysis
- Profile everything

### Very Large Graphs (n > 1,000,000)

- ✅ Streaming or batch processing
- ✅ Sample-based statistics
- ✅ Distributed processing if needed
- Consider if Groggy is right tool

---

## See Also

- **[Performance Guide](../guide/performance.md)** - Detailed performance tutorial
- **[Architecture](../concepts/architecture.md)** - Why these patterns work
- **[Design Decisions](design-decisions.md)** - Rationale for performance choices
- **[Glossary](glossary.md)** - Performance-related terms
