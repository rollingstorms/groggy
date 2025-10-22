# Performance Optimization

Groggy is designed for high performance with a Rust core, but optimal performance requires understanding the library's design patterns. This guide covers benchmarking, optimization techniques, and scaling strategies.

---

## Performance Philosophy

Groggy's performance comes from:

1. **Rust core**: Zero-cost abstractions, memory safety without garbage collection
2. **Columnar storage**: Cache-friendly data layout for bulk operations
3. **Lazy evaluation**: Views and deferred computation where beneficial
4. **Parallel algorithms**: Multi-threaded execution for large graphs
5. **Attribute-first design**: Fast filtering and queries on node/edge attributes
6. **Explicit trait-backed delegation**: Direct PyO3 methods (100ns FFI budget) instead of dynamic lookups

!!! info "Trait-Backed Delegation (v0.5.1+)"
    Starting in v0.5.1, Groggy uses explicit PyO3 methods backed by Rust traits rather than dynamic `__getattr__` delegation. This provides:
    
    - **20x faster method calls**: ~100ns per FFI call vs ~2000ns for dynamic lookup
    - **Better IDE support**: All methods discoverable via autocomplete
    - **Clear stack traces**: Direct method names in debugging
    
    See [Trait-Backed Delegation](../concepts/trait-delegation.md) for architectural details.

---

## Benchmarking

### Basic Timing

Measure operation performance:

```python
import groggy as gr
import time

g = gr.generators.erdos_renyi(n=10000, p=0.01)

# Time an operation
start = time.perf_counter()
components = g.connected_components()
elapsed = time.perf_counter() - start

print(f"Connected components: {elapsed*1000:.2f} ms")
```

### Using timeit

For more accurate measurements:

```python
import timeit

def benchmark():
    g = gr.generators.erdos_renyi(n=10000, p=0.01)
    return g.connected_components()

# Run multiple times
times = timeit.repeat(benchmark, number=10, repeat=3)
avg_time = min(times) / 10

print(f"Average: {avg_time*1000:.2f} ms")
```

### Memory Profiling

Track memory usage:

```python
import tracemalloc

tracemalloc.start()

g = gr.generators.erdos_renyi(n=100000, p=0.001)
components = g.connected_components()

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024**2:.1f} MB")
print(f"Peak: {peak / 1024**2:.1f} MB")

tracemalloc.stop()
```

---

## Optimization Patterns

### Pattern 1: Bulk Operations

**Avoid:** Loops over individual operations
**Prefer:** Bulk operations

```python
# ❌ Slow: Individual operations
g = gr.Graph()
for i in range(10000):
    g.add_node(id=i, value=i*2)

# ✅ Fast: Bulk operations
nodes_data = [{"id": i, "value": i*2} for i in range(10000)]
g.add_nodes(nodes_data)
```

**Why faster:**
- Single FFI call vs 10,000 FFI calls
- Rust can optimize batch processing
- Better cache locality

### Pattern 2: Attribute-First Filtering

**Avoid:** Fetching all data then filtering in Python
**Prefer:** Push filters to Rust

```python
# ❌ Slow: Filter in Python
all_nodes = g.nodes.table().to_pandas()
young = all_nodes[all_nodes['age'] < 30]

# ✅ Fast: Filter in Rust
young = g.nodes[g.nodes['age'] < 30]
```

**Why faster:**
- Filtering happens in Rust (compiled code)
- Only matching data crosses FFI boundary
- Columnar storage optimized for attribute queries

### Pattern 3: Views vs Materialization

Understand when to materialize:

```python
# Views are free
sub = g.nodes[g.nodes['age'] < 30]  # O(1) view creation

# But repeated access on views can be costly
for _ in range(1000):
    count = sub.node_count()  # Recomputes filter each time

# Materialize for repeated access
sub_graph = sub.to_graph()  # O(n) materialization
for _ in range(1000):
    count = sub_graph.node_count()  # O(1) cached result
```

**Guidelines:**
- Use views for single-pass operations
- Materialize for repeated access
- Profile to find the crossover point

### Pattern 4: Avoid Unnecessary Conversions

```python
# ❌ Slow: Multiple conversions
df = g.nodes.table().to_pandas()
ages = df['age'].tolist()
mean_age = sum(ages) / len(ages)

# ✅ Fast: Use native operations
mean_age = g.nodes['age'].mean()
```

**Why faster:**
- No DataFrame overhead
- No Python list creation
- Computation stays in Rust

### Pattern 5: Sparse vs Dense Matrices

For large graphs, understand matrix representations:

```python
# Large sparse graph
g = gr.generators.erdos_renyi(n=10000, p=0.001)

# Adjacency is sparse (fast)
A = g.adjacency_matrix()  # ~100K edges, not 100M

# Operations respect sparsity
A2 = A.matmul(A)  # Sparse matrix multiply

# Dense conversion only if needed
if A.is_sparse() and some_condition:
    A_dense = A.dense()  # Careful: 400MB for 10K nodes
```

### Pattern 6: Parallel Algorithms

Groggy automatically parallelizes when beneficial:

```python
# Automatically parallel for large graphs
g = gr.generators.erdos_renyi(n=100000, p=0.0001)
components = g.connected_components()  # Uses multiple threads
```

**No configuration needed:**
- Algorithms choose parallelization based on size
- Rust's rayon handles thread pool
- GIL released during computation

### Pattern 7: Incremental Updates

For evolving graphs, use incremental operations:

```python
# ❌ Slow: Rebuild entire graph
for new_edge in stream:
    g_old = g
    g = gr.Graph()
    # Copy all nodes/edges + new edge

# ✅ Fast: Incremental updates
for new_edge in stream:
    g.add_edge(*new_edge)  # O(1) append
```

---

## Scaling Strategies

### Small Graphs (< 1K nodes)

**Characteristics:**
- FFI overhead dominates
- Memory not a concern
- Single-threaded often faster

**Recommendations:**
```python
# Minimize FFI crossings
data = prepare_all_data()  # Python-side prep
g = gr.Graph()
g.add_nodes(data['nodes'])  # Single bulk call
g.add_edges(data['edges'])
```

### Medium Graphs (1K - 100K nodes)

**Characteristics:**
- Computation and memory balanced
- Parallelization starts helping
- Attribute queries very fast

**Recommendations:**
```python
# Use bulk operations
g.connected_components(inplace=True, label='component')

# Leverage columnar storage
high_degree = g.nodes[g.nodes['degree'] > 10]

# Cache computed results
degrees = g.degree()  # Compute once
for analysis in analyses:
    analysis.use_degrees(degrees)
```

### Large Graphs (100K - 10M nodes)

**Characteristics:**
- Memory becomes critical
- Parallel algorithms essential
- Sparse representations required

**Recommendations:**
```python
# Work with subgraphs
core = g.nodes[g.nodes['degree'] > 5]
core_analysis = core.connected_components()

# Stream processing for edges
# (if streaming API available)

# Use sparse matrices
A = g.adjacency_matrix()
assert A.is_sparse()  # Verify sparse
```

### Very Large Graphs (> 10M nodes)

**Characteristics:**
- Cannot fit all in memory
- Must use sampling/partitioning
- Approximate algorithms needed

**Recommendations:**
```python
# Sample for analysis
sample = g.nodes.sample(100000)
sample_graph = sample.to_graph()
metrics = sample_graph.degree()

# Partition by components
components = g.connected_components()
for comp in components[:10]:  # Process largest 10
    analyze(comp.to_graph())

# Use approximate algorithms
# (when available)
# estimate = g.approximate_pagerank()
```

---

## Common Performance Issues

### Issue 1: Python Loops Over Graph Elements

**Problem:**
```python
# Very slow for large graphs
total = 0
for node in g.nodes.ids():
    total += g.nodes[node]['value']
```

**Solution:**
```python
# Much faster
total = g.nodes['value'].sum()
```

**Why:**
- First version: N FFI calls, Python arithmetic
- Second version: 1 FFI call, Rust arithmetic

### Issue 2: Repeated DataFrame Conversions

**Problem:**
```python
for attribute in ['age', 'score', 'rank']:
    df = g.nodes.table().to_pandas()  # Slow conversion each time
    print(df[attribute].mean())
```

**Solution:**
```python
# Convert once
df = g.nodes.table().to_pandas()
for attribute in ['age', 'score', 'rank']:
    print(df[attribute].mean())

# Or better: stay in Groggy
for attribute in ['age', 'score', 'rank']:
    print(g.nodes[attribute].mean())
```

### Issue 3: Unnecessary Materialization

**Problem:**
```python
# Materializes even though we only need count
filtered = g.nodes[g.nodes['age'] < 30].to_graph()
count = filtered.node_count()
```

**Solution:**
```python
# View is sufficient
filtered = g.nodes[g.nodes['age'] < 30]
count = len(filtered)  # or filtered.node_count()
```

### Issue 4: Inefficient Attribute Access

**Problem:**
```python
# Access pattern unfriendly to columnar storage
for node in g.nodes.ids():
    process(
        g.nodes[node]['age'],
        g.nodes[node]['score'],
        g.nodes[node]['rank']
    )
```

**Solution:**
```python
# Columnar access
ages = g.nodes['age'].to_list()
scores = g.nodes['score'].to_list()
ranks = g.nodes['rank'].to_list()

for age, score, rank in zip(ages, scores, ranks):
    process(age, score, rank)

# Or even better: bulk operation
result = process_bulk(
    g.nodes['age'],
    g.nodes['score'],
    g.nodes['rank']
)
```

---

## Profiling and Debugging

### Finding Bottlenecks

Use Python profilers:

```python
import cProfile
import pstats

def analysis():
    g = gr.generators.karate_club()
    g.connected_components(inplace=True)
    return g.nodes.table().to_pandas()

# Profile
profiler = cProfile.Profile()
profiler.enable()
result = analysis()
profiler.disable()

# View results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Rust-Level Profiling

For core Rust performance:

```bash
# Build with profiling
cargo build --release --features profiling

# Use system profilers
# macOS: Instruments
# Linux: perf, valgrind
```

### Memory Leaks

Check for memory growth:

```python
import gc

for i in range(100):
    g = gr.generators.erdos_renyi(n=10000, p=0.01)
    components = g.connected_components()

    if i % 10 == 0:
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        print(f"Iteration {i}: {current / 1024**2:.1f} MB")
```

---

## Performance Checklist

### Before Optimization

- [ ] Profile to find actual bottleneck
- [ ] Measure baseline performance
- [ ] Identify the slowest 20% of code
- [ ] Understand data size and growth

### Graph Construction

- [ ] Use bulk `add_nodes()` and `add_edges()`
- [ ] Prepare data in Python, send in batches
- [ ] Avoid repeated small additions
- [ ] Set attributes during creation when possible

### Graph Queries

- [ ] Use attribute-based filtering
- [ ] Keep filters in Rust (not Python)
- [ ] Materialize only when needed
- [ ] Cache frequently accessed results

### Algorithms

- [ ] Let Groggy parallelize automatically
- [ ] Work on subgraphs when appropriate
- [ ] Use in-place operations when available
- [ ] Batch algorithm runs when possible

### Data Export

- [ ] Convert to DataFrame once, not repeatedly
- [ ] Use appropriate format (CSV, Parquet, Bundle)
- [ ] Export only needed attributes
- [ ] Stream large exports if available

---

## Real-World Examples

### Example 1: Social Network Analysis

```python
# Load large social network
g = gr.GraphTable.load_bundle("social_network.bundle")

# ✅ Efficient analysis
start = time.time()

# Filter in Rust
active_users = g.nodes[g.nodes['last_active'] > cutoff_date]

# Compute in bulk
degrees = active_users.degree()

# Materialize for multi-use
active_graph = active_users.to_graph()
components = active_graph.connected_components()

# Export efficiently
results = {
    'degrees': degrees.to_list(),
    'num_components': len(components)
}

print(f"Analysis: {time.time() - start:.2f}s")
```

### Example 2: Real-Time Updates

```python
# Streaming graph updates
g = gr.Graph()

batch = []
for event in event_stream:
    batch.append(event)

    # Batch updates
    if len(batch) >= 1000:
        edges = [(e['src'], e['tgt'], e['attrs']) for e in batch]
        g.add_edges(edges)
        batch.clear()

        # Periodic analysis
        if g.edge_count() % 10000 == 0:
            metrics = compute_metrics(g)
            emit_metrics(metrics)
```

### Example 3: Large Graph Sampling

```python
# Too large to process fully
g = gr.GraphTable.load_bundle("huge_graph.bundle")

# Sample and analyze
sample_size = 10000
sample = g.nodes.sample(sample_size)
sample_graph = sample.to_graph()

# Fast analysis on sample
degrees = sample_graph.degree()
clustering = sample_graph.clustering_coefficient()

# Extrapolate to full graph
estimated_avg_degree = degrees.mean() * (g.node_count() / sample_size)
```

---

## Performance Goals

Groggy targets these performance characteristics:

| Operation | Target Complexity | Notes |
|-----------|------------------|-------|
| Add node | O(1) amortized | Append to columnar storage |
| Add edge | O(1) amortized | Append to edge list |
| Attribute query | O(n) | Full column scan |
| Attribute filter | O(n) | Vectorized in Rust |
| BFS/DFS | O(V + E) | Parallel for large graphs |
| Connected components | O(V + E) | Union-find with path compression |
| Degree computation | O(E) | Linear in edge count |
| Subgraph creation (view) | O(1) | Lazy view |
| Subgraph materialization | O(V + E) | Copy induced subgraph |

---

## Quick Reference

### Fast Operations

```python
# These are optimized
g.add_nodes(data)                    # Bulk insert
g.nodes[g.nodes['age'] > 30]        # Attribute filter
g.degree()                           # Degree computation
g.connected_components()             # Parallel algorithm
A = g.adjacency_matrix()             # Sparse matrix
g.nodes['age'].mean()                # Columnar stats
```

### Slow Patterns to Avoid

```python
# Avoid these
for node in g.nodes.ids():           # Python loop
    g.nodes[node]['value']           # Individual access

df = g.to_pandas()                   # Repeated conversions
for analysis in analyses:
    df = g.to_pandas()

sub.to_graph()                       # Unnecessary materialization
if sub.node_count() > 10:           # When view suffices
```

### When to Materialize

```python
# Keep as view
sub = g.nodes[filter]
count = len(sub)                     # One-time use

# Materialize
sub = g.nodes[filter]
sub_g = sub.to_graph()
for _ in range(100):                 # Repeated access
    analysis(sub_g)
```

---

## See Also

- **[Graph Core Guide](graph-core.md)**: Understanding core operations
- **[Algorithms Guide](algorithms.md)**: Algorithm complexity reference
- **[Arrays Guide](arrays.md)**: Columnar data operations
- **[Matrices Guide](matrices.md)**: Sparse vs dense matrices
- **[Integration Guide](integration.md)**: Efficient data exchange
- **[Architecture Docs](../concepts/architecture.md)**: System design
