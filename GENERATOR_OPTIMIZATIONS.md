# Graph Generator Optimizations

## Summary

Optimized graph generator methods to use bulk operations for building graphs efficiently. This enables generation of graphs with 100k+ nodes in seconds rather than minutes or hours.

## Performance Improvements

### Key Speedups (n=5000)
- **watts_strogatz**: >120s (timeout) → 0.019s = **6,300x faster**
- **barabasi_albert**: 0.582s → 0.022s = **26x faster**
- Both generators now scale to 100k+ nodes efficiently

### Benchmark Results (100k nodes)
```
Generator                    Nodes    Edges      Time    Edges/sec
----------------------------------------------------------------
barabasi_albert(100k, m=3)  100,000   299,994   3.43s    87,575
watts_strogatz(100k, k=6)   100,000   300,000   3.33s    90,032
cycle_graph(100k)           100,000   100,000   2.98s    33,511
```

## Technical Changes

### 1. Watts-Strogatz Generator
**Problem**: O(n²) complexity due to checking all existing edges during rewiring
```python
# Before: O(n²) - builds list by filtering all edges for each rewire
possible_targets = [x for x in range(n) 
                   if x != i and x not in [edge[1] for edge in rewired_edges if edge[0] == i]]
```

**Solution**: O(nk) complexity using set-based tracking
```python
# After: O(nk) - track targets per node in sets
node_targets = [set() for _ in range(n)]
# Direct set membership check: O(1)
if new_target != i and new_target not in node_targets[i]:
    node_targets[i].add(new_target)
```

### 2. Barabási-Albert Generator
**Problem**: O(n²m) complexity due to O(n) weighted sampling per attachment
```python
# Before: O(n) per sample - iterate through all nodes
rand_val = random.randint(0, total_degree - 1)
cumsum = 0
for j, degree in enumerate(degrees):  # O(n) iteration
    cumsum += degree
    if rand_val < cumsum:
        target = j
```

**Solution**: O(nm) using repeated-nodes trick
```python
# After: O(1) per sample - direct random choice from list
# Each node appears in list proportional to its degree
repeated_nodes.append(target_idx)  # Target gains one appearance
target = random.choice(repeated_nodes)  # O(1) uniform sampling
```

### 3. Erdős-Rényi Generator
**Enhancement**: Smart algorithm selection based on graph density

```python
# Very sparse (p < 0.05, n > 1000): O(E) sampling instead of O(N²)
if p < 0.05 and n > 1000:
    # Sample edges directly - complexity proportional to edges
    # instead of all possible pairs
    
# Dense graphs: Traditional O(N²) iteration
else:
    # Most accurate for dense graphs, acceptable performance
```

### 4. All Generators
**Consistent pattern**: Bulk node/edge creation
```python
# ✅ BULK: Create all nodes at once
node_data = [{"index": i, **node_attrs} for i in range(n)]
nodes = g.add_nodes(node_data)

# ✅ BULK: Collect all edges, then add at once
edge_pairs = [(nodes[i], nodes[j]) for i, j in edge_list]
g.add_edges(edge_pairs)
```

## Architecture Adherence

These optimizations follow the repository's core principles:

1. **Bulk Operations First**: Changed from per-item loops to batch operations
   - `add_nodes(count)` instead of repeated `add_node()`
   - `add_edges(edge_list)` instead of repeated `add_edge()`

2. **No Business Logic in FFI**: All algorithm optimizations in Python layer
   - FFI layer already optimized for bulk attribute setting
   - Rust core provides efficient bulk primitives

3. **Maintain O(1) Amortized Performance**: 
   - Pre-allocate lists when size is known
   - Use set operations for O(1) lookups
   - Avoid nested iterations over growing collections

4. **Columnar/Vectorized Operations**:
   - Bulk node creation with attributes
   - Single bulk edge addition
   - Minimizes FFI boundary crossings

## Testing

All generators validated for correctness:
```bash
$ python -c "from groggy import generators; ..."
✓ erdos_renyi: 100 nodes, 224 edges
✓ barabasi_albert: 100 nodes, 294 edges
✓ watts_strogatz: 100 nodes, 300 edges
✓ cycle_graph: 100 nodes, 100 edges
✓ path_graph: 100 nodes, 99 edges
✓ star_graph: 100 nodes, 99 edges
✓ tree: 100 nodes, 99 edges
✓ grid_graph: 100 nodes, 180 edges
✓ karate_club: 34 nodes, 78 edges
```

Core graph tests pass:
```bash
$ pytest tests/modules/test_graph_core.py -v
============================== 39 passed in 0.04s ==============================
```

## Files Modified

- `python-groggy/python/groggy/generators.py`: Optimized all generator methods
  - `complete_graph()`: Pre-allocation hints
  - `erdos_renyi()`: Smart algorithm selection
  - `barabasi_albert()`: Repeated-nodes preferential attachment
  - `watts_strogatz()`: Set-based rewiring
  - All others: Already using bulk patterns, verified

## Usage Examples

```python
from groggy import generators

# Generate 100k node scale-free network in ~3 seconds
g = generators.barabasi_albert(100000, m=3, seed=42)
print(f"{g.node_count()} nodes, {g.edge_count()} edges")
# 100,000 nodes, 299,994 edges

# Generate 100k node small-world network in ~3 seconds  
g = generators.watts_strogatz(100000, k=6, p=0.1, seed=42)
print(f"{g.node_count()} nodes, {g.edge_count()} edges")
# 100,000 nodes, 300,000 edges

# Sparse random graph with smart algorithm selection
g = generators.erdos_renyi(50000, p=0.01, seed=42)
# Uses O(E) sampling instead of O(N²) iteration
```

## Future Enhancements

Potential additional optimizations (not implemented):
1. Numba JIT compilation for hot loops
2. Cython implementation of core algorithms
3. Move generator logic to Rust core for maximum performance
4. Parallel edge generation using multiprocessing
5. Memory-mapped file support for very large graphs

## References

- NetworkX generator implementations
- "Efficient generation of large random networks" (Batagelj & Brandes, 2005)
- Repository principle: Attribute-first, columnar operations
