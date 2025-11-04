# Connected Components - Detailed Profiling Guide

## Overview

The Connected Components algorithm now includes comprehensive, granular profiling that tracks every phase and operation with call counts and timing. This profiling system provides insights similar to `gprof` or `callgrind` but integrated directly into the algorithm execution.

## Enabling Profiling

Set the environment variable to enable detailed profiling output:

```bash
export GROGGY_PROFILE_CC=1
python your_script.py
```

The profiling report will be automatically printed after algorithm execution.

## Profiling Metrics

### Call Counter System

The profiling system tracks:
- **Call count**: How many times each phase was executed
- **Total duration**: Cumulative time across all calls
- **Average duration**: Mean time per call (in microseconds)

### Phases Tracked

#### High-Level Phases

| Phase | Description | Call Count |
|-------|-------------|------------|
| `cc.total_execution` | End-to-end algorithm runtime | 1 |
| `cc.collect_nodes` | Gathering node IDs from subgraph | 1 |
| `cc.cache_miss` / `cc.cache_hit` | Cache lookup result | 1 |
| `cc.write_attributes` | Writing results to graph | 1 (if persisting) |
| `cc.store_output` | Storing output for non-persist mode | 1 (if not persisting) |

#### Graph Construction

| Phase | Description | Notes |
|-------|-------------|-------|
| `cc.build_indexer` | Creating dense/sparse node indexing | Uses heuristic for dense vs HashMap |
| `cc.get_snapshot` | Retrieving graph snapshot | Includes borrowing and dereferencing |
| `cc.build_csr` | Building CSR representation | Two-pass: count degrees, fill neighbors |
| `cc.csr_nodes` | CSR node count | Stored as nanoseconds for metric tracking |
| `cc.csr_edges` | CSR edge count | Total edges in adjacency structure |

#### BFS-Specific (Undirected/Weak Mode)

| Phase | Description | Call Count Pattern |
|-------|-------------|-------------------|
| `cc.bfs.alloc_visited` | Allocating visited array | 1 |
| `cc.bfs.alloc_queue` | Allocating BFS queue | 1 |
| `cc.bfs.component_traversal` | Per-component BFS | # of components |
| `cc.bfs.total_components` | Total components found | 1 (value stored) |
| `cc.bfs.nodes_processed` | Total nodes visited | 1 (count stored) |
| `cc.bfs.edges_scanned` | Total edges examined | 1 (count stored) |

#### Tarjan-Specific (Strong Mode)

| Phase | Description | Call Count Pattern |
|-------|-------------|-------------------|
| `cc.tarjan.alloc_arrays` | Allocating state arrays | 1 |
| `cc.tarjan.alloc_call_stack` | Allocating explicit DFS stack | 1 |
| `cc.tarjan.scc_extraction` | Extracting each SCC | # of SCCs |
| `cc.tarjan.total_components` | Total SCCs found | 1 (value stored) |
| `cc.tarjan.nodes_visited` | Total nodes in DFS | 1 (count stored) |
| `cc.tarjan.edges_examined` | Total edges examined | 1 (count stored) |
| `cc.tarjan.max_recursion_depth` | Maximum call stack depth | 1 (value stored) |
| `cc.tarjan.stack_pushes` | SCC stack push operations | 1 (count stored) |
| `cc.tarjan.stack_pops` | SCC stack pop operations | 1 (count stored) |

#### Post-Processing

| Phase | Description | Condition |
|-------|-------------|-----------|
| `cc.build_node_lists` | Building component node lists | Always |
| `cc.assign_edges` | Grouping edges by component | If `persist_results()` |
| `cc.skip_edge_assignment` | Skipped edge grouping | If not persisting |
| `cc.edges_processed` | Total edges processed | If assigning |
| `cc.build_assignments` | Building node assignments | Always |
| `cc.convert_to_attr_values` | Converting to attribute values | If persisting |
| `cc.store_cache` | Caching results | If persisting & not empty |

## Example Output

```
================================================================================
Profiling Report: Connected Components
================================================================================

Phase                                                   Calls   Total (ms)     Avg (μs)
------------------------------------------------------------------------------------
cc.total_execution                                          1        2.324     2324.167
cc.write_attributes                                         1        1.234     1233.542
cc.build_csr                                                1        0.398      397.708
cc.get_snapshot                                             1        0.364      364.459
cc.store_cache                                              1        0.217      216.584
cc.assign_edges                                             1        0.053       53.250
cc.bfs.component_traversal                                 61        0.017        0.276
cc.build_node_lists                                         1        0.012       12.250
cc.collect_nodes                                            1        0.006        6.292
cc.build_assignments                                        1        0.006        5.750
cc.build_indexer                                            1        0.004        3.916
cc.bfs.edges_scanned                                        1        0.004        3.878
cc.csr_edges                                                1        0.004        3.878
cc.convert_to_attr_values                                   1        0.003        3.209
cc.csr_nodes                                                1        0.002        2.000
cc.input_nodes                                              1        0.002        2.000
cc.bfs.nodes_processed                                      1        0.002        2.000
cc.edges_processed                                          1        0.002        1.939
cc.alloc_labels                                             1        0.000        0.334
cc.cache_miss                                               1        0.000        0.125
cc.bfs.alloc_visited                                        1        0.000        0.083
cc.bfs.total_components                                     1        0.000        0.061
cc.bfs.alloc_queue                                          1        0.000        0.041
================================================================================
```

## Interpreting Results

### Identifying Bottlenecks

1. **Sort by Total Time**: Phases are sorted descending by total duration
   - Top entries are where the algorithm spends most time
   - In the example above, `write_attributes` takes 53% of total time (1.234ms / 2.324ms)

2. **Check Call Counts**: High call counts with low average time are normal
   - `cc.bfs.component_traversal` called 61 times (one per component)
   - Average of 0.276 μs is excellent

3. **Look for Unexpected Patterns**:
   - Many cache misses? Consider pre-warming cache
   - High `assign_edges` time? Consider disabling persistence if edges aren't needed
   - Long `get_snapshot` time? Graph access patterns may need optimization

### Performance Characteristics

**Expected Times** (for 2000 node, 2000 edge graph):
- Total execution: 1-3 ms
- CSR build: 0.3-0.5 ms
- BFS traversal: < 0.1 ms (sub-microsecond per component)
- Edge assignment: 0.05-0.1 ms

**Tarjan Performance** (800 node directed graph):
- Total execution: 0.3-0.5 ms
- Tarjan traversal: < 0.1 ms
- SCC extraction: 50-100 ns per SCC
- Max recursion depth: proportional to graph diameter

### Comparing Modes

| Metric | Undirected/Weak | Strong (Tarjan) |
|--------|-----------------|-----------------|
| Allocation overhead | Lower (simple arrays) | Higher (multiple state arrays) |
| Per-node cost | ~1 ns | ~1-2 ns |
| Per-component cost | Linear BFS | SCC extraction |
| Memory usage | O(V + E) | O(V + E) + stack overhead |

## Common Bottlenecks and Solutions

### 1. High `write_attributes` Time

**Symptom**: `write_attributes` > 50% of total time

**Causes**:
- Large attribute write batch
- Graph mutation overhead
- Persistence layer latency

**Solutions**:
- Use non-persist mode if you don't need attributes written
- Batch attribute writes at application level
- Check storage backend performance

### 2. Slow `build_csr` Phase

**Symptom**: `build_csr` > 20% of total time

**Causes**:
- Sparse node ID distribution causing HashMap overhead
- Many reverse edges being added (weak mode)
- Large degree nodes

**Solutions**:
- Compact node IDs to dense range if possible
- Use strong mode if directional information isn't needed
- Pre-filter high-degree nodes if appropriate

### 3. Many Small Components

**Symptom**: `cc.bfs.component_traversal` called hundreds of times

**Impact**: Minimal (each traversal is microseconds)

**Context**: Many small components is a graph property, not a performance issue. The algorithm handles this efficiently.

### 4. High Cache Miss Rate

**Symptom**: Always seeing `cache_miss`, never `cache_hit`

**Causes**:
- Cache disabled
- Graph changing between calls
- Different algorithm parameters

**Solutions**:
- Enable persistence mode for caching
- Avoid mutating graph between connected components calls
- Use same mode (undirected/weak/strong) consistently

## Advanced Analysis

### Using Call Counts as Statistics

Some phases use call counts to store graph statistics:

```python
# Extract statistics from profiling report
import os
os.environ['GROGGY_PROFILE_CC'] = '1'

# Run algorithm...
# Then parse output or access context.call_counters()

# Metrics stored as "duration" but actually counts:
# - cc.bfs.nodes_processed: total nodes visited
# - cc.bfs.edges_scanned: total edges examined
# - cc.tarjan.max_recursion_depth: maximum DFS depth
# - cc.input_nodes: input node count
```

### Comparing Algorithms

Run profiling on same graph with different modes:

```python
# Undirected
g.apply(algorithms.community.connected_components(
    mode='undirected', output_attr='cc_undirected'))

# Strong
g.apply(algorithms.community.connected_components(
    mode='strong', output_attr='cc_strong'))
```

Compare:
- Total execution time
- Per-component costs
- Memory allocation patterns
- Edge processing efficiency

### Integration with External Profilers

The profiling can complement external tools:

**Using `perf` (Linux)**:
```bash
perf record -g python script.py
perf report
```

**Using `py-spy`**:
```bash
py-spy record -o profile.svg -- python script.py
```

**Using `cProfile`**:
```python
import cProfile
cProfile.run('g.apply(algorithms.community.connected_components(...))')
```

The built-in profiling provides algorithm-specific details that external profilers miss (like BFS vs Tarjan phases, cache hits, etc).

## Performance Tuning Checklist

✅ **Graph Preparation**
- [ ] Compact node IDs to dense range
- [ ] Remove self-loops if not needed
- [ ] Pre-filter isolated nodes

✅ **Algorithm Configuration**
- [ ] Choose appropriate mode (undirected/weak/strong)
- [ ] Disable persistence if only computing components
- [ ] Enable caching for repeated calls

✅ **Monitoring**
- [ ] Enable profiling during development
- [ ] Establish baseline metrics for regression detection
- [ ] Track cache hit rates

✅ **Optimization Targets**
- [ ] Keep `build_csr` < 20% of total time
- [ ] Minimize `write_attributes` if possible
- [ ] Achieve cache hits on repeated runs

## Summary

The detailed profiling system provides gprof-like granularity for understanding Connected Components performance:

- **Every phase is timed** with microsecond precision
- **Call counts reveal algorithm behavior** (components found, edges scanned)
- **Bottlenecks are immediately visible** through sorted output
- **Cache efficiency is tracked** for optimization opportunities
- **Statistics are embedded** as pseudo-duration metrics

Use `GROGGY_PROFILE_CC=1` during development to understand performance characteristics, then disable in production for minimal overhead.

---

**Version**: 0.2.0 with detailed profiling  
**Feature**: Granular phase tracking with call counters  
**Overhead**: Minimal (<1% when profiling enabled via environment variable)
