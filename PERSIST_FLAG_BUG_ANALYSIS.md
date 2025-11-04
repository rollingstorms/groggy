# persist=False Flag Not Being Respected - Bug Analysis

## Problem

The `persist=False` parameter added to the `apply()` API is not being respected by most algorithms. Only `connected_components` properly checks the `ctx.persist_results()` flag before writing attributes.

## Investigation Results

### Test Results

**Connected Components (respects persist flag):**
```
persist=True:  0.012914s
persist=False: 0.006585s
Speedup: 1.96x ✅
```

**PageRank (ignores persist flag):**
```
persist=True:  0.015006s
persist=False: 0.008931s
Speedup: 1.68x (but attributes still written!)
```

The speedup is seen because there's overhead in the pipeline infrastructure itself, but the actual attribute writes are NOT being skipped in most algorithms.

### Root Cause

**Only 1 out of ~15 algorithms checks persist_results():**

```bash
$ grep -rn "ctx.persist_results()" src/algorithms/ --include="*.rs"
src/algorithms/community/components.rs:404:        if ctx.persist_results() {
```

**Algorithms that IGNORE the flag:**
- `pagerank.rs` - line 205: unconditional `set_node_attrs()`
- `lpa.rs` - line 288: unconditional `set_node_attr_column()`  
- `betweenness.rs` - unconditional writes
- `closeness.rs` - unconditional writes
- `louvain.rs` - unconditional writes
- `leiden.rs` - unconditional writes
- `infomap.rs` - unconditional writes
- All pathfinding algorithms
- All other centrality algorithms

### Example from PageRank

```rust
// src/algorithms/centrality/pagerank.rs:195-205
let mut attrs: HashMap<AttrName, Vec<(NodeId, AttrValue)>> = HashMap::new();
attrs.insert(
    self.output_attr.clone(),
    rank.into_iter()
        .map(|(node, score)| (node, AttrValue::Float(score as f32)))
        .collect(),
);

subgraph
    .set_node_attrs(attrs)  // ❌ Always writes, never checks ctx.persist_results()
    .map_err(|err| anyhow!("failed to persist PageRank scores: {err}"))?;
```

### Example from Connected Components (correct implementation)

```rust
// src/algorithms/community/components.rs:404-415
if ctx.persist_results() {  // ✅ Checks the flag first!
    // Convert to AttrValue in single step (no intermediate HashMap!)
    let attr_values: Vec<(NodeId, AttrValue)> = node_assignments
        .iter()
        .map(|(node, comp_id)| (*node, AttrValue::Int(*comp_id)))
        .collect();

    ctx.record_timer("community.connected_components.write_attrs", || {
        subgraph.set_node_attr_column(output_attr, attr_values)
    })
    .map_err(|e| anyhow!("failed to write component attributes: {e}"))?;
}
```

## Impact

1. **Benchmark comparisons are UNFAIR**: The benchmark script now uses `persist=False`, but most algorithms still write attributes, making groggy slower than it should be
2. **API contract is broken**: Users expect `persist=False` to skip attribute writes, but it doesn't
3. **Performance opportunity lost**: We could be 2-5x faster on many algorithms if we properly skipped persistence

## Solution Required

All algorithms need to be updated to check `ctx.persist_results()` before writing attributes:

### Pattern to Follow

```rust
// After computing results...
if ctx.persist_results() {
    ctx.record_timer("algorithm.name.write_attrs", || {
        subgraph.set_node_attrs(attrs)
        // or subgraph.set_node_attr_column(...)
        // or subgraph.set_edge_attrs(...)
    })
    .map_err(|e| anyhow!("failed to write attributes: {e}"))?;
}
```

### Files That Need Updates

**Centrality algorithms:**
- `src/algorithms/centrality/pagerank.rs`
- `src/algorithms/centrality/betweenness.rs`
- `src/algorithms/centrality/closeness.rs`

**Community algorithms:**
- `src/algorithms/community/lpa.rs`
- `src/algorithms/community/louvain.rs`
- `src/algorithms/community/leiden.rs`
- `src/algorithms/community/infomap.rs`
- `src/algorithms/community/girvan_newman.rs`
- `src/algorithms/community/modularity.rs`

**Pathfinding algorithms:**
- `src/algorithms/pathfinding/dijkstra.rs`
- `src/algorithms/pathfinding/astar.rs`
- `src/algorithms/pathfinding/bfs_dfs.rs`

## Priority

**HIGH** - This is a correctness issue. The API promises functionality that doesn't work, and it affects benchmark results and user expectations.

## Workaround

For now, only `connected_components` properly supports `persist=False`. For other algorithms in benchmarks, we're still paying the attribute write cost even though we're passing the flag.
