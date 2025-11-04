# Builder Algorithm State Leakage - Debugging Stopping Point

## Current Status

**Issue Confirmed**: Builder-based algorithms show state leakage that causes incorrect results on first execution after a different algorithm has run.

**Reproduction**: Run `python reproduce_state_leak.py` to see the bug in action.

## What We Know

### The Bug Pattern
1. Run `algo1.apply(graph1.view())` → correct results
2. Run `algo2.apply(graph2.view())` **first time** → **WRONG** results (7.37e-02 error)
3. Run `algo2.apply(graph2.view())` **second time** → **CORRECT** results (7.45e-09 error)

### Key Characteristics
- Happens even with just **1 iteration** (not accumulation)
- Affects **different graphs** (graph1 ≠ graph2)
- Affects **different algorithms** (algo1 ≠ algo2)  
- Values are **completely wrong**, not just slightly off
- Node degrees step works correctly (so CSR building itself is OK)
- Second run magically fixes itself

### What We Ruled Out
- ❌ Loop/iteration accumulation (happens with 1 iter)
- ❌ StepVariables reuse (created fresh each time)
- ❌ Global mutable state (none found in algorithms/)
- ❌ Thread-local CSR scratch (properly cleared)
- ❌ Cross-graph subgraph caching (each graph has own subgraph)
- ❌ Python-level caching (no @cache decorators)
- ❌ Basic CSR operations (node degrees are correct)

## Most Likely Root Causes

### Hypothesis 1: CSR Cache Key Collision (80% confidence)
- Subgraph IDs might collide between graph1 and graph2
- Content-based hashing uses node/edge IDs which might overlap
- First run of algo2 gets graph1's CSR from cache
- Second run rebuilds correctly

**Test**: Print `subgraph.subgraph_id` for both graphs, check if same.

**Fix**: Use atomic counter for IDs instead of content hashing, or include Graph pointer in hash.

### Hypothesis 2: View Cache Version Bug (15% confidence)
- Python FFI caches views using `(node_count + edge_count)` as version
- If graph1 and graph2 have same node/edge count, version collides
- Cache returns wrong subgraph

**Test**: Check if graph1 and graph2 have same (3 nodes, 3 edges) vs (5 nodes, 6 edges).

**Note**: This is unlikely since graph1 has 3 nodes and graph2 has 5 nodes.

### Hypothesis 3: CSR Cache Not Cleared Between Graphs (5% confidence)
- Despite version checking, CSR cache might return stale entry
- First execution doesn't detect staleness, second execution rebuilds

**Test**: Add logging to `csr_cache_get()` to see hit/miss/stale patterns.

## Recommended Next Steps

### Step 1: Confirm Subgraph ID Collision
```python
# Add to reproduce_state_leak.py after creating graphs
import ctypes
view1 = graph1.view()
view2 = graph2.view()
# Print internal subgraph IDs somehow
# If they're the same → that's the bug!
```

### Step 2: Add CSR Cache Logging
```rust
// In src/subgraphs/subgraph.rs:398
pub(crate) fn csr_cache_get(&self, add_reverse: bool) -> Option<Arc<Csr>> {
    let version = ...;
    let key = CsrCacheKey { add_reverse };
    
    log::info!("CSR cache lookup: subgraph_id={}, version={}, key={:?}", 
               self.subgraph_id, version, key);
    
    if let Some(entry) = self.topology_cache.borrow().get(&key) {
        if entry.version == version {
            log::info!("  → HIT (version matches)");
            return Some(entry.csr.clone());
        } else {
            log::info!("  → STALE (cached={}, current={})", entry.version, version);
        }
    } else {
        log::info!("  → MISS");
    }
    None
}
```

Run with: `RUST_LOG=groggy=info python reproduce_state_leak.py`

### Step 3: Dump CSR on First vs Second Run
```rust
// In src/algorithms/steps/aggregations.rs:587 (NeighborAggregationStep)
log::info!("Building CSR: node_count={}, edge_count={}", 
           nodes.len(), edges.iter().count());
log::info!("CSR offsets: {:?}", &csr.offsets[..min(10, csr.offsets.len())]);
log::info!("CSR neighbors: {:?}", &csr.neighbors[..min(20, csr.neighbors.len())]);
```

Compare output from first run (wrong) vs second run (correct).

## Files to Reference

- `reproduce_state_leak.py` - Minimal repro script (< 5 seconds to run)
- `STATE_LEAK_INVESTIGATION.md` - Detailed investigation notes
- `tests/test_builder_pagerank.py` - Original failing tests
- `src/subgraphs/subgraph.rs:398` - CSR cache implementation
- `src/algorithms/steps/aggregations.rs:587` - NeighborAggregationStep
- `python-groggy/src/ffi/api/graph.rs` - Python view caching

## Test Commands

```bash
# Reproduce bug (< 5 sec)
python reproduce_state_leak.py

# Run failing tests individually (pass)
pytest tests/test_builder_pagerank.py::test_builder_pagerank_matches_native -xvs

# Run failing tests in sequence (fail)
pytest tests/test_builder_pagerank.py::test_builder_pagerank_basic tests/test_builder_pagerank.py::test_builder_pagerank_matches_native -xvs
```

## When You Return

1. Run `python reproduce_state_leak.py` to confirm bug still exists
2. Add subgraph ID logging to identify if IDs collide
3. If IDs collide → fix by using atomic counter instead of content hash
4. If IDs don't collide → add CSR cache logging to see hit/miss patterns
5. Instrument CSR building to compare first vs second run
6. Once root cause found, add regression test that runs two algos in sequence

## Expected Resolution

Once subgraph ID collision is fixed (if that's the cause), all tests should pass on first run. The fix would be in `src/subgraphs/subgraph.rs` around line 176 where subgraph_id is generated.

Alternative fix: Clear CSR cache at start of each algorithm execution, but that would hurt performance.
