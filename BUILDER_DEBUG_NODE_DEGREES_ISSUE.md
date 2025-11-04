# Builder Debug: Node Degrees Returning Doubled Values

## Issue Summary

The builder-based PageRank and LPA implementations are producing incorrect results. Root cause identified: the `node_degrees` primitive returns doubled degree values.

### Expected vs Actual

For the 5-node test graph with edges: 0→1, 1→2, 2→0, 2→3, 3→4, 4→2

**Expected degrees:**
- Node 0: 1  
- Node 1: 1
- Node 2: 2
- Node 3: 1
- Node 4: 1
- Sum: 6 (matches edge count)

**Actual degrees from builder:**
- Node 0: 2
- Node 1: 2
- Node 2: 4
- Node 3: 2
- Node 4: 2  
- Sum: 12 (exactly 2x edge count)

### What We've Verified

✅ **Working correctly:**
- Raw graph `node.neighbors` returns correct degree counts
- `graph_ref.out_degree(node)` when called directly returns correct values
- `init_nodes` with default values works
- `init_nodes(unique=True)` creates sequential indices
- `broadcast_scalar` works
- Simple 3-node cycle PageRank produces correct results

❌ **Broken:**
- `node_degrees(source_map)` returns 2x the correct values
- This breaks PageRank (wrong neighbor contributions)
- This breaks LPA (wrong degree-based tie-breaking)

### Code Locations

**Rust implementation:**
- `src/algorithms/steps/structural.rs` lines 40-71: `NodeDegreeStep::apply()`
- Lines 63-65 compute degree: `let degree = graph_ref.out_degree(node).unwrap_or(0);`

**Python builder wrapper:**
- `python-groggy/python/groggy/builder.py`: `node_degrees()` method
- Passes `"source": nodes.name` to the step

**Test files created for debugging:**
- `debug_test_exact.py` - reproduces exact test failure
- `debug_map_contents.py` - shows degrees are doubled
- `debug_degrees_no_source.py` - attempted to test without source
- `test_raw_degrees.py` - proves raw graph has correct degrees
- `debug_5node_single_iter.py` - shows single iteration with wrong degrees

### Changes Made So Far

1. **Fixed `init_nodes` ordering (COMPLETED)**
   - File: `src/algorithms/steps/init.rs` line 38-46
   - Changed `scope.subgraph().nodes()` to `scope.subgraph().ordered_nodes()`
   - Rebuilt with `maturin develop --release`

### Hypotheses to Investigate

1. **Double CSR building?** - Could CSR be built once for the subgraph and again somewhere else, causing edges to be counted twice?

2. **Edge direction confusion?** - Could `out_degree` be counting both directions for some reason, even though graph is directed?

3. **Source map iteration issue?** - When `node_degrees` is passed a source map, could it be iterating nodes differently than expected?

4. **Variable aliasing in loop?** - Could loop unrolling be causing degrees to be computed multiple times and accumulated?

### Recommended Debug Steps

1. **Add Rust logging to NodeDegreeStep::apply:**
   ```rust
   eprintln!("DEBUG: Computing degrees for {} nodes", nodes.len());
   for (i, node) in nodes.iter().enumerate() {
       let degree = graph_ref.out_degree(node).unwrap_or(0);
       eprintln!("DEBUG: Node {}: out_degree={}", node, degree);
       // ... rest of code
   }
   ```

2. **Check if degrees are being computed multiple times** by looking at step execution logs

3. **Verify CSR building** doesn't somehow double-count edges when source map is present

4. **Test with undirected graph** to see if issue persists

### Test Commands

```bash
# Run failing PageRank test
pytest tests/test_builder_pagerank.py::test_builder_pagerank_matches_native -xvs

# Run degree debug script
python debug_map_contents.py

# Verify raw degrees are correct
python test_raw_degrees.py

# Test exact reproduction
python debug_test_exact.py
```

### Current Tolerances

- PageRank test expects: `max_diff < 1e-6`
- Benchmark expects: `max_diff < 1e-5`  
- Current actual diff: ~0.155 (way too large)

### Build Command

```bash
cd /Users/michaelroth/Documents/Code/groggy
maturin develop --release
```

## Notes

- The issue appears ONLY when using builder-based algorithms
- Native PageRank and LPA work correctly
- Simple 3-node cycle works, suggesting issue is graph-size or structure dependent
- Degrees being exactly doubled suggests systematic double-counting, not random error
- The fix to `init_nodes` ordering was correct but didn't solve this issue

## Next Session TODO

1. Add debug eprintln to `NodeDegreeStep::apply` 
2. Run `debug_map_contents.py` and check console output
3. If degrees are correct in Rust but wrong in Python result, investigate result attachment/retrieval
4. If degrees are wrong in Rust, trace through `graph_ref.out_degree()` implementation
5. Consider if subgraph filtering could cause edges to appear twice

