# Builder PageRank Profiling Findings

## Summary

Created detailed profiling script `profile_pr_detailed.py` based on the connected components profiling approach. The profiling reveals critical performance bottlenecks in the builder-based PageRank implementation compared to the native implementation.

## Key Findings

### Performance Gap
- **Builder PageRank (50k nodes, ~275k edges)**: 4,890ms total time
- **Native PageRank (same graph)**: 72.6ms total time  
- **Performance gap**: **67.37x slower** (builder vs native)

### Correctness
- ✅ Results are **numerically correct**
  - Max difference: 0.00000017
  - Avg difference: 0.00000002
- The builder primitives produce the right answer, just very slowly

### Bottleneck Analysis

From the 100k node profiling run, the dominant time is spent in:

1. **FFI Overhead**: 4.98 seconds total
   - `ffi.pipeline_run`: 4.98s
   - This is the cross-language boundary cost

2. **Neighbor Aggregation**: 439ms across 20 iterations
   - Each `neighbor_agg` step: 20-30ms
   - 20 iterations × ~22ms = **440ms just on neighbor aggregation**
   
3. **Algorithm Phase Timings**:
   - `pipeline.run`: 1,505ms
   - `algorithm.builder.step_pipeline`: 1,505ms
   - `neighbor_agg.total`: 439ms
   
4. **Per-Step Overhead**:
   - 349 total pipeline steps (20 iterations × ~17 steps/iteration)
   - Each small operation (mul, add, where) is a separate FFI call
   - Hundreds of tiny 0.001-0.010ms operations add up

### Root Causes

#### 1. FFI Call Overhead
Every primitive step requires:
- Python → Rust function call
- Argument marshalling
- GIL release/acquire
- Return value marshalling
- Python → Rust for next step

With 349 steps, even 1ms overhead per step = 349ms of pure FFI tax.

#### 2. No Step Fusion
The builder emits individual primitives:
```rust
contrib = mul(ranks, inv_degrees)      // Step 1: FFI call
contrib = where(is_sink, 0.0, contrib) // Step 2: FFI call  
neighbor_sum = neighbor_agg(contrib)   // Step 3: FFI call
damped = mul(neighbor_sum, 0.85)       // Step 4: FFI call
```

Native PageRank fuses these into a single hot loop:
```rust
for node in nodes {
    let contrib = if is_sink { 0.0 } else { rank[node] / degree[node] };
    // accumulate directly into neighbor sums
}
```

#### 3. Data Movement
Each primitive:
- Reads from HashMap (StepVariables)
- Allocates new Vec for output
- Writes back to HashMap
- Next step reads from HashMap again

Native code uses pre-allocated buffers and ping-pongs between two arrays.

#### 4. No CSR Reuse Across Iterations
`neighbor_agg` builds or retrieves CSR on every call. Even with caching, there's lookup overhead. Native PageRank builds CSR once and reuses the same pointers for all iterations.

## What Native PageRank Does Differently

Looking at `src/algorithms/centrality/pagerank.rs`:

1. **Single allocation**: Two rank buffers (current, next)
2. **Fused iteration loop**:
   ```rust
   for _ in 0..max_iter {
       for node in nodes {
           let out_deg = degrees[node];
           let contrib = if out_deg == 0 { 0.0 } else { ranks[node] / out_deg };
           
           for neighbor in csr.neighbors(node) {
               next_ranks[neighbor] += damping * contrib;
           }
       }
       
       // Handle teleport + sinks inline
       for node in nodes {
           next_ranks[node] += teleport_term + sink_contrib;
       }
       
       std::mem::swap(&mut ranks, &mut next_ranks);
       next_ranks.fill(0.0);
   }
   ```

3. **No HashMap lookups**: Direct array indexing
4. **No FFI boundar**: Entire algorithm in Rust
5. **Cache-friendly**: Sequential memory access patterns

## Optimization Strategies (Not Prioritized)

### Strategy 1: Step Fusion (High Impact)
- Detect common patterns in builder pipelines
- Fuse into specialized "macro steps"
- Example: `PageRankIterStep` that does one full iteration
- Tradeoff: Reduces composability, adds special cases

### Strategy 2: Batch Execution (Medium Impact)  
- Accumulate multiple primitive ops
- Execute as batch with shared state
- Reduces FFI crossings
- Tradeoff: Complicates error handling, debugging

### Strategy 3: JIT Compilation (High Impact, High Effort)
- Compile builder pipeline to native code at runtime
- Use cranelift or similar
- Removes FFI overhead entirely
- Tradeoff: Major engineering effort, complex

### Strategy 4: Smarter Caching (Low-Medium Impact)
- Pre-build CSR once per pipeline, not per step
- Reuse allocations across loop iterations  
- Intern intermediate results
- Tradeoff: State management complexity

### Strategy 5: Parallel Primitives (Medium Impact)
- Run independent steps in parallel
- Use rayon for node-parallel operations
- Tradeoff: Limited by Amdahl's law, overhead for small graphs

## Recommendations

**Do NOT pursue step fusion or macro steps.** The profiling clearly shows the issue is systemic FFI overhead and data movement, not individual primitive logic. The goal is to make **any algorithm** fast through primitives, not to hardcode PageRank.

### Next Steps (High Priority)

1. **Profile FFI Call Costs**
   - Instrument exact time in FFI boundary vs Rust logic
   - Measure GIL acquire/release overhead
   - Identify if marshalling or function dispatch dominates

2. **Implement Batch Mode**
   - Execute multiple primitives without returning to Python
   - Pass entire pipeline spec to Rust, execute there
   - Return only final results
   - This keeps composability while removing FFI tax

3. **Memory Pooling**
   - Reuse Vec allocations across steps
   - Use arena allocator for step variables
   - Ping-pong buffers for loop iterations

4. **CSR Pre-building**
   - Build CSR once at pipeline start
   - Pass CSR handle through steps
   - neighbor_agg becomes pure indexing

### Success Criteria

Target: **10x improvement** (670ms → 67ms to match native)

- FFI overhead: 4.98s → <500ms (batch execution)
- Neighbor agg: 439ms → <50ms (CSR reuse, memory pooling)
- Step overhead: Eliminate 349 separate FFI calls

## Files

- **Profiling script**: `profile_pr_detailed.py`
- **Output**: `pr_profiling_output.txt`
- **Builder implementation**: `python-groggy/python/groggy/builder.py`
- **Primitive steps**: `src/algorithms/steps/*.rs`
- **Native PageRank**: `src/algorithms/centrality/pagerank.rs`

## Notes

- The correctness is solid - focus is purely on performance
- Small graphs (4-100 nodes) show builder is competitive due to fixed overhead
- Medium-large graphs (10k-100k+ nodes) show the 50-150x gap
- The primitives themselves are fast (each mul/add is <0.01ms)
- The architecture (FFI per step) is the bottleneck, not the code quality

---

**Status**: Profiling complete, ready for optimization phase
**Next**: Deep-dive FFI overhead analysis and batch execution prototype
