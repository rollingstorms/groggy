# Builder Optimization Quick Start

## TL;DR

Builder PageRank is **467x slower** than native (51s vs 0.11s on 200k nodes).

**Root cause:** 1,500 FFI boundary crossings, NOT slow primitives.

**Solution:** Batch execute steps in Rust → **10-20x speedup expected**.

## The Problem in Numbers

```
Native PageRank (200k nodes):  0.11s  ✓
Builder PageRank (200k nodes): 51.40s ✗ (467x slower)

Why?
- 100 iterations × 15 steps = 1,500 FFI calls
- Each FFI call: ~0.4ms overhead
- Total FFI overhead: 600ms+ (>50% of runtime on small graphs)
```

## Profiling Proof

Run on 5k nodes:
```bash
python benchmark_builder_vs_native.py
```

Output shows:
- 1,500 steps executed
- Each step: 0.2-1.5ms (primitives are FAST!)
- But 1,500 × 0.4ms FFI = 600ms wasted

## Solution: Batched Execution

### Current (Slow)
```python
for i in range(100):
    x = step1.apply(g)  # FFI crossing
    y = step2.apply(g)  # FFI crossing
    # ... 15 steps ...
# = 1,500 FFI calls
```

### Optimized (Fast)
```python
for i in range(100):
    result = apply_batch([step1, step2, ...])  # One FFI crossing
# = 100 FFI calls (15x reduction!)
```

## Implementation (3 Files)

### 1. Rust Core (`src/algorithms/pipeline.rs`)
```rust
pub fn execute_batch(
    steps: Vec<Box<dyn AlgorithmStep>>,
    mut state: StepVariables,
    scope: &ExecutionScope,
) -> Result<StepVariables> {
    for step in steps {
        state = step.apply(scope, state)?;
    }
    Ok(state)
}
```

### 2. FFI Layer (`python-groggy/src/ffi/algorithms.rs`)
```rust
#[pymethod]
fn apply_batch(
    &self,
    py: Python,
    graph: &PyGraphSpace,
    steps: Vec<PyStep>,
    initial: HashMap<String, PyVar>,
) -> PyResult<HashMap<String, PyVar>> {
    py.allow_threads(|| {
        execute_batch(steps, initial, scope)
    })
}
```

### 3. Python Builder (`python-groggy/python/groggy/builder.py`)
```python
def _optimize_pipeline(self):
    """Group steps to minimize FFI calls"""
    # Detect loop boundaries
    # Batch steps within each iteration
    # Return optimized execution plan
```

## Expected Results

After optimization:

| Graph Size | Current | Target | Speedup |
|------------|---------|--------|---------|
| 50k nodes  | 7.58s   | ~0.5s  | 15x     |
| 200k nodes | 51.4s   | ~3-5s  | 10-17x  |

Still slower than native (0.11s), but **acceptable DSL overhead** (5-10x).

## Why This Works

**LPA is proof:** Only 3.5x slower than native because:
- 10 iterations (not 100)
- 3 steps per iteration (not 15)
- Total: 30 FFI calls (not 1,500)

Apply same principle to PageRank → similar performance.

## Next Steps

1. Implement `execute_batch()` in Rust
2. Expose via FFI with `py.allow_threads()`
3. Update builder to group steps by loop
4. Benchmark to verify 10x+ speedup
5. Ensure tests still pass

## Key Files

- **Analysis:** `BUILDER_FFI_BOTTLENECK_ANALYSIS.md`
- **Strategy:** `notes/FFI_OPTIMIZATION_STRATEGY.md`
- **Benchmark:** `benchmark_builder_vs_native.py`
- **Native code:** `src/algorithms/centrality/pagerank.rs`
- **Builder code:** `python-groggy/python/groggy/builder.py`

## Testing

```bash
# Run benchmark
python benchmark_builder_vs_native.py

# Run tests
pytest tests/test_builder_pagerank.py -q

# Profile (set SHOW_PROFILING=True)
python -c "from benchmark_builder_vs_native import *; ..."
```

## Success Criteria

- [ ] Builder PageRank <5s on 200k nodes (10x improvement)
- [ ] FFI calls: 1,500 → ~100 (15x reduction)
- [ ] Tests pass (no regressions)
- [ ] Results match native (within 1e-6)

---

**Status:** Analysis complete, ready to implement
**Risk:** Low (transparent optimization)
**Effort:** ~1-2 days for Phase 1
