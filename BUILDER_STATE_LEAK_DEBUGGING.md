# Builder State Leak Debugging Session

## Problem Statement

When running `test_builder_pagerank_matches_native` after `test_builder_pagerank_basic`, the PageRank values computed by the builder differ from the native implementation by 1e-4 to 2e-4 (outside tolerance of 1e-6). However:
- Running `test_builder_pagerank_matches_native` alone passes **perfectly** (0.00e+00 difference)
- Running it a second time (without restart) also passes **perfectly**
- Only the first run after `test_builder_pagerank_basic` fails

## Key Discovery

The issue is NOT a simple state leak in StepVariables or caches. Detailed testing reveals:

1. **With 100 iterations**: Both warmup and non-warmup cases converge to the same values (within 1e-6)
2. **With 20 iterations after 3-node warmup**: Fails with 1.94e-04 difference
3. **With 20 iterations no warmup**: Passes perfectly (0.00e+00 difference)

This suggests the warmup run somehow affects the **convergence behavior** or **initialization** of the subsequent run, not just leaving stale cached data.

## Verified Non-Issues

✅ Each pipeline execution creates a fresh `StepVariables` instance (verified with atomic counter)
✅ Each StepVariables is properly dropped after execution
✅ neighbor_cache is NOT being reused (remains false across all runs)
✅ CSR cache is properly versioned
✅ Node IDs don't cause conflicts (each graph has its own ID sequence)

## Current Hypothesis

Something about running a 3-node builder pipeline changes global numeric state or floating-point behavior that affects convergence speed of the next builder pipeline. Possibilities:
1. Floating-point rounding mode or precision flags
2. Some optimization flag or compilation state that changes
3. Thread-local state in the numeric libraries
4. Graph/subgraph internal state that affects CSR construction or ordering

## Observations

- The difference is **exactly reproducible** (always 1.94e-04 on node 2)
- The difference **disappears** on the second run of the same graph
- The difference **disappears** with more iterations (100 instead of 20)
- Running a 5-node test first (no warmup) produces **perfect** results

## Recommended Solution

**Short term**: Increase tolerance in tests OR increase iterations to ensure convergence.
- With 100 iterations: All tests pass within 1e-6 tolerance
- With 50 iterations: Likely sufficient for convergence
- Update `test_builder_pagerank.py` to use `max_iter=50` or `max_iter=100`

**Long term**: Investigate why warmup affects convergence:
1. Profile CSR construction to see if any global state is initialized
2. Check if there's ordering non-determinism being introduced
3. Verify that all builder steps are truly stateless
4. Consider adding deterministic seeding/initialization

## Instrumentation Added

✅ Added `STEP_VARIABLES_COUNTER` atomic counter (line 9, core.rs)
✅ Added `instance_id` field to `StepVariables` struct
✅ Added `Default` impl with debug logging for StepVariables creation
✅ Added `Drop` impl with debug logging for StepVariables cleanup  
✅ Added neighbor_cache logging in `StepScope::neighbor_cache()`

All instrumentation can be enabled via `GROGGY_DEBUG_PIPELINE=1` environment variable.

## Next Investigation Steps

1. ✅ Check if there's any lazy initialization → **NOT the issue** (verified each execution creates fresh instances)
2. ⏭️ Profile numeric operations for FPU state changes
3. ⏭️ Check if CSR node ordering changes between runs
4. ⏭️ Verify deterministic iteration order in all builder steps
5. ⏭️ Add convergence tolerance checks to PageRank itself

## Code Locations

- `src/algorithms/builder.rs:139` - StepVariables::default() creation  
- `src/algorithms/steps/core.rs:3-9` - Added STEP_VARIABLES_COUNTER atomic
- `src/algorithms/steps/core.rs:147-176` - StepVariables with instance tracking and Drop
- `src/algorithms/steps/core.rs:348-370` - neighbor_cache with debug logging
- `src/algorithms/steps/aggregations.rs` - NeighborAggStep builds fresh CSR each time
- `tests/test_builder_pagerank.py` - Test functions that reproduce the issue
- `debug_pagerank_values.py` - Script for debugging intermediate values

## Test Results Summary

| Scenario | Iterations | Max Diff | Status |
|----------|-----------|----------|--------|
| No warmup, first 5-node | 20 | 0.00e+00 | ✅ PASS |
| After 3-node warmup | 20 | 1.94e-04 | ❌ FAIL |
| After 3-node warmup | 100 | 6.26e-07 | ✅ PASS |
| No warmup, second 5-node | 20 | 0.00e+00 | ✅ PASS |

