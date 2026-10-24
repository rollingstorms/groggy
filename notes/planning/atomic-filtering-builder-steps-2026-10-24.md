# Atomic property filtering and ordering — 2026-10-24

Closes **09.4b.2n**, extending **30.1 / AR7.8**.

## Implemented

- Certify node/edge property filters, node property sorting and node top-k for
  atomic property execution and nested loops. Property readers use StepScope's
  shared overlay, so earlier attachments are visible; transient reads still use
  owner values. No selection/topology mutation is introduced.
- Decode predicate JSON payloads before deserialization, repairing the prior
  tagged-wrapper failure. Validate ordered/non-NaN range bounds during preparation.
- Propagate property read errors instead of silently treating them as absent or
  nonmatching. Filters retain sparse 1-valued maps, skip absent properties, and
  can match present null through explicit None equality. Unsupported property-to-
  step conversions remain nonmatches, preserving existing predicate support.
- Preserve typed Eq/Ne equality. Relational/range comparisons and ordering reuse
  the numerical core's exact integer/float comparator, fixing rounding near 2**53
  without inventing a second numeric order.
- Sort/top-k validate one comparable scalar domain before sorting. Numeric, text
  and Boolean domains are separate; missing/null rank last in either direction.
  Reject NaNs, unsupported values and mixed domains rather than treating them as
  equal. Ordered infinities are valid numeric sort values.
- Break equal-value ties by ascending entity ID for both directions. Sorting
  returns ranks for all selected nodes; top-k returns ranks for at most k source
  nodes. It does not return original scores. Validate top-k source IDs against
  the active selection and check rank conversion to i64.
- Check cancellation during reads/value validation, around sorting and before
  result installation. Comparison sorting itself is not interruptible. Failure
  leaves old result variables intact; atomic runs discard earlier staged writes.

## Remaining work and costs

Statistical/reduction and neighbor aggregation families remain uncertified. Their
numeric extraction/order/empty/error behavior needs separate review; enabling this
filtering slice does not silently certify them. Streaming, alternate batch plans,
optimizer effects and atomic defaults remain parent work.

Ordering validates the input and performs a full O(n log n) sort with O(n) local
storage. Top-k does not claim a heap selection optimization. These paths retain
only touched property staging and working variables, not a graph clone.

`benches/filtering_builder_steps.rs` records four native workloads at 1,000/10,000
nodes/edges, five samples each: node/edge filters, sorting and top-k with staged
node/edge properties and one publication. Fixture/spec construction is excluded;
all rank/mask outputs and clean publication are checked. Raw receipt:
`notes/benchmarks/filtering-builder-steps-2026-10-24.json`.
This is a bounded baseline, not matched speedup, allocation/RSS, FFI or production
throughput evidence.

## Verification

Native cases check exact mixed numeric ordering, deterministic ties, typed equality,
range/domain rejection and stale reads preserving old result variables. Python
coverage checks mixed staged node/edge filtering, nested loops, transient owner
reads, sparse masks, null versus absent, deterministic null ordering, exact top-k/
range behavior near 2**53, malformed range rejection, and atomic rollback on NaN,
collections and mixed sort domains. Full gate results follow below.

- Full offline/locked native suite: **950 passed, 52 ignored**.
- Strict all-target native Clippy with warnings denied: passed.
- Rebuilt/staged in-tree extension using existing macOS dynamic lookup settings;
  three existing FFI warnings remain. No package installation.
- **111 focused atomic-builder cases passed**.
- Full root Python API gate: **1,187 passed, 1 skipped**, excluding the unrelated
  owner benchmark test and separate legacy modules survey.
- **44 executable documentation examples passed**.
- Eight release workloads passed all output/publication checks. Observed median
  milliseconds at 1,000/10,000 nodes are recorded below; distinct kernel timings
  do not establish universal speedup or production budgets.
  - core.sort_nodes_by_attr: 4.063/35.933 ms.
  - core.top_k: 2.869/28.891 ms.
  - core.filter_nodes_by_attr: 3.295/34.411 ms.
  - core.filter_edges_by_attr: 3.320/34.592 ms.
- `git diff --check` passed; owner development/benchmark changes remain separate.
