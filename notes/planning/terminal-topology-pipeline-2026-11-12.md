# Atomic terminal topology removals — 2026-11-12

Existing scopes: 09.4b.2 / AR3 and 30.1 / AR7.8. Complete bounded removal
inventory; parent compound/read-after-topology scope stays open. No new rows.

## Implemented

Two native primitives, `graph.remove_nodes` and `graph.remove_edges`, accept
explicit IDs or keys of matching typed map variables (nodes also accept columns).
Strict controls reject unknown parameters, bool/fractional/negative IDs, ambiguous
inputs and noninteger effective timestamps. Source values are not predicates.

A persistent local atomic builder can perform certified property work, followed
by multiple terminal removals. Steps only stage touched properties/IDs; the
existing checked GraphBatch publisher validates and accepts one mixed delta.
No intermediate publication or writable graph clone. Removed-target property
writes, duplicate removals, invalid identities and invalid endpoint timelines
reject the complete operation. Node cascades include owner edges outside the
selection and use the canonical explicit/default effective-time rules.

Local output rollback is reused. Failure restores earlier Context outputs;
staging is cleared for reuse. No-op removal stages publish no state. On success,
the returned selection keeps surviving originally selected IDs, including
isolates; omitted parallel edges do not reappear. A subsequent ordinary outer
pipeline algorithm receives this selection. Python freezes the result normally.

## Checks and measurements

- Full Rust gate: **1,551 passed / 54 ignored**; a final focused native suite
  passes **5 tests**, including the subsequently added exact-edge selection case.
- Strict all-target Clippy passed. Release in-tree extension rebuilt/staged;
  three existing FFI warnings remain.
- Final maintained Python gate: **2,633 passed / 1 skipped / 15 warnings**.
  The graph cyclic-GC warning remains under 19.2; see its separate diagnostic.
- **263 executable documentation examples** pass, including two new examples.
- New native tests cover mixed acceptance, directed/undirected cascades,
  retained snapshots, per-change/default clocks, duplicate/late errors,
  cancellation, selection/domain checks, transient/outer atomic rejection,
  context reuse and exact selected edge retention. Python additionally covers
  final temporal rejection, explicit edge/cascade precedence, stale sources,
  frozen returned values and following ordinary pipeline execution.

`atomic_builder_properties --terminal-removals` measures five native release
runs on each full path graph. Fixture and initial capture are excluded; the two
stages remove edge 0 and the last node, including its implicit incident edge.
Median: **0.362 ms at 1,000 nodes / 999 edges**, **33.164 ms at 100,000 nodes /
99,999 edges**. Raw samples: `../benchmarks/terminal-topology-pipeline-2026-11-12.json`.
Includes scope validation, staging, checked publication and surviving membership
reconstruction. This is a receipt, not a before/after speedup, allocation/RSS,
FFI or billion-node claim. Membership rebuilding scans selected nodes/edges;
the existing Python frozen result capture cost is outside this native receipt.

## Remaining boundary

This section records the initial `d65b0165` boundary. The same-day follow-on
`staged-graph-capture-2026-11-12.md` extends it with explicit immutable graph/table
reads after removals; ordinary unadapted calculations remain prohibited there.

Non-atomic and transient topology builders reject. Outer atomic pipelines still
certify property-only effects and reject these builders before executing any
algorithm. Removal stages must end the local step program; even aliases and
immutable projections must precede them. Generic staged topology reads,
additions, endpoint revision steps, loops and opaque custom effects remain open.
Custom native capability declarations are trusted, not an effect sandbox.

Public contract/examples: `docs/api/terminal-topology-pipelines.md`.
Inventory remains **215 checked / 53 open / 268 total**.
