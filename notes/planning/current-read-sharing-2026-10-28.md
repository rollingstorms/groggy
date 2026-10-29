# Current read payload sharing — 2026-10-28

Existing priority components: 21.2, AR7.5 and measured capture work in 32.2.
The preceding query-capture receipt measured 138.665 ms to capture 100k nodes,
including a full new materialization on each independent current capture.

## Implemented

GraphSpace now keeps one weak reference to the current frozen ReadData payload.
Graph.pin_current_state reuses it while readers hold it; GraphView, query captures,
explicit Subgraph capture and DurableGraph use the existing shared path. There is
no new view type, graph clone, automatic index build, checkpoint or history pin.

All active-set/column-mapping mutators invalidate reuse independently of topology
generation. This matters for manual property edits at the same accepted head.
Endpoint revisions and replacements of physical column mappings invalidate too.
The key additionally checks accepted state ID and direction. Crate-private mutable
pool access invalidates before handing out the borrow; the raw snapshot validator
remains uncached. Failed public input validation does not discard a valid cache.

Only ReadData is shared. Every capture receives fresh branch/write coordinates
and the latest retained group catalog, including group installation without an
accepted graph delta. Independent captures retain separate capture identity;
clones/derived selections retain their original identity. The full-suite table
writeback regression caught the initial conflation of identity and shared data;
the implementation preserves the existing independent-export rejection. No test
or writeback restriction was weakened.

## Evidence

Eight added native cases verify weak lifetime, shared data/separate identity,
same-head branches, dirty manual properties, unsets, rewiring, membership removal,
failed-input preservation, corruption rejection/recovery, physical compaction,
group catalog freshness, trusted pool mutation and manual acceptance identity.
Five Python cases cover warm current captures across mixed bulk writes, unsets,
endpoint/node changes, branch-bound writeback, compaction/purge and durable reopen.
Existing table-export tests cover independent versus same-capture writeback.

Native final gate: **1,441 passed / 56 ignored**; strict all-target core Clippy.
The rebuilt/staged extension passes **2,314 root Python cases / 1 skipped** and
**226 executable documentation examples**. The owner’s separate algorithm
benchmark and nested historical surveys remain outside this root contract run.
Three pre-existing FFI warnings remain. Scoped formatting, Python syntax and
diff checks pass; unrelated owner benchmark/review changes are excluded.
Commands use offline locked Cargo and `CARGO_INCREMENTAL=0
CARGO_TARGET_DIR=python-groggy/target`; `cargo test`, strict all-target Clippy,
`cargo bench --bench current_read_reuse` and the full-ABI CPython 3.9 release FFI
build. In-tree extension staging uses atomic replacement. No external packages.

## Release measurements

`benches/current_read_reuse.rs` checks pointer sharing and old/new values. The
fixture has one score column plus normal built-in node attributes, no edges,
and one manual property change following an explicitly accepted bulk fixture.
Five samples each perform 10,000 shared captures. A separate cold capture follows
mutation and another follows releasing all readers. The recorded executable was
rerun after compilation to reduce CPU contention. Raw evidence:
`notes/benchmarks/current-read-reuse-2026-10-28.json`.

| Nodes | Initial cold capture | Shared capture, median | First capture after property change | After last reader release |
| --- | --- | --- | --- | --- |
| 1,000 | 0.352 ms | 122.633 ns | 0.384 ms | 0.350 ms |
| 100,000 | 93.910 ms | 109.158 ns | 88.768 ms | 91.040 ms |

Shared capture cost is independent of the full payload copy on these fixtures;
these native local numbers are not an FFI budget or production latency claim.
Cold timing differences from the preceding receipt are not attributed to reuse.
The mutation times are single observations, not mutation-throughput evidence.
Mutation invalidation takes constant work and never reconstructs a read payload.

## Remaining work

First/invalidated capture still owns a full logical-state copy. Holding multiple
readers across changes retains each payload until those readers release it.
The cache adds only weak metadata, not an additional retention root. Sparse
adjacency allocation in cold captures is a measured follow-up candidate.
Legacy connected Subgraph defaults still read the live owner; their immutable
migration stays in 26.1/AR4/AR7.5, and legacy filter migration in 21.2. Compound
topology/custom effects remain 09.4b.2. No new rows or parent closure:
**211 checked / 57 unchecked**.
