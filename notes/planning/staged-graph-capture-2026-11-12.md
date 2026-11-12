# Explicit immutable staged graph/table values — 2026-11-12

Existing scopes: 30.1 / AR7.8, 09.4b.2 / AR3. Completed selected pending-state
read component; these broad parents remain open. No new checklist rows.

## Implemented

`graph.capture_staged(output=...)` captures exactly the current execution
selection with staged property assignments, explicit removals and node-removal
cascades. It produces the existing GraphView type. Tables use the existing nine
immutable read primitives and native columnar readers; no second interpreter,
mutable graph clone or new publication mechanism.

Each result is an independent immutable materialization, preserving typed
values, sparse identities, selected edge instances/loops and retained meta-group
source references. Later pending changes and accepted owner changes do not
alter it. Captures survive owner destruction and support worker-thread reads.
Table outputs remain independent owned Python tables. Null remains a present
column, distinct from a missing property.

Native `AtomicStepEffect` conservatively describes unsupported steps, existing
property work, immutable-input reads, staged captures and topology staging.
Existing custom steps retain default property certification behavior. After the
first topology effect, only further removals, explicit staged captures,
immutable graph/table stages and variable aliases may execute. Ordinary
structural/property computations still reject before running the program.
Nested/outer property-only certification remains conservative; no unsupported
loops or custom callbacks are silently admitted.

Local atomic execution and persistence are required. A capture-only run accepts
no delta. Final canonical validation remains authoritative: invalid duplicate,
property or temporal effects reject the run; late projection failures discard
staging and produced outputs and restore earlier Context outputs.

## Provenance and cost

A pending capture is **provisional and read-only**, never an accepted-state pin:
write_state_id is None even when no writes have yet been staged. Its internal
snapshot uses the base state number as projection bookkeeping; this is not a
claim that the base version contains the pending values. Captures have independent
source identities. For accepted write provenance, capture the owner after the
run; effective-time reconstruction remains the separate temporal API.

Selected values/adjacency are materialized. Retained group catalogs share their
immutable payloads, but the graph/property projection is not a lazy overlay or
constant-time snapshot. Cost is proportional to selected data and value sizes,
plus a canonical base capture if no source is already retained. No permanent
cache or implicit per-mutation capture was added. The whole pipeline still runs
on the owner thread; inner capture copying is not interruptible, with cancellation
checked before/after the stage and before acceptance.

The same release binary runs five samples per 1k/100k path graph. Fixtures,
initial source capture, compilation and assertions are excluded:

| Native program | 1,000 nodes | 100,000 nodes |
| --- | ---: | ---: |
| Two removals + publication + surviving selection | 0.426 ms | 38.466 ms |
| Same, plus pending graph capture + node table + emission | 1.177 ms | 193.629 ms |

Medians; raw samples in `../benchmarks/staged-graph-capture-base-2026-11-12.json`
and `../benchmarks/staged-graph-capture-2026-11-12.json`. This measures optional
explicit materialization work, not an implicit mutation slowdown or a before/after
optimization. No FFI, allocation/RSS, vector-heavy or billion-node guarantee.
The earlier removal-only receipt used a different run and retains its own samples.

## Verification and remaining work

- Full Rust gate: **1,555 passed / 54 ignored**; final targeted staged, terminal
  and typed suites cover the final equivalent lint adjustment.
- Strict all-target Clippy passes. Rebuilt/staged release in-tree extension;
  three existing FFI warnings remain.
- Final maintained Python gate: **2,645 passed / 1 skipped / 15 warnings**;
  **264 executable documentation examples** pass. The known PyGraph cyclic-GC
  warning remains 19.2. Final focused native suites pass **13 tests**.
- Tests cover before/after snapshots, late errors/output rollback, cross-domain
  guards, no-op/empty captures, exact wide integers/f64/Null/bool/text, source
  identity/write-pin separation, worker lifetime and retained meta-group sources.

Remaining: additions/endpoint-revision pipeline effects, ordinary algorithms over
staged topology, wider nested/outer effect scopes, high-level DSL/IR lowering,
streaming/cancellation, generated naming/cache/repeatability and broad scale
budgets. A14.1 stays unanswered. Provisional captures do not expand the writable
owner threading contract. **215 checked / 53 open / 268 total**.

Contract: `docs/api/staged-graph-captures.md`.
