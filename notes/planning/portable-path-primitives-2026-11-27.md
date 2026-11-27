# Portable selected path primitives — 2026-11-27

Finite component of 03.2, 09.4b.2, 28.16, 30.1 and AR7.8. All three existing path
step implementations now delegate through the sealed immutable read boundary.
Stable checklist counts remain 217 checked / 51 open; no new row is added.

- [x] Share shortest-path maps, Yen paths and seeded random walks on workers.
- [x] Preserve selected pending geometry, staged weights and exact endpoint domains.
- [x] Preserve integer BFS/f64 Dijkstra output, loopless node sequences, minimum
  parallel-arc path costs, per-edge walk multiplicity and local seeded RNG.
- [x] Keep finite nonnegative weight checks, cost overflow, cancellation and old outputs.
- [x] Compare independent distances/paths, actual workers, memory and reopening.
- [x] Qualify Python examples, zero-work validation, fixed children and rollback.

Only input access and owner adapters changed; the path algorithms are unchanged.
A missing weight remains unit cost while explicit null/nonnumeric/negative values
reject. k=0 and length=0 still validate IDs and weights. Fixed children cannot
read outside their selected membership. Computation without attachment creates no
history. The existing exhaustive four-node Yen oracle and native weight/RNG/error
tests remain passing.

`tests/durable_path_primitives.rs` compares four direction/weight domains against
known path/distance expectations, actual worker execution, memory and recovery;
its rejection cases preserve the journal and allocated IDs. Twenty Python cases
cover public outputs/types, pending weights, selected frames, zero/trivial work,
recovery, readonly execution and failure. The executable public contract is in
[durable pipelines](../../docs/api/durable-pipelines.md).

Local gates: **1,714 Rust tests** (54 ignored), **3,706 Python tests** (one skipped,
23 expected warnings), **306 documentation examples**, strict all-target Clippy,
formatting, Python 3.8 syntax and 62 packaged Python/stub byte checks. All 500
production sources match runtime `5f19e15e`.
[Local receipt](../benchmarks/portable-paths-local-qualification-2026-11-27.json),
[wheel receipt](../benchmarks/portable-paths-wheel-qualification-2026-11-27.json).
No new path performance/scale or cross-platform claim.

Remaining native owner-history readers are the seven temporal snapshot/diff/window/
lifecycle producers. Porting them needs explicit immutable retained-history inputs,
including Context-selected heads and owner/materialization identity; a writable
scratch owner or a fabricated current-state history is not a substitute. Remaining
frontends, alternate plans, callbacks/streaming, topology domains and A14 decisions
stay in their existing parents.
