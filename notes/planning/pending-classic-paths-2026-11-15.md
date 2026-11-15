# Classic path algorithms after pending topology — 2026-11-15

## Completed fixed four-handle family

BFS, DFS, Dijkstra and A* registry wrappers now accept supported pending topology
in outer atomic pipelines. A shared native AlgorithmFrame keeps the original
continuation separate from a private pending read pin, matching the step
interpreter's separation of execution ownership and read state. Later builders
can consume staged path columns without attempting to write from an unpublished
snapshot. The outer coordinator remains the sole publisher/projector.

The wrappers retain their existing kernels, parameters, result types, profiling
keys and source/goal presence semantics. Captured weights/heuristics use the same
borrowed scalar validators, including minimum parallel-edge weights, invalid
passthrough values, nonnegative/finite constraints and heuristic consistency.
Direction comes from the selected source. A shared bulk endpoint reader chooses
frozen data or one live owner guard; it avoids per-edge owner locking. Ordinary
standalone/property-only execution retains its live preparation path, without a
new snapshot copy. No FFI business logic or storage format changes.

Result writes retain existing update semantics: only reachable/path result rows
are assigned; other stored values are not implicitly cleared. The remaining nine
classic wrappers retain pending-topology preflight guards. Their existing captured
algorithm-table readers remain supported.

## Qualification

Locked/offline native suite: **1,604 passed / 54 ignored / zero failed**, 223 suite
results including doctests. Strict all-target Clippy, Rust formatting and whitespace
checks pass. The five-test classic native target additionally passes with the old
live CSR warmed before pending revisions. The new endpoint regression reads a
frozen inactive edge while the current owner is mutably borrowed, then proves the
live source rejects that edge. In-tree focused Python: **37 passed**.

A fresh root macOS arm64 CPython 3.9 wheel passes isolated `-I -S` smoke, all 105
declared exports/version agreement and archive/durable application workflows.
Against that unpacked artifact: **3,089 maintained Python tests passed / one skipped /
14 established warnings**, plus **283 executable documentation examples**.
Receipt: `notes/benchmarks/pending-path-wheel-qualification-2026-11-15.json`.
No wheel or external package was installed. Three established binding warnings
remain; Black/isort/pre-commit remain unavailable. Cross-platform requalification
of this new source is still a release gate, separate from the older candidate.

Twenty-four new Python regressions cover directed/undirected sparse selected
scopes, provisional start markers/goals, revisions/cascades, loops/parallel edges,
staged weights, later builder reloads, consecutive classic algorithms, invalid
weights/heuristics, late failure and no consumed IDs. Existing independent finite
path references, option/domain/cancellation tests and all thirteen classic
property-only pipelines remain maintained. Initial compilation needed trait
imports moved to their actual caller/test scopes. The first benchmark attempt
registered builders but omitted local path factories; the harness now registers
both before constructing its pipeline. No runtime fallback was introduced.

## Current native workload measurements

Five observations each, native release, milliseconds (medians):

| Chain edges | Algorithm | Property-only chain | Pending ring/provisional goal |
| --- | --- | --- | --- |
| 100 | BFS | 0.541 | 0.713 |
| 100 | DFS | 0.484 | 0.634 |
| 100 | Dijkstra | 0.429 | 0.605 |
| 100 | A* | 0.399 | 0.450 |
| 1,000 | BFS | 2.898 | 5.434 |
| 1,000 | DFS | 2.456 | 4.813 |
| 1,000 | Dijkstra | 2.509 | 4.847 |
| 1,000 | A* | 2.998 | 4.275 |

Both stage weights=2 and publish once. Topology cases replace the last node with a
provisional goal reached from node 0 and rescue the last edge to form a ring;
property-only cases retain the chain. These have different graphs/answers and are
**not matched before/after performance comparisons**. Timings include preflight,
pending preparation, kernels and publication; exclude fixtures, initial capture,
algorithm construction, assertions and compilation. One-process observations;
no allocation/RSS, FFI or general scale claim.

Receipt: `notes/benchmarks/pending-classic-paths-2026-11-15.json`; reproduce through
atomic_builder_properties `--classic-pending-paths` with GROGGY_ATOMIC_BUILDER_PROFILE.

## Existing roadmap and remaining boundaries

Advances 09.4b.2, 03.2/AR2, 30.1, AR7.8 and conformance under 01.2/19.2/28.16.
No added checklist rows or premature broad closures: **215 checked / 53 unchecked**,
including **44 unchecked release rows**. Remaining nearby work: the other nine
classic wrappers' adjacency/value sources; general topology loops and alternate
plans/lowering; the provisional lifecycle question; release/service/scale work.
Wide pending captures still copy selected logical properties, so property-heavy
workloads need their own measurements before a broader performance claim.
Unrelated owner files remain untouched.
