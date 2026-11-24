# Shared numeric bindings — 2026-11-24

Continuation of existing performance row 32.2 and execution rows 30.1/AR7.8.
The prior numeric-lowering milestone measured 1.30–3.11× the equivalent checked
native loop time. This slice targets its repeated register-map copying.

## Finite completion boundary

- Share immutable variable values privately while preserving public `StepValue`,
  getter, setter, mutable accessor and iterator signatures.
- Preserve independent mutation through copy on write for node maps and columns;
  cloned frames and aliases must retain their earlier values.
- Share exact-membership numeric register loads/stores and carried values.
  Keep subset projections, numeric validation and column output representation.
- Allocate one shared initial zero map for nonempty register files; zero-slot
  plans must not allocate an unused node map.
- Run the same recurrence benchmark as the prior committed baseline, qualify the
  native and packaged Python surfaces, commit the implementation, and record
  measured gains and residual costs without adding roadmap rows.

## Results

Implemented and committed in `0ff2417776cde8b11d012f549038841c3926d3a1`.

- [x] Private `Arc<StepValue>` bindings preserve every public value variant and
  accessor/iterator signature. Typed setters install immutable values; map and
  column mutable accessors copy on write only after validating the requested kind.
- [x] Exact-membership numeric register loads/stores and carried assignments share
  values. Subsets still project and validate; column outputs keep their type.
- [x] Nonempty register files share one zero map. Empty register files allocate
  no unused zero map. Native arithmetic and backend publication are unchanged.
- [x] Four new native tests verify physical sharing, independent mutation,
  retained values after rebinding/drop, wrong-kind errors without detaching,
  subset projection, invalid rows and column output representation.
- [x] Full native gate: **1,658 passed, 54 ignored**, 227 suite results including
  ten passing doctests. The seven existing numeric integration cases pass.
- [x] Fresh wheel gate: **3,432 Python tests passed, one skipped**, fourteen
  established warnings; **293 documentation examples** pass. All 62 packaged
  Python/stub files match; Python 3.8 syntax checks pass.
- [x] Strict all-target Clippy, formatting and diff checks pass. A metadata-only
  Rust assertion verifies public `StepVariables: Send + Sync`.

All 500 production source files match the committed Git archive byte for byte.
Source tree SHA-256:
`44abcf4a0f83e33c59d1bd4f6148fd78f6ff1247bbceb672f359fd658043ec7d`.
[Local qualification](../benchmarks/shared-bindings-local-qualification-2026-11-24.json)
and [wheel qualification](../benchmarks/shared-bindings-wheel-qualification-2026-11-24.json)
retain the artifact/source identities and gate-log hashes. Earlier platform
qualification remains historical; no fresh platform-wide claim is made here.

## Same-workload performance comparison

The benchmark source is byte-identical to runtime `4fa2be08`:
`4996965166c53c756fd4eeeaa39ff5a9cdf0e201844cd9d5b5a12a3e848c6ce4`.
Three independent processes supply seven observations per case each, with warmed
plan caches, alternating memory execution order and setup/compilation excluded.
Samples were collected after local builds and test gates ended. All computed
answers match. Memory times include final atomic publication; durable preparation
and synced acceptance remain separately timed.

Median milliseconds across 21 observations per case:

| Nodes | Iterations | Prior lowered | Shared lowered | Current native | Durable prepare | Durable accept |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 0.024 | 0.024 | 0.022 | 0.024 | 5.692 |
| 1,000 | 10 | 3.998 | 2.634 | 2.371 | 2.399 | 7.262 |
| 10,000 | 10 | 37.782 | 23.059 | 20.768 | 20.430 | 22.856 |
| 1 | 100 | 0.115 | 0.095 | 0.071 | 0.102 | 4.883 |
| 1,000 | 100 | 24.627 | 9.935 | 7.906 | 9.885 | 6.433 |
| 10,000 | 100 | 250.611 | 105.405 | 83.223 | 103.070 | 24.498 |

Lowered execution takes **34–39% less time** in the larger ten-iteration cases
and **58–60% less time** in the larger hundred-iteration cases. Their current
lowered/native ratios are about **1.11×** and **1.26×**, respectively. The
one-node ten-iteration lowered case is essentially flat (+1.9%); the native
comparison median rose 19.9% there, about four microseconds, and 11.1% in the
1,000-node ten-iteration case. Native medians for the larger hundred-iteration
cases were within -2.3% to -0.1% of the prior run. The full aggregate retains
before/after values and ranges; this is not a blanket speedup claim.

[Aggregate/raw sample identities](../benchmarks/shared-bindings-2026-11-24-summary.json)
include all timings and hashes. OS cache, power state and unrelated background
activity were uncontrolled. Physical-sharing tests prove the removed deep
copies; timing does not isolate every cause or measure allocator bytes/RSS.

## Remaining existing scope and trade-offs

A first mutation of an aliased map or column copies that whole value. Arithmetic
results still allocate maps; membership/numeric checks still scan rows; scalar
bindings now use a small reference-counted allocation. These are explicit costs,
not general constant-time computation or a production latency/memory guarantee.
Acceptance and sync costs are unchanged by this implementation.

Performance budgets and broader scaling remain under existing 32.2/A15. Remaining
neighbor/in-place/fused lowering and portable ordinary composition retain their
existing execution rows. No new policy, roadmap row or public value variant was
introduced. The global inventory remains **217 checked / 51 open**.
