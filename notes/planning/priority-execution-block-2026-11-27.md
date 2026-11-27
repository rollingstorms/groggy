# Priority execution block — 2026-11-27

Seven finite slices are complete within the existing roadmap. This extends an
already substantial engine; it does not redefine the original graph contract or
replace the fixed checklist. Thirteen additional native step implementations now
share memory/durable execution. Two normalization options and a reusable CI
evidence collector are complete too.

## Completed slices

| Slice | Completed scope | Code source |
| --- | --- | --- |
| [CI evidence](release-ci-evidence-2026-11-27.md) | Read-only matrix/artifact/source validation; 17 tooling tests | `057e844d` |
| [Edge transformation](portable-edge-weight-scale-2026-11-27.md) | Selected scaling, explicit factors, checked finite results | `a8022325` |
| [Community/decay](portable-community-and-decay-2026-11-27.md) | Three community helpers and numerical decay; map/column/edge layouts | `fb969e53` |
| [Normalization](checked-normalize-minmean-2026-11-27.md) | Signed minimum/mean modes, checked lowering and strict explicit control/result checks | `5155a878`, `1ffdf5eb` |
| [Collections/foreach](portable-read-collections-2026-11-27.md) | Four collection producers, immutable portable arrays, recursively prepared fixed children | `372568a5` |
| [Path primitives](portable-path-primitives-2026-11-27.md) | Existing shortest-map, Yen and seeded-walk kernels on workers | `5f19e15e` |
| [Membership cost](shared-collection-membership-2026-11-27.md) | Immutable membership sharing and checked empty-body frame elimination | `57cf0129` |

The normalization review also closed silent defaults for malformed explicit controls,
validated numeric controls before zero-iteration bodies and rejected nonfinite
clipped outputs. Conventional unbounded defaults and valid alias precedence remain.

No algorithms moved into FFI and no mutable scratch owner is constructed on a
worker. Existing direction, parallel/loop incidence, typed values, precision,
empty input, tie and RNG definitions remain. Invalid controls/nonfinite results
reject before replacing outputs or publishing; failed atomic programs preserve
journal bytes, properties and ID allocation. Computation alone creates no graph
version. Attachments still publish one checked operation with explicit per-change
effective times or the established engine-clock default.

## Qualification and cost

Final local source `1ffdf5eb5e15abcd5fdd564634a2bb826215830a` passes **1,716 native tests**
(54 ignored), **3,753 packaged Python tests** (one skipped, 23 expected warnings)
and **306 executable documentation examples**. Strict all-target Clippy, formatting,
whitespace, Python 3.8 syntax, 62 packaged Python/stub byte comparisons and all
500 production-source pins pass. Compared with the preceding numerical-safety
runtime this adds 21 native tests, 156 Python cases and four examples. The 17
collector tooling tests are separate from the maintained root Python count.

[Final local receipt](../benchmarks/normalization-final-controls-local-qualification-2026-11-27.json)
and [wheel receipt](../benchmarks/normalization-final-controls-wheel-qualification-2026-11-27.json).
Each component note retains its own source/artifact boundary. Five older local
receipt links to the prior alias performance report had a date-shifted filename;
those links now point to the actual 2026-11-25 report. Gate/source/artifact hashes
are unchanged by that metadata correction.

The first collection port exposed a small readonly overhead. The follow-up shares
immutable membership and avoids frames for empty bodies after validating all
selectors. Matched readonly operation medians improve 11–32% across measured
100/1k/10k/100k directed paths; nonempty write medians range about −0.2% to +2.8%.
For the larger corpus, two fresh processes per runtime show whole-process peak
RSS **788–792 MiB before, 734–739 MiB after**. Memory includes imports, setup,
warmups and both workloads; it is not an allocation or per-operation RSS count.
The [performance receipt](../benchmarks/shared-collection-membership-performance-2026-11-27.json)
retains raw ABBA samples, process timing output and wheel/source/script hashes.
There is no broad speed, production budget or billion-scale claim.

Manual [membership CI run 46078071480](https://github.com/rollingstorms/groggy/actions/runs/46078071480)
qualifies source `57cf0129`; [final normalization CI run 46079596058](https://github.com/rollingstorms/groggy/actions/runs/46079596058)
qualifies `1ffdf5eb`. Final matrix evidence is recorded separately when complete;
local gates alone do not certify every platform. Superseded control run 46079465175
was cancelled after the clipped-result guard changed source and is not counted
as a successful qualification.

## Existing priorities after this block

The stable inventory is **217 checked / 51 open**, comprising 42 expanded-core
rows and nine deferred rows. No broad parent is closed merely because one of
its components is complete; no new stable row was added. Completed components
are linked directly from 09.4b.2, 30.1 and AR7.8 and the workbook appendix.

1. Prepare the seven retained-history step producers from explicit immutable
   accepted lineage, preserving Context-selected heads, materialization/owner
   identity and provisional-ID guards. Reuse detached recorded-read plans rather
   than copying a mutable owner or inventing history from current membership.
2. Finish edge-aware register domains and remaining public facade/IR/alternate
   plan lowering. Min/mean normalization and native collections/foreach are
   complete; the standalone sampler facade remains a separate entry point.
3. Complete remaining fixed-child/provisional topology domains through canonical
   publication, with explicit unsupported combinations and recovery/failure tests.
4. Continue existing cross-option/reference/scale certification and module/public
   compatibility gates; retain original provenance and concurrency/service scope.

A14 naming/cache/run-history and A15 production workload/budget choices remain
owner questions. Streaming/custom effects retain their documented boundary.
These are remaining requirements of existing parents, not newly invented tasks.
