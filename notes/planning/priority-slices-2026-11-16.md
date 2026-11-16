# Priority slices — 2026-11-16

Continuation tracked from 01:48 UTC against the requested 90-minute window.
Branch: groggy-roadmap-2026-10-20. Five coherent implementation slices are
committed and pushed; the native runtime source is e517e639.

| Commit | Completed slice |
| --- | --- |
| 6523a66a | Owned pending effects separate payload preparation from publication; complete coordinate validation, no preparation-side allocation/publication, and range-based final projection. |
| a6ba26b9 | Pending captures avoid cloning accepted values that staged assignments replace; unchanged rows and earlier immutable captures remain exact. |
| c60b1824 | Existing edges can be rewired to explicit batch-global new-node references through shared parsing and selected-node validation, including cascade rescue and temporal recovery. |
| f131e8f9 | Pending read materialization consumes a checked immutable base and explicit membership; live-owner validation stays in its adapter. |
| e517e639 | Numerical backend slices retain owned snapshots; unavailable transfers reject, live guards remain available, and generic aligned allocation initializes valid custom numeric zeros. |

Details: [owned effects](owned-pipeline-effects-2026-11-16.md),
[value copying](pending-value-copying-2026-11-16.md),
[new endpoint references](staged-revision-endpoints-2026-11-16.md),
[immutable materialization](immutable-pending-materialization-2026-11-16.md), and
[numerical ownership](numerical-memory-ownership-2026-11-16.md).

The fixed inventory remains **268 rows: 215 checked / 53 unchecked**, including
**44 open expanded-release rows and nine deferred rows**. These five bounded slices
are complete; their overlapping parents retain broader exit criteria. No rows were
added. The longest execution rows now summarize current work and link the ledger,
rather than repeatedly appending old “remaining” statements that are now completed.

## Final local qualification at e517e639

- Locked/offline native tests: **1,629 passed / 54 ignored / zero failed**,
  225 suite results, including the new numerical memory public API doctest.
- Separate serial process-kill checkpoint recovery: **one passed** on macOS.
- Strict all-target Clippy, Rust formatting and whitespace checks pass.
- Fresh root macOS arm64 CPython 3.9 wheel: isolated imports, version agreement,
  **105 exports**, archive and journal application workflows, **3,305 maintained
  Python tests / one skipped / 14 established warnings**, and **289 executable
  Python documentation examples** all pass.
- Fresh nested wheel: **36 ownership/workflow tests**, identical native extension
  and all **61 Python/stub files** compared with the fully tested root artifact.
- Three established FFI warnings remain; Black/isort/pre-commit are unavailable
  in the existing environment. No wheel/dependency was installed.

Receipts:
`../benchmarks/priority-slices-local-qualification-2026-11-16.json`,
`priority-slices-wheel-qualification-2026-11-16.json`, and
`priority-slices-nested-wheel-qualification-2026-11-16.json`.
Root wheel SHA256: c6f812da41857d96ad5ee1c5975b191e2b1c49889b50200e6efac55df75ecb68.
Nested wheel SHA256: 19fb1a388baf76e8ec5d0b0412092107228c0e0633b3d63210a893d09d19da1c.
Shared native module SHA256:
dff07f13602008c8f2e2978458774616db558063298587eee889c8d30fda3e51.

## Performance evidence and limits

The matched replacement workload uses three fresh processes before/after, five
observations per case. At 1,000 entities with two 1,024-value vector columns
replaced, requested allocation volume drops **37.45 → 20.46 MB** for nodes and
**38.15 → 21.75 MB** for edges. Zero-replacement controls allocate exactly the same
bytes. Peak requested live bytes improve much less; timing variation in the
unchanged controls prevents attributing the full wall-time ratios to this fix.
This is not an RSS or production-scale guarantee. See the value-copying note and
its six raw before/after receipts for exact source hashes and measurement scope.

The native memory ownership corpus runs after local builds/tests stop, with three
fresh processes and seven observations each after warmup. At one million f64
values, the explicit owned backend snapshot requests **8,000,024 bytes in two
allocations**, with a median **300.084 microseconds**. Live read-guard endpoint
access requests **zero bytes** (0.042 microseconds); explicit same-size refresh
also requests **zero bytes** (245.500 microseconds). These operations have different
ownership semantics; this is not a before/after speed comparison. Timing includes
counter/checksum overhead and output drop, excludes fixture/build work, and does
not measure whole-process RSS. All full outputs are compared outside timing.
Raw receipts: `../benchmarks/numerical-memory-views-2026-11-16-{1,2,3}.json`.

## Cross-platform qualification

Manual run, pinned to e517e639:
https://github.com/rollingstorms/groggy/actions/runs/43792347879

Started at 02:28 UTC. Native gates and extracted-source rebuild have passed;
19 configured wheel combinations are still completing. The publish job cannot
run for this manual dispatch. Final conclusions and artifact hashes will be
recorded separately when available; the older be6512d1 platform receipt remains
an earlier qualification rather than being relabeled as this source.

## Remaining priority work

1. **Durable invocation/dispatch:** owned effects and immutable selected
   materialization are independent of live owners, but Algorithm/StepScope still
   requires owner-bound Subgraph invocation. Finish that adapter through existing
   journal/schema/coordinator acceptance, with immutable outputs and preserved
   history/allocation authority. Do not lend a writable durable owner or clone a
   scratch graph. The concrete invocation map also records the existing pipeline
   cascade versus strict view-writer distinction, public compatibility adapters
   and a bounded first whole-program parity target.
   [Boundary context](durable-pipeline-boundary-2026-11-16.md).
2. **Remaining execution routes:** fixed child/provisional entity topology,
   alternate batch plans and public IR lowering, streaming and unsupported custom
   effects retain their guards. Sequential root topology loops, all thirteen
   classic wrappers, staged read/table/property work and new-node endpoints are
   completed within their stated contracts.
3. **Unanswered history policy:** provisional-node lifecycle/window queries remain
   rejected while the existing owner question is pending. No retained history is
   fabricated; accepted-history reads remain distinct from pending geometry.
4. **Expanded release scope:** property/schema/provenance, storage evolution and
   recovery memory, numerical/algorithm and scale certification, remaining
   compatibility consolidation and service/concurrency requirements retain their
   existing parent rows. Numerical memory safety repairs do not certify every
   matrix storage/backend/decomposition domain.

Earlier execution foundations and detailed receipts remain under the
[expanded scope ledger](release-060-expanded-scope-2026-11-15.md),
[previous priority hour](priority-hour-pending-execution-2026-11-16.md), and
[preceding pending pass](priority-pass-pending-execution-2026-11-15.md).
Owner development/benchmark and untracked review/notebook files remain untouched.
No tag, package publication, merge or PR was created.
