# Complete integer view sampling — 2026-10-29

Existing 26.1 / AR4 / AR7.5 read-facade migration component; no new checklist row.

## Reproduced boundary and implementation

Graph.sample(k) and connected Subgraph.sample(k) delegated to an FFI placeholder
that selected a temporary prefix but discarded it and returned the whole input.
Seven pre-fix Python cases failed: four actual connected/graph zero/small-count
failures and three cases for the missing immutable sampling operation. Oversized
connected requests already returned all nodes and remain supported.

- Subgraph and GraphView now share the existing Algorithm R reservoir kernel with
  pipeline node/edge masks. Business logic lives in Rust; the FFI translates counts,
  seeds and results. The sampler caps output allocation to population size, uses a
  local RNG, and preserves cancellation checks for pipeline callers.
- Seeds use ascending original IDs and do not reseed/consume the shared RNG.
  Unseeded calls remain random. Zero returns empty; oversized counts retain all.
- Results induce only edges in the original selection with both endpoints sampled,
  preserving parallel instances, orientation and loops. No entities or states are
  allocated/published by sampling. Connected input validates full active endpoint
  closure even for zero and oversized counts.
- Immutable results share source capture, effective clocks, readonly/current write
  provenance and retained group catalogs. They survive owner mutation/destruction
  and worker execution. Checked later writes target only the sampled selection and
  return a refreshed view without changing the input.
- Component/subgraph collection sampling now propagates invalid input errors rather
  than dropping failed items. Existing neighborhood collection forwarding inherits
  the fix. Signatures remain unchanged for collections; graph/view integer calls
  add optional explicit seeds. Counts/seeds reject booleans and invalid ranges.
- Subgraph.sample(sampler) keeps pipeline semantics; separate seed keywords reject
  for sampler objects, whose seeds belong in their specification. Temporary sampler
  handles are dropped on failed execution as well as success.

The measured zero-size path originally still sorted IDs/scanned edges. Dedicated
shortcuts now skip that work after live validation; neither shortcut bypasses
invalid scope checks. Immutable GIL release is supported; Rc-connected work stays
under the GIL. No writable owner clone is introduced.

## Verification and bounded costs

Four new native regressions cover direction/parallel/loops/excluded instances,
count boundaries, shared captures/provenance, seeded diversity/RNG independence,
stale and malformed scope, owner-free workers and reservoir cancellation. Eighteen
Python cases cover all public forms, collections, controls, immutable lifetime,
meta sources, explicit temporal clocks, subsequent checked writes and actual
builder-mask/direct sampling parity. Existing nineteen native sampling cases pass.

Full gates: **1,498 native / 54 ignored**, strict all-target core Clippy,
**2,537 root Python / 1 skipped**, **246 executed documentation examples**. The
pending next-slice column-indexing tests and owner's untracked benchmark are
excluded from this slice's Python gate. Release full-ABI extension rebuilt and
atomically staged; three unrelated existing FFI warnings remain. Scoped formatting,
Python syntax and diff checks pass; owner edits remain untouched.

Native release benchmark: 1k/100k chains, five samples per process, fixed seed,
analytic edge checks, construction/initial scope validation excluded and immutable
capture reported separately. Three alternating process pairs compare the correct
sampler before/after only zero-size shortcuts, not the prior incorrect clone.
At 100k, zero medians change **2.352 ms → 0.208 µs connected**, **1.308 ms → 0.083 µs
captured**. Sixteen-node sampling remains **4.609 ms connected / 1.428 ms captured**;
full selection returns share membership. Tiny metadata timings include timer
resolution; no universal FFI, production memory or whole-engine latency claim.
Raw data: `../benchmarks/view-node-sampling-2026-10-29.json`.

API: `../../docs/api/view-node-sampling.md`. Remaining fixed boundary: connected
reads/defaults are still live, and immutable whole sampler/effectful pipelines
still require their original owner/execution migration. **211 checked / 57 open**.
