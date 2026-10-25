# Checked nested sampler execution bundle — 2026-10-24

Closes **09.4b.2ad / 09.4b.2ae / 09.4b.2af**, extending **30.1 / AR7.8**.
The explicit open for-each child recorded in the preceding bundle is complete;
two related implementation slices make output cardinality/owned frames separately
reviewable. This continues the requested larger cadence without adding a policy
approval gate. Broader execution, standalone sampler transaction and pinned-view
parents remain open.

## Recursive child effects and property boundary (09.4b.2ad)

ForEachSubgraphStep advertises property-atomic support exactly when every child
step does. Nested loops/for-each recurse through their own capability checks;
unsupported descendants reject during atomic algorithm construction even for
empty input arrays. Direct execution with an existing staging buffer repeats the
capability guard. Unsupported-capability fixtures now use the still-unreviewed
temporal.diff_nodes family.

All items share the enclosing context/property buffer. Input scopes get fresh
variables, while later property loads see earlier persistent staged assignments.
Overlapping items therefore accumulate updates in input order and accept once at
the end of the outer algorithm. Child failures/cancellation abort that algorithm
and discard its pending properties. Standalone sampler pipelines still reject
atomic=True, and ordinary non-atomic persistent writes retain their incremental
failure behavior. No per-item commit/checkpoint or owner/history clone is added.

## Complete child selector results (09.4b.2ae)

Validate every input selector/owner before executing any child. Each child body
runs inside its selector. Revalidate active input topology after the body and
require explicit child outputs to stay within that item, with the same owner and
valid endpoint closure. Keep all explicit output selectors, in item order then
child-array order, including duplicates. An explicit empty array means zero
results. Absent __sample_output__ retains the input once; a wrong-typed variable
errors. Source/target aliasing installs only after successful traversal.

This deliberately fixes historical first-result truncation and fallback for an
explicit empty array. Outputs remain legacy owner-connected selectors; immutable
value/source-pin migration is not implied by independently owned memberships.

## Private owned-output frames (09.4b.2af)

Move prior context outputs aside while running the body and collect each child
step's outputs. Node/edge attribute columns merge by ID across steps/items and
then with prior columns at the same key, with later assignments winning and final
rows sorted. Untouched prior rows/unrelated outputs survive. Aggregate touched
columns once per frame with ordered maps rather than repeatedly concatenate and
sort a growing vector after each item.

Retain non-column payloads under for_each.<target>.<item_index>.<original_key>,
including histograms from every item. Nested frames add enclosing paths. Reusing
a target/item/output path replaces its prior payload; this is current result
ownership, not every loop iteration's history. Original entity-column keys remain
available for ordinary profile consumers. Frames do not simulate a transient
graph: persist=False property loads still read owner values.

On body errors, cancellation, invalid selector outputs or output-type conflicts,
discard frame-local owned results and restore the prior context outputs/target
array. The enclosing atomic algorithm owns property rollback; callers must not
catch a child failure and continue the same staging buffer. Python validation now
recognizes property attachments nested in loops/for-each instead of issuing a
false missing-output warning.

## Cost and remaining work

Work includes outer selected-topology preparation, per-item fresh variables and
liveness checks, flattened output membership storage, merged touched columns and
per-item non-column payloads. Overlapping selectors still duplicate memberships.
Retaining every result uses more memory than a truncating executor; deduplication
is only for entity-column IDs, not selector arrays. Sorting/merging and large
kernels do not provide bounded cancellation latency. No new FFI business logic is
introduced. Remaining temporal/path certification, optimizer effects, native
storage/default escapes, standalone sampler transaction migration and pinned
views/provenance remain under their existing parent entries.

## Verification and bounded release receipt

Final gates: **975 native tests passed, 52 ignored; 1,532 root Python API tests
passed, 1 skipped; 68 executable documentation examples passed; strict all-target
core Clippy passed.** The focused foreach/sampler/atomic Python run passed 129
cases before the full final gate. Native regressions cover flatten/empty/fallback,
recursive capability checks, cancellation restoring prior outputs, wrong-typed
and escaping child results, all-owner validation before children, and owned-column
last-write/prior-row preservation. Python fixtures cover shared overlapping
properties, transient owner reads, per-item histograms, nested loops and both
public lowering paths with/without optimization. Builds used offline locked
dependencies and in-tree extension artifacts; root Python excludes the owner
benchmark test and separate legacy modules survey.

API/examples: `../../docs/api/builder-foreach.md`.
Release harness: `benches/foreach_sampler_steps.rs`. Six modes at 1,000/10,000 nodes
and edges, five observations each: flat/empty selector results, persistent and
transient overlapping properties, item histograms and nested property bodies.
Fixture/spec preparation and validation excluded; execution/frame collection and
zero/one checked publication included. Validate full cardinality/order, merged
values, every histogram and publication. No matched speedup, RSS/FFI/allocation
or production-scale claim.

Raw receipt: `../benchmarks/foreach-sampler-bundle-2026-10-24.json`.

| Workload | Nodes (= edges) | Median ms, five observations |
| --- | ---: | ---: |
| flat_selectors | 1,000 | 6.074 |
| empty_selectors | 1,000 | 2.564 |
| persistent_overlap | 1,000 | 3.678 |
| transient_overlap | 1,000 | 2.885 |
| item_histograms | 1,000 | 2.333 |
| nested_properties | 1,000 | 6.739 |
| flat_selectors | 10,000 | 43.046 |
| empty_selectors | 10,000 | 22.832 |
| persistent_overlap | 10,000 | 37.753 |
| transient_overlap | 10,000 | 29.802 |
| item_histograms | 10,000 | 23.742 |
| nested_properties | 10,000 | 70.911 |
