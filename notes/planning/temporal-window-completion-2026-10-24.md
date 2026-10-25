# Temporal primitive family complete — 2026-10-24

Closes the four remaining existing temporal primitives under **30.1 / AR7.8**:
window_aggregate, filter, mark_changed_nodes and window. The fixed eight-primitive
inventory is now **8 complete / 0 remaining** (remaining six → four → zero).
Existing checklist requirements and evidence are updated; no checkbox rows or new
capability workstreams were added. Broader pipeline/intermediate/pin, effective-
time reconstruction and global index lifecycle requirements remain as written.

## Shared retained branch index

Add TemporalIndex::from_history_at and Graph::build_temporal_index_at, matching
snapshot reconstruction's first-parent lineage. Graph::build_temporal_index now
captures its accepted current head, excluding sibling branches. Scoped indexes
hold the same private source token used by snapshots plus a captured head. Supplied
indexes must match owner/head; stale/foreign/unscoped/wrong-typed values error and
are not replaced. A matching older Context head can still use its immutable index.
Legacy direct from_history remains whole-forest and is not accepted by the checked
helpers. No storage schema or global cache lifecycle is changed.

When no index is supplied, helpers derive a local ephemeral index, preserving the
convenience read path without installing it into variables or graph caches.
Index build errors cannot publish partial state or leave a cached index behind.
Actual retained records drive range iteration, so large/sparse numeric gaps do not
cause dense loops or snapshot construction for nonexistent IDs. Unknown bounds
and sibling-branch coordinates reject. Zero denotes the empty root. State IDs
must increase on the captured first-parent lineage; invalid cycles/nonmonotonic
lineages reject rather than produce misleading timelines.

Activation interval lists preserve repeated activation/deactivation, with deletion
exclusive. Existing lifetime-range metadata remains an envelope; existence uses
actual intervals. Restoration with no property assignments cannot inherit stale
values through the index from its previous activation. Point lookups choose the
last assignment within a state, and history slices use binary-searched bounds.
Endpoint rewiring still rejects explicitly under existing 04.2/13.3.

## Four completed helpers

- Window aggregation uses only actual property assignment records in inclusive
  retained-state bounds. Remove legacy index_ placeholder guessing, mutable-pool
  resolution, numeric-ID snapshot fallback and silent dropping of nonnumeric
  history. Count includes null/unsupported property kinds without conversion and
  returns zero for no changes. First/last preserve the selected supported logical
  value or error if its type cannot be represented; empty returns None. Shared
  checked reductions handle finite numeric sum/mean/min/max, exact large integer
  cancellation and finite means even when sum overflows. Numeric null/Boolean/
  text/collection/nonfinite inputs reject. Empty numeric windows return None.
- Lifecycle filtering uses the captured lineage and original creation state,
  validated point/range coordinates, and interval-based existence. Preserve the
  old modified_in_range meaning of any node event, including activation/deletion.
- Mark-changed uses exact created/deleted/property event lists. Multiple is a
  summary rather than a wildcard for every event kind. ChangedEntities add/merge
  combine types instead of overwriting previous events. Context.changed_entities
  validates the captured head/range and iterates retained records.
- Temporal window marks selected existence in inclusive retained-state bounds;
  activation includes its creation state and excludes its deletion state, so
  restoration gaps stay inactive. Register the existing native step for specs.

All four support atomic property runs and produce dense selected 0/1 maps or
aggregate maps. Historical inactive IDs may be inspected when known to the
retained lineage; explicit property writes still validate active owner targets.
Output maps are prepared privately, cancellation checks precede installation,
and failures preserve prior variables and discard earlier staged attachments.
Historical reads ignore staged current-property edits and create no deltas.

Existing constructors remain. Aggregate/mark gain with_window(start,end) and
paired optional spec bounds; absent explicit bounds require a native Context
window. Temporal window requires start/end/output. Coordinates are state IDs,
not seconds/effective-time instants. This does not imply arbitrary time-at-T graph
reconstruction. Unsupported-effect tests now use the existing unimplemented
streaming execution-block mode, retaining meaningful recursive preparation gates
after all temporal primitives became certified.

## Verification

- Full native gate: **1,000 passed / 52 ignored**. Seven new native regressions
  cover scoped provenance/heads, sparse sibling bounds, cancellation, restoration
  gaps, exclusive deletion, literal text/null/checked reductions, wrong/stale/
  foreign indexes and exact mixed event types. Existing temporal tests pass too.
- New Python integration file: **50 passed**. All reducers and empty history,
  persistent/transient outputs, literal index_ text/null, invalid numeric history,
  lifecycle/change masks, all four kernels in one accepted batch, invalid bounds
  and strict paired-coordinate parsing. Combined focused builder/for-each gate:
  **177 passed**.
- Full root Python gate: **1,695 passed / 1 skipped**; **81 executable doc examples**
  passed (three new window examples). Owner benchmark test remains excluded.
- Strict all-target core Clippy passed. In-tree FFI rebuild passed with the same
  three existing unrelated warnings. No FFI/Python business logic was added.

Commands/logs for this pass use cargo test/clippy --offline --locked, with
CARGO_INCREMENTAL=0; the extension is built/loaded from in-tree artifacts. Python
runs use PYTHONPATH=/tmp/groggy-python-check. No external graph library execution,
package installation or wheel reinstall occurred.

## Bounded release receipt and costs

`benches/temporal_window_steps.rs` and
`../benchmarks/temporal-window-bundle-2026-10-24.json` contain **12 workloads ×5
observations**. Two retained states assign one scalar property to every node.
Native optimized timing includes validation, reductions/result construction, and
index build where named. Fixture/variable preparation, result validation, FFI and
publication are excluded. Every history/result and unchanged owner is validated.
Median milliseconds:

| Operation | 1,000 nodes | 10,000 nodes |
| --- | ---: | ---: |
| Scoped index build | 0.5233 | 3.9288 |
| Aggregate, explicit reused index | 0.3157 | 3.0897 |
| Aggregate, local ephemeral index | 0.7859 | 7.9300 |
| Lifecycle filter, reused index | 0.0390 | 0.5368 |
| Exact change mask, reused index | 0.1402 | 1.0905 |
| Existence window, reused index | 0.0877 | 0.6855 |

These expose the cost of deriving an index versus explicit reuse; they are not a
matched speedup claim or whole-engine/FFI/RSS/allocation/production-scale gate.
Index storage scales with retained events/intervals and shares immutable deltas.
Borrowed bounded property slices avoid cloning full histories; changed payloads
still require conversion/copies for owned results. Each default helper builds its
own ephemeral index, so native callers should supply a matching explicit index
for repeated reads. Cancellation checks cover lineage and query loops, but one
commit's processing, sorting, reductions and large clones have no bounded latency
promise.

```sh
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target cargo bench --offline --locked --bench temporal_window_steps --no-run
GROGGY_WINDOW_PROFILE=notes/benchmarks/temporal-window-bundle-2026-10-24.json python-groggy/target/release/deps/temporal_window_steps-8bd4981e622d013b
```

Contract/examples: `../../docs/api/builder-temporal-windows.md`.
