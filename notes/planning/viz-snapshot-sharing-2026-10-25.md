# Immutable visualization capture ownership — 2026-10-25

Continuation of the existing 27.1/AR7.10 capture adapter and 28.15 visualization
boundaries. No checklist rows added and no broad parent declared complete.

## Changes

GraphDataSource now owns one Arc<GraphSnapshot> containing entities, types,
direction and a stable process-local identity. Clones share the complete frozen
payload. Counts derive from the stored vectors; duplicate count fields and
duplicate visualization-specific payload getters were removed.

The Python VizAccessor registry previously used the address of its temporary
stack argument as the source key. Distinct captures could receive the same key
and reuse another visualization's server. It now uses the native capture ID.
Clone/move/thread transfer preserves identity; each fresh capture, including an
empty or same-shape capture, receives a distinct identity. This is ephemeral
process identity, not graph state/row identity or a persistent content hash.

Keep one Python `.viz` accessor for repeated `.show()`/`.update()` calls. A fresh
`.viz` property read creates another capture; it does not update an older
capture's server. FFI owns no ID allocation or graph semantics.

## Verification and cost

One native pointer/last-owner case plus four native public-contract cases cover
whole-capture sharing, identity through clone/move/thread transfer, independent
same-shape captures, cache-key separation, frozen graph values and isolated owned
payload exports. A process-isolated Python test exercises real HTTP/websocket
servers: separate captures use distinct ports, update-before-show rejects on an
independent capture, repeated use of the original accessor reuses its own port,
and actual server snapshots retain sparse node/edge IDs. Only notebook rendering
is replaced; the child process releases its servers on exit.

Matched release receipts use identical `benches/viz_snapshot_sharing.rs` source
(hash embedded in both receipts), 100 clone/drop iterations per observation,
five observations and dense 1k/10k/100k chains with two node properties. Fixture
and initial capture excluded; clone, black_box, constant-time count assertion
and clone destruction included. Before is b00e0ba5's deep owned clone; after is
the shared Arc payload. At 100k nodes median clone/drop falls from about 26.06ms
to 10ns. The amortized measurement is above clock resolution; it is not a single
10ns timestamp reading. No RSS/Python/GIL/FFI/server or whole-graph COW claim.

Receipts: `notes/benchmarks/viz-snapshot-clone-before-2026-10-25.json` and
`notes/benchmarks/viz-snapshot-clone-after-2026-10-25.json`.

Initial capture is still proportional to rows/properties. DataSource payload
exports still clone owned vectors; layout, serialization and viewer controls
remain separately scoped. Native legacy table DataSource readers and
interactive_viz option/fallback paths retain their previous limitations.

Full verification: **1,143 native passed / 51 ignored; 1,870 root Python passed /
1 skipped; 130 executable docs passed; strict all-target Clippy passed**.
