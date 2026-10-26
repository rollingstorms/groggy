# GraphTable facade exit: explicit grouped immutable views

Owner confirmed `to_subgraphs(by=...)` means explicit node-property groups,
returning immutable graph views with internal edges. Implemented natively with
one checked frozen capture, shared by all groups, using the existing one-pass
induced edge dispatcher. Python marshals direction and returns GraphView wrappers
while releasing the GIL for native work. Table edits are captured as table values;
views do not silently read stale source properties or acquire writeback authority.
Sparse IDs, loops, parallel edges, null groups, empty declared groups and shared
node/edge column names are retained/covered. Missing grouping columns, malformed
identity/endpoints and pooled references reject rather than lose source content.

This completes the fixed 27.1 facade inventory and AR7.10 table foundation scope:
separate shapes/components, induced subsets, explicit component aliases, checked
materialization and explicit grouped views. Legacy live accessor migration stays
26.1; pooled history portability 07.3/10.3/12.2; legacy visualization coverage
AR7.6/28.15. These are existing scopes, not new completion blockers for 27.1.

Verification: `cargo test --test graph_table_subsets` (8 passed), rebuilt Python
`tests/test_graph_table_components.py` (10 passed); executable grouped-view docs
are part of the full documentation gate. Full gate receipt is in the status log.
