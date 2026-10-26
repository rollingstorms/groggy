# Graph storage access and mutation ownership

`Graph::pool()` and `Graph::space()` expose read-only inspection. Raw mutable owner
access is not public: `pool_mut` is crate-private for prepared retained-source
installation/replay; the `space_mut` corruption hook exists only in native tests.
Use Graph, GraphWriter or GraphBatch methods for topology/property mutations.
Explicit manual graphs still record changes and accept them through their manual
workflow; they do not expose arbitrary untracked owner writes.

This deliberately retires the former public raw mutation escape. Standalone
GraphPool/GraphSpace storage types remain usable for independent low-level work;
a caller cannot attach their mutable storage to an existing Graph through a raw
owner accessor. Read-only returned values, immutable snapshots and retained source
payloads keep their existing ownership rules.

Native compile-fail cases verify that even `&mut Graph` cannot call the old raw
owner accessors. Internal corruption/recovery fixtures keep trusted access, so
malformed-index, stale-value, allocation, source-binding, snapshot-copy and read
validation coverage remains in the suite. Those 46 tests moved from ten integration
files into `src/api/graph/storage_boundary_tests/`; run them with:

```text
cargo test --offline --locked --lib api::graph::storage_boundary_tests
```

## Group attributes are not meta-node attributes

Pooled group handles have no independent attribute store. Their previously stubbed
setter returned success without storing the supplied value, and their getter
always reported absence. Both now return typed `GraphError::NotImplemented` errors.
They never allocate a property column, publish a version or alter a source.

Use the owning meta-node's ordinary properties for meta metadata. Subnode property
edits remain separate; changing a source/membership revision requires an explicit
new frozen reference. No automatic propagation into every containing group occurs.
Generic GraphEntity node/meta/edge mutations still use their Graph coordinator;
pooled group/neighborhood/component/path/filter attribute operations reject.

This closes the raw-access audit component of 09.4b.2. It does not claim completed
view migration, all legacy import validation, or topology-wide pipeline staging.
