# Native graph-table feature matrices

`GraphTable::to_matrix<T>()` and the f64/f32/i64 conveniences copy node
properties into an owned numerical matrix. `edges_to_matrix<T>()` copies edge
properties. These existing Rust adapters share one implementation; they are not
new Python methods.

- Rows retain physical table order. Features retain table column order.
- Nodes exclude `id`/`node_id`; edges exclude `id`/`edge_id`/`source`/`target`.
- Node IDs become `node_<id>` row labels. Edge IDs are validated without being
  mislabeled as node IDs; edge matrices have no row labels.
- Feature names survive both conversions, including zero-row tables. An ID-only
  table gives an N×0 matrix; a zero-row table with K features gives 0×K.
- IDs must be unique nonnegative integers. Invalid or duplicate IDs reject even
  when the matrix has no features.
- Every nonstructural column is converted using the existing `FromAttrValue`
  policy: null becomes zero, booleans become zero/one, numeric text is parsed.
  Unsupported properties or range failures reject with feature name and row.
  This adapter does not silently select a numeric subset or reinterpret vectors.
- An output is an owned numerical copy. Editing the input afterward leaves the
  matrix unchanged; it has no implicit connection for graph writeback.

Previously both adapters supplied column-major values to a row-major constructor.
The repair changes incorrectly arranged results; edge outputs also regain their
feature names. Invalid IDs now reject instead of being accepted by permissive
legacy ID extraction. This does not certify the separate CSV bundle format or
all graph/table reshaping adapters.

Native regression coverage: `tests/graph_table_features.rs`. Current-cost benchmark:
`benches/graph_table_features.rs` (1k/10k/100k rows, eight columns, both domains).
