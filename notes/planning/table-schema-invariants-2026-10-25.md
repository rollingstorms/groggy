# Table schema invariant completion — 2026-10-25

Existing 27.1 / AR7.10 foundation; no new checkbox rows.

Completed the fixed schema boundary inventory:

- Ordered construction requires every physical column exactly once. Duplicate,
  hidden and unknown names reject on zero-row and populated input. Existing
  equal-length checks remain; sparse from_columns null padding is unchanged.
- Base/node/edge/composite projection rejects duplicate and unknown requested
  names. Required node/edge structural columns remain automatically retained.
- Adding/replacing a native column in a table with declared columns must match
  its row count, even zero. A truly empty 0x0 table can still initialize from
  its first column. N-row zero-column projections keep N when columns are added.
- Python specialized selection now delegates to native selection, removing
  independent FFI deduplication/required-column business logic. Native projection
  runs with the GIL released. Python call shapes remain unchanged.
- Rejections leave original data, source captures, row identity and graph history
  unchanged. Schema checks run before copying selected output columns.

Six native and six Python cases cover constructor/order invariants, zero shapes,
all projection layers, explicit padding and source binding/rejection behavior.
API examples: `docs/api/table-schema-invariants.md`.

These are schema validation corrections, not new coercion or graph subsetting
semantics. They add O(schema width) validation to affected construction/selection
paths and no new numerical/row kernel. No throughput, memory or fixed FFI-budget
claim is made; no additional performance benchmark is introduced for these checks.


### Shared table schema validation — 2026-10-25

Closed ordered constructor, projection and column-growth invariants under
27.1/AR7.10. Every physical column must appear exactly once in an ordered schema;
projections reject duplicate/unknown requests; declared zero-row schemas cannot
be expanded by changing only one column. Truly empty initialization, sparse
from_columns padding and N-row zero-column projections retain their contracts.
Python node/edge projection delegates to native validation with GIL release.

Evidence: `table-schema-invariants-2026-10-25.md`; 6 native / 6 Python cases.
Gate: **1,120 native passed / 51 ignored; 1,861 root Python passed / 1 skipped;
124 executable docs passed; strict all-target Clippy passed**. These are O(schema
width) checks, without a new performance claim/kernel. No new rows: **192 checked /
76 unchecked**. Composite routing/subsetting and broad legacy migration remain.
