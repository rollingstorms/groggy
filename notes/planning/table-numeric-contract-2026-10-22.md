# Table numerical closeout — 2026-10-22

Checklist 29.61c closes the inventoried table aggregate/pivot/describe/numeric
ordering paths. It does not choose table identity/writeback policy (A10/27).

## Shared adapters and output rules

- Group/whole-table numeric aggregation skips null and nonnumeric cells, includes
  compact integers, and rejects nonfinite numbers. Count still counts selected
  rows. No numeric cells gives Null for sum/mean/min/max.
- Sum uses canonical column output types: all-integer sums return checked Int;
  any float makes the result Float. This deliberately replaces the old heuristic
  that guessed an Int from an already rounded f64 sum. Integer/f32 overflow fails
  instead of saturating or yielding infinity. Mean remains checked Float.
- Min/max preserve the original selected numeric scalar and exact mixed ordering.
- Pivot sum/mean retain checked Float output, skip nulls and reject nonnumeric
  cells; all-null sum is zero, all-null mean is Null. Pivot count counts rows.
  Pivot min/max preserve original scalar types with value ordering (including
  text); numeric nonfinite cells are rejected.
- Numeric sorting compares integer/float values before conversion; nonnumeric
  values use established AttrValue ordering. This fixes lexical text sorting as
  well as false numeric ties. Single/multi-column sorting and numeric top-N share
  this table comparator. Generic BaseArray<T> ordering is unchanged.

## Pivot identity

Pivot unique-key construction and matching use the same AttrValue equality/hash
contract. Numeric sort ties do not establish key identity. This prevents unrelated
text keys or distinct large integer/float keys from collecting the same values.
If distinct keys generate the same output column name (e.g. Int(1)/Float(1.0)),
or a pivot output collides with an index name, return an explicit error. Choosing
an alternate naming scheme would require a separate public API decision.

## Describe

Compute one complete summary per column, retaining count/mean/population std/
min/25%/50%/75%/max row order and f32 output. Reuse checked typed centering,
column means and quantiles, skip nonnumeric cells and reject nonfinite numbers.
The old repeated eight-summary calculation is removed; no new table writeback or
live-view behavior is introduced.

## Verification

Native/Python fixtures cover i64 cancellation/overflow, mixed numeric ordering,
compact values, skipped cells, independent text and large numeric pivot keys,
ambiguous names, invalid empty aggregation, high-offset population spread and
mixed midpoint cancellation. Existing numerical/array/table suites are rerun on
the rebuilt extension before completion is committed.

Executed: full locked offline native suites, strict all-target core Clippy
(clean), rebuilt FFI and numerical/array/table Python gate: **560 passed, 1
existing skip**. Idle release corpus: **52 cases**, prior 50 checksums match
within 1e-13; high-offset describe measurements are in
`notes/benchmarks/numerical-table-statistics-2026-10-22.md`.
