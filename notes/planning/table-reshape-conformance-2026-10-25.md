# Typed table filtering and reshape conformance — 2026-10-25

Existing requirements: 27.1 / AR7.10. No new checklist rows.

## Closed inventory

- Node and edge attribute filters delegate to BaseTable typed property equality.
  Quotes/query syntax, nonidentifier column names, vectors, nulls, small integers
  and precise f64 stay literal. Direct selection retains exact source row IDs.
- Generic pivot reuses the stable typed group mechanism. First-key occurrence
  order replaces randomized HashMap order; duplicate index names reject.
- Generic melt rejects duplicate/overlapping ID/value columns and colliding
  output names, including empty input, and checks multiplied row counts.
- Native node/edge pivot and melt retain required integer structural columns and
  unique entity IDs, or reject with instructions to use the underlying BaseTable
  for general analytical layouts. They no longer return a structurally invalid
  specialized table. No new graph identity is generated.
- Python melt exposes native `id_vars` as an optional keyword; existing positional
  value_vars/var_name/value_name calls remain valid. Both analytical bridges and
  entity filters release the GIL after marshaling. Python stubs are updated.

Analytical reshape outputs remain read-only derived source captures. They do not
invent automatic writeback ownership. Filtered direct rows keep existing binding.

## Evidence

Six native regressions cover literal typed values, stable grouping, empty-input
validation, native specialized ID/endpoints, general long tables and source pins.
Seven Python cases cover the exposed filters, melt identifier keyword and
backwards-compatible positional use, group order and source capture retention.
API examples: `docs/api/table-reshaping.md`.

`benches/table_reshaping.rs` measures five current-cost observations for each
operation at 1k/10k/100k input rows. Complete operation/allocation is timed;
fixtures/assertions are excluded. Receipt:
`notes/benchmarks/table-reshaping-2026-10-25.json`. At 100k rows, medians are
0.417 ms typed filter, 13.840 ms pivot and 18.868 ms two-column melt on this
machine. No matched speedup, memory or FFI latency claim.

Remaining parent scope: composite GraphTable subsetting/routing and bundle
conversion/integrity, plus the separately tracked legacy accessor/pool boundaries.


### Typed filters and checked reshapes — 2026-10-25

Closed node/edge typed equality filtering, stable generic pivot grouping,
checked melt schema/row counts, specialized pivot/melt identity validation and
the Python melt identifier adapter under 27.1/AR7.10. Existing positional Python
melt calls remain valid; `id_vars` is an optional keyword. General long layouts
use BaseTable; specialized outputs cannot discard or duplicate graph identity.

Evidence: `table-reshape-conformance-2026-10-25.md`. Gates: **1,090 native passed /
51 ignored; 1,819 root Python passed / 1 skipped; 112 executable docs passed;
strict all-target Clippy passed**. Current-cost receipt:
`notes/benchmarks/table-reshaping-2026-10-25.json`. No new checklist rows;
191 checked / 77 unchecked. Composite/bundle adapter scope remains open.
