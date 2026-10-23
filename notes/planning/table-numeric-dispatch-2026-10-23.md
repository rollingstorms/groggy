# 07.3a.2e — Whole-table f64 dispatch and typed matrix display

Whole-table entry points still detected only legacy scalar widths even after
numeric column kernels supported f64. Include f64 in describe, ranking, pairwise
statistics and all-column windows; include compact integers in window dispatch.
Propagate checked corr/cov errors instead of replacing them with null success.
Use the shared correlation kernel on diagonal cells as well, preserving its
constant/insufficient-observation semantics instead of fabricating one.

Move typed scalar-to-property conversion to one internal numeric helper shared
by NumArray materialization and matrix display. Matrix reads are checked, errors
propagate and explicit widths survive. Remove the temporary NumericInput vector
from NumArray materialization; matrix display retains one property buffer per
column. Display formatting can still round visible text, without changing values.

Native tests cover f64 rank/describe/windows, compact-integer windows, pairwise
values/invalid inputs/diagonal parity, and f64/i64/f32 matrix display properties.
Python regressions exercise public whole-table operations. CSV/JSON, outliers,
other view/step/vector consumers and Python defaults remain open. No history or
publication policy is changed here. See docs/api/table-numeric-dispatch.md.

Executed: full locked offline native tests, strict all-target Clippy, in-tree
FFI rebuild, 28 focused Python regressions and 17 documentation examples. Release
1k/10k materialization samples are retained in
`benches/typed_property_materialization_2026-10-23.json`; scope/limits are in
`notes/benchmarks/typed-property-materialization-2026-10-23.md`. Three existing
FFI warnings remain. No external packages installed.
