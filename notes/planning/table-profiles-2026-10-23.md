# 07.3a.2c — Checked table profiles

BaseTable.profile previously ignored f64 cells, rounded computed values and i64
extrema to f32, sorted floating values with a panicking NaN comparison, and counted
text content plus its debug representation as separate unique values. Empty ratio
fields produced NaN. Those behaviors blocked the property-default migration.

The table adapter now borrows numeric source values alongside shared typed scalar
hints. It uses exact mixed ordering, the checked column mean, scaled population
standard deviation and exact product accumulation for even medians. Minimum,
maximum and rank quartiles retain source types; mean/median/std/ratios are f64.
Null/text/Boolean/vector cells remain outside numeric observations. Nonfinite
numeric cells fail explicitly. Dtype names distinguish f64, ties are deterministic,
and typed uniqueness uses borrowed property hashes without debug/string copies.

Population std, one-observation std zero and floor(n*q) rank quartiles remain the
profile conventions. Empty ratios are null. These differ from describe's
interpolated quartiles; see docs/api/table-profiles.md. This slice does not switch
ordinary Python float inference or complete every table/display adapter.

Verification: native regression cases cover sub-f32 values, high i64 spread and
extrema, all quartile residue classes, maximum-f64 medians/means, wide std,
nonfinite errors, text uniqueness, empty columns and nonnumeric vectors. Python
regressions exercise the public profile method. Release profile fixtures at
1k/10k rows measure f64 and high-i64 columns with setup excluded. Receipts are in
`notes/benchmarks/table-profiles-2026-10-23.md` and
`benches/table_profiles_2026-10-23.json`.

Executed: full locked offline native tests, final targeted table/profile tests,
strict all-target Clippy, in-tree FFI rebuild, 26 focused Python regressions and
16 documentation examples. Profiling releases the GIL during Rust computation.
Three pre-existing FFI warnings remain; no packages installed.
