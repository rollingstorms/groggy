# 07.3a.2a — Exact f64-aware array/table predicates

Consumer migration begins with numeric selection. Give numeric attribute-to-input
conversion and symbolic comparisons one owner under storage::numeric::scalar.
Graph numeric ordering, array comparison queries, table numeric predicates and
cross-table extrema share exact wide comparisons. Keep typed property identity
and graph attribute equality unchanged; numeric predicate equality is a distinct
numeric operation. Boolean/text domains retain their existing handling.

Replace array/table epsilon checks and f32 integer coercion with exact/IEEE
comparison. Array floating literals retain f64 precision. Fix >=/<= token prefix
ordering, add f64 vector length, and stop parity predicates from truncating
fractions or saturating nonfinite floats into integers. Cross-table extrema reject
unordered NaNs rather than pretending equal. No storage/default-inference changes.

Native references cover every operator at sub-f32 precision, integer boundaries
above 2^53, compact integers, mixed widths, NaNs, infinities and wide finite values.
Additional cases cover vector length, fractional/nonfinite parity and mixed-width
cross-table extrema. Python table-query checks retain exact values and large IDs.
See docs/api/property-numeric-predicates.md.

Benchmarks retain native array/table scalar-query scan receipts at 1k/10k f64
values. They include the existing repeated scalar parsing, not compiled predicate
planning, output materialization, FFI or graph mutations. Full AR7.7 remains open;
remaining arithmetic/results, legacy algorithm paths and default switch remain
open under 07.3a.2–3.

Executed: locked offline full native tests; strict all-target Clippy; in-tree FFI
rebuild; 86 focused Python predicate/property/statistics/read-view/builder checks;
14 documentation examples; release scan receipts. Three pre-existing FFI warnings
remain. No external packages installed.
