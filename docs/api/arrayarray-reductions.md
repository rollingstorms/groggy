# ArrayArray numeric reductions

Item 29.13 routes native `ArrayArray<f64>` sum/mean/min/max/sample std through
the shared Rust numerical kernels. Ragged arrays retain their order and lengths;
input data and keys stay unchanged. Empty inner arrays still produce zero for
these legacy Vec<f64> APIs. Finite singleton std is zero; NaNs propagate to every
reduction, and nonfinite std is NaN, including singletons. Floating cancellation,
overflow-safe means and independently scaled std follow the NumArray kernels.

## Numeric projection and Python

`ArrayArray<AttrValue>.numeric_arrays()` owns the existing numeric projection:
Int/SmallInt/Float/Bool values become f64; null/text/collection values are skipped.
Boolean values are zero/one. Preserve both group keys and their actual column name.
The Python bridge delegates this conversion and multi-aggregation dispatch to Rust.
Existing `count()` on the Python bridge counts projected numeric elements, while
native `ArrayArray<T>.count()` counts all elements. Neither meaning changes here.
Large integers still lose precision at this explicit f64 conversion boundary;
exact typed grouped integer statistics remain open.

The Python constructor remains empty-only; grouped table or subgraph attribute
extraction supplies populated containers. This repair does not introduce a new
constructor, null policy or strict schema.

## Tables and precision

Native reductions return f64, but BaseTable stores scalar floating AttrValues as
**f32**. Single/multi-aggregation table results retain that existing storage format.
Finite f64 results that would narrow to infinity now return an error. Ordinary
f32 rounding and underflow remain; actual NaN/infinity results remain representable
IEEE values. This is not lossless f64 table storage. Counts materialized in tables
also retain the existing floating representation.

Single and multi-aggregation outputs preserve named keys, reject output names that
would overwrite keys, and reject duplicate or unknown multi-aggregation requests.
Multiple functions use one Rust dispatch path; Python only marshals names and errors.
Supported names remain mean/sum/min/max/std/count. Source containers stay unchanged.

Evidence: `tests/arrayarray_reductions.rs` covers cancellation, large means, tiny std,
ragged/empty/NaN input, numeric projection, key identity, narrowing overflow and output
collisions. `tests/test_arrayarray_reductions.py` covers grouped extraction, named keys,
single/multi aggregation, documented f32 output precision and flattening. Storage
format widening, exact grouped i64 moments, general grouped-table implementations,
GIL release, throughput and query semantics remain open.
