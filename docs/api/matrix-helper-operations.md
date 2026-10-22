# Checked helper operations

Lower `MathFunctions` now preserve IEEE floating results for exp/log/sqrt/power:
log(0)=-infinity, log(negative)=NaN, sqrt(negative)=NaN. Integer/custom
outputs that cannot represent a result return an error. Finite f64-to-f32
narrowing overflow remains an error. Integer abs is exact and rejects MIN;
there is no wrapping/panic or fallback to zero. ReLU preserves NaN.

`UnifiedMatrix.try_map_elementwise` prepares an owned result with fallible
per-element conversion. Zero-preserving CSR maps visit only stored entries;
nonzero images of implicit zero cause real fill-in. Pure elementwise functions
are assumed: an implicit zero's image is computed once and reused. Full CSR
matrices never evaluate an absent implicit zero (important for integer log).
In-place maps prepare/validate the logical result before replacing the matrix,
and now support CSR. Owned dense outputs use row-major storage.

Reshape and special-matrix helpers check dimension/byte multiplication.
CSR reshape/slice preserve canonical sparse storage, including empty slices.
Two CSR inputs concatenate without dense materialization. Mixed storage
concatenation produces owned dense output. Native GraphMatrix concatenate and
split delegate to checked storage/selection paths and preserve positional
labels; duplicate names are allowed. Row concatenation combines row labels
when both inputs supply them; column concatenation retains the left row labels.
Split keeps the existing increasing-positive-points convention and omits empty
segments. `materialize` now returns checked owned concrete storage and detaches
autodiff state instead of cloning potentially unsupported view/lazy storage.

Native fixtures in `tests/matrix_helpers_checked.rs` and Python fixtures in
`tests/test_matrix_helpers_checked.py` cover these contracts. Generic zip_map
still materializes its arbitrary callback output; automatic differentiation,
release measurements and other neural helper audits remain separate gates.

## Public matrix routes and typed maps (29.51)

Public `GraphMatrix.abs/exp/log/sqrt` now delegate to the checked lower kernels.
They preserve positional labels, empty dimensions and CSR storage. Floating
log(0) is -inf; negatives/nonfinite values follow the documented IEEE behavior.
Integer absolute values remain exact/checked; integer transcendental outputs
retain the lower typed conversion/truncation policy. Cast to floating dtype
when fractional results are needed.

`UnifiedMatrix.try_map_into<U>` shares fallible typed logical/CSR mapping across
casts, mathematical functions and native boolean comparison masks. Zero images
stay implicit; nonzero zero-images create actual fill-in. This consolidates
previous parallel implementations instead of maintaining separate dtype/FFI
math. `GraphMatrix.comparison_mask` returns an owned boolean matrix with exact
mixed integer/float ordering and IEEE NaN comparisons. Existing Python ordered
comparisons keep their floating 0/1 matrix outputs via native bool-to-f64 casts;
large Python integer scalars are marshalled without first rounding them to f64.

Scalar add/subtract/multiply now support native NumericType outputs through
checked scalar conversion/arithmetic; integer scalar inputs must be exactly
representable. Native scalar division requires floating dtype and rejects zero.
Python's named `scalar_multiply` now executes the native method, and `abs(matrix)`
uses the same native path as `matrix.abs()`.

Tile and repeat use checked positional selections, preserving order, duplicate
labels and sparse storage. Their output dimensions are checked before building
index vectors; zero repetitions/empty extents retain their mathematical shapes.

Evidence: `tests/matrix_public_helpers.rs` and
`tests/test_matrix_public_helpers.py` cover native i64/f32 boundaries, malformed
layouts, CSR casts/maps/selection, IEEE domains, Python precision and empty shapes.
