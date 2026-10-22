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
