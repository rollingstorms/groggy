# Economy SVD

`svd_thin()` returns owned m×k U, k descending singular values and k×n Vt,
where k=min(m,n). Native/Python `svd()` retains its existing full square factors.
`svd_tracked()` retains matrix-valued singular values and now uses the same
actual economy kernel instead of building and slicing full factors.

The in-house bounded Jacobi routine shares rotations, spectrum ordering and
rank-deficient basis construction across both modes. Economy mode only completes
the requested k columns. Wide input uses the transposed tall kernel. Workspace
and factor storage follow m×k, n×k and k², without m²/n² null-basis allocations.
This does not change spectral convergence/range/domain policies or promise exact
rank. Rank/condition estimates, minimum-norm least squares and thin SVD JVP/VJP
now use economy factors internally.

Thin U retains source row labels and named `svd_i` basis columns; Vt retains
source column names and basis row labels. Tracked singular values use basis rows
and one `singular_value` column. Invalid label counts reject. Rectangular empty
and rank-deficient inputs retain the requested shapes and orthonormal thin bases.

CSR inputs still require dense logical work for SVD/spectrum/least-squares and
arbitrary gradients; this is not an iterative sparse eigensolver or sparse SVD.
Callers wanting a full arbitrary null basis must request full SVD explicitly.

`tests/matrix_economy_svd.rs` checks full/thin agreement, reconstruction, empty
shapes and a 10,000×2 rank/condition/least-squares/tracked case.
`tests/test_matrix_economy_svd.py` adds independent NumPy references, wide/tall
10,000-dimension cases and orthogonality. Existing spectral derivative references
continue to pass. Release allocation/time receipts are in
[the economy benchmark notes](../../notes/benchmarks/matrix-economy-svd-2026-10-22.md).
