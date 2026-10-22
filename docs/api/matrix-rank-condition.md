# Numerical rank and condition estimates

Item 29.30 replaces rank's truncated-column Gaussian scan and fixed absolute threshold
with a shared normalized real singular spectrum. Native/Python rank() uses
relative cutoff epsilon(input dtype)*max(rows,cols,1); count singular values whose
ratio to the largest is strictly above that cutoff. rank_with_tolerance(relative)
accepts an explicit finite nonnegative relative cutoff. This is numerical rank,
not a proof of exact mathematical rank. Cutoff zero can count floating residuals. The SVD preserves computed nonzero singular values, including diagonal values far below machine epsilon; basis completion no longer silently imposes an additional rank cutoff. Finite-precision rotations may still lose unresolved directions or fail explicitly.

Normalize finite floating inputs before the existing bounded in-house SVD, so common
matrix scale and singular-value output overflow do not change the ratio calculation.
The whole rectangular spectrum is used, including independent trailing columns.
Native f32 uses its input precision for the default cutoff; f64 uses f64 precision.
Integer/nonfinite domains reject. Empty dimensions and all-zero input have rank zero.

condition_number() estimates the spectral 2-norm condition number as largest/smallest
computed normalized singular value. It does not apply rank's cutoff: a matrix can
have finite estimated condition while rank's chosen cutoff classifies it as numerically
singular. Zero smallest value/all-zero matrices yield infinity; empty dimensions error.
Ratios beyond f64 range also yield infinity. No exact bound, arbitrary-precision or
forward-error guarantee is claimed; normalized underflow and the SVD's convergence/
precision limitations remain. This is a dense calculation with the existing SVD's
factor allocation cost, not a cheap cached property or sparse condition estimator.

Sources and identity metadata stay unchanged. Python delegates both new methods to
the native kernels. Condition estimates do not silently alter solve/inverse policies.
Residual reporting, refinement and least squares remain separate work.

Evidence: tests/matrix_rank_condition.rs covers trailing-column rank, common scaling,
f32/f64 cutoff differences, huge finite entries, empty/zero/nonfinite/integer input
and invalid cutoffs. tests/test_matrix_rank_condition.py compares rectangular ranks/
conditions to independent NumPy singular spectra and tests explicit cutoffs/domains.
