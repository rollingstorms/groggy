# Two-component PCA projection

Rust `PCAProjection::new(center, standardize).project_matrix(&matrix)` returns an
owned N-by-2 GraphMatrix. The existing visualization `project_embedding` method
uses this same implementation and converts its rows to positions. Python exposes
it through a translation-only method:

```python
scores = matrix.pca_projection(center=True, standardize=False)
assert scores.shape == (matrix.shape[0], 2)
```

Input rows are observations, columns are features. Require at least one row and
two features, with finite f64 values. Retain input row order and optional row
labels; output columns are `pc_1` and `pc_2`. Python `row_labels` returns a copy of
optional matrix labels. Projection does not guess graph IDs or mutate the input.

## Calculation and edge cases

Centering subtracts each feature's mean. Standardization uses population deviation
(ddof=0), computed around the feature mean even with `center=False`. Constant
features use scale one: centered constants become zero; uncentered constants stay
unchanged. These conventions also define a singleton input without dividing by
N-1. With `center=False`, this is a principal-direction projection about the
origin, without implicit centering.

Column-scaled mean/deviation calculations avoid sums and squares of raw extreme
values. The prepared matrix is then divided by a common maximum magnitude before
forming its feature-space mean scatter. A common positive normalization changes
eigenvalues but leaves principal directions unchanged. The existing bounded native
symmetric eigensolver supplies descending orthogonal directions; project on its
first two columns and restore the prepared scale. Check preprocessing and scores
for finite range, returning errors instead of NaN/infinite coordinates.

Signs and bases in repeated eigenspaces are not unique. A tied eigenvalue at the
two-component cutoff may admit several equally valid projections. Compare score
geometry/retained energy rather than exact axis signs. This method returns scores,
not a fitted reusable model, components/explained-variance report or whitening.

The [StandardScaler reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
documents the population-deviation/constant-feature conventions used here. The
runtime remains Groggy's native implementation, with no scikit-learn dependency.

## Costs and limits

Dense feature scatter requires O(NF²) work and O(F²) storage, in addition to
O(NF) prepared data and N-by-2 scores. The bounded dense Jacobi solve adds up to
O(sweeps * F³) work; no sparse or reduced-rank acceleration is introduced. This
avoids allocating full N-by-N SVD factors, but forming scatter still squares the
condition number. Scaling protects uniform large/small magnitudes; it does not
promise relative accuracy for arbitrary mixed extremes or directions below
floating resolution. Nonrepresentable outputs reject. Ordinary allocation failure
is not a recoverable-memory guarantee.

Calls retain the existing numerical API's GIL/ownership behavior; no safe GIL
release, cancellation or production-scale latency guarantee is added. Matrices
are materialized results, not automatically state-pinned graph views.

## Evidence

`tests/pca_projection.rs` covers analytic rank-two geometry, scales 1e-200/1e200,
constant/singleton/uncentered inputs, label/order preservation, trait/matrix
agreement, invalid shapes/values and output range rejection.
`tests/test_pca_projection.py` compares all center/standardize variants on tall,
wide and singleton inputs against NumPy SVD using score Gram matrices, with
separate extreme-scale and repeated-spectrum checks. Input values remain unchanged.
These are bounded correctness fixtures, not a full PCA/ML or performance claim.
