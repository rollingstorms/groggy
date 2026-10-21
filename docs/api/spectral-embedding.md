# Undirected spectral embedding

Rust retains `SpectralEmbedding`, `SpectralEmbeddingBuilder`, the existing
GraphEmbeddingExt method and the spectral factory. Python adds a thin graph method:

```python
coordinates = graph.spectral_embedding(
    2, normalized=True, eigenvalue_threshold=1e-8, skip_nullspace=True,
)
node_labels = coordinates.row_labels  # copied list: node_<id>, ascending active ID
scores_2d = coordinates.pca_projection()  # retains the row labels
```

The returned GraphMatrix owns coordinates; parent mutation does not change its
values/labels. Computation does not allocate graph IDs, write graph properties or
record history. This is a materialized numerical result, not completion of the
ordinary connected-view snapshot contract.

## Operator and domain

Only undirected graphs are supported by this symmetric Laplacian/eigensolver
path. Directed input rejects explicitly instead of silently symmetrizing stored
endpoints. There is no edge-weight attribute option: every active edge instance
has unit adjacency mass. Parallel instances accumulate. A self-loop contributes
once to its adjacency diagonal and adjacency row degree.

With A the symmetric adjacency matrix and D its row-sum degree matrix:

- Unnormalized: L = D - A. Self-loops cancel on the diagonal.
- Normalized: L = D^(-1/2) (D-A) D^(-1/2), with inverse degree zero for degree-zero
  nodes. Isolated rows/diagonals remain zero. Loops can affect degree normalization
  when a node also has other edges.

These loop conventions are explicit; do not assume a backend that drops loop
degrees or counts them twice computes the same normalized operator. Endpoints are
read through checked active Graph APIs and mapped without unchecked hash indexing.
Output row order/labels are ascending original active node IDs, including gaps.
Columns are named `spectral_0`, `spectral_1`, etc.

## Null space and dimensions

Return orthonormal eigenvector columns in ascending eigenvalue order. With the
default skip mode, exclude all component null modes and additional eigenvalues
whose absolute value is at most the finite positive threshold. A dense traversal
of negative off-diagonal support establishes the exact component-nullity lower
bound, so a tiny cutoff cannot accidentally retain a numerically perturbed zero
mode. The threshold controls selection, never degree normalization.

Disconnected graphs, isolates and loop-only nodes contribute separate null modes.
Requesting more retained directions than exist rejects; no padding or artificial
isolate modes are supplied. An empty graph rejects. A singleton has no default
nonconstant direction, but explicit `skip_nullspace=False` can return its one
zero-mode coordinate. The same option permits zero modes of larger edgeless or
disconnected graphs. Rust's existing builder name
`skip_constant_eigenvector(false)` supplies this option, despite the historical
singular wording. Normalized null vectors generally scale with square-root degree,
rather than being constant over a component.

Bases/signs in repeated eigenspaces are nonunique. Compare residuals, subspaces
and geometry rather than exact columns across numerical libraries. Large cutoffs
can legitimately leave no dimensions. Zero, negative or nonfinite thresholds and
zero/excessive requested dimensions reject.

## Evidence and limits

A native operator regression checks exact expected loop/parallel/isolate entries
in both Laplacians, including a large threshold that must not change normalization.
`tests/spectral_embedding.rs` checks all 64 three-node undirected edge sets with
loop/parallel instances against independently constructed Laplacians: full basis
residuals/order/orthogonality, threshold independence and component-null-space
selection. Other fixtures cover labels, owned results, sparse IDs, directed and
invalid controls, and explicit singleton zero modes.
`tests/test_spectral_embedding.py` compares both modes against NumPy eigh and tests
the rebuilt Python methods and label-copy/PCA interoperation.

Two existing tests used a sparse 34-node fixture connecting only five nodes. Their
fifth default nonconstant dimension depended on spurious isolate modes. They now
require its four real positive directions and rejection of a fifth.

Dense storage costs O(N²); nullity scanning/normalization also cost O(N²). The
bounded native Jacobi solve costs up to O(sweeps * N³), with its documented finite
precision/iteration limits. Checked dense size arithmetic rejects overflow, but
ordinary allocation failure is not a recoverable-memory guarantee. No sparse
acceleration, incremental update, cancellation, safe GIL release, directed variant
or production-scale capacity is claimed. Eigenvalue accuracy near numerical
resolution and overly aggressive threshold choices remain caller-visible limits.

The broader analytics audit remains open: other visualization/energy callers, t-SNE/UMAP-inspired scope, realtime failure fallbacks, and
pipeline/history/backend/scale verification are separate work under item 28.15. The [flat-layout repair](flat-embedding.md) and [composite composition](composite-embedding.md) have their own supported scopes.
