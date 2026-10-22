# Graph matrix operators

`to_degree_matrix()` returns the diagonal of weighted **row** sums. A stored
self-loop contributes once to a row sum; this differs from incidence degree in
some undirected graph APIs. Finite values and checked typed sums are required;
boolean matrices require an explicit numeric cast. Dense/CSR storage and node
coordinate labels are retained.

`to_laplacian()` retains D-A. `normalized_adjacency(eps=-0.5, k=1)` explicitly
computes `(D^eps A D^eps)^k`. Zero-degree rows use a zero scaling factor;
`k=0` returns identity. Powers now use the checked matrix-power kernel, fixing
both the old repeated-squaring exponent error for k>2 and the k=0 result.

The parameterized legacy `to_normalized_laplacian(eps,k)` still computes this
normalized **adjacency** formula, including Python's existing eps=0.5 default.
Use the unambiguous names for new code. In particular eps=-1 scales both sides;
it is not the one-sided random-walk operator.

`normalized_laplacian()` and native `to_normalized_laplacian_standard()` return
`diag(positive row degree) - D^-1/2 A D^-1/2`. Isolate rows/columns are zero.
Graph `laplacian_matrix(normalized=True)`, MatrixType::Laplacian and the adjacency
builder's normalized-Laplacian helper now return this actual operator, repairing
their former adjacency scaling. Directed input need not produce symmetry/PSD.

`transition_matrix()` returns D^-1 A with zero dangling rows. Row values are
scaled before summing/dividing, so normalization works even when an unscaled
finite degree would overflow. Native/Python graph methods and MatrixType::Transition
now execute this transform instead of raising a placeholder error or returning
unchanged adjacency.

Normalization/transition require floating dtypes and finite nonnegative weights.
Integer callers must cast explicitly. Degree powers and typed degrees have
explicit working-range errors; generalized symmetric scaling does not promise
representable intermediates for every extreme input. Graph transforms return
owned operators; tracked inputs must detach explicitly until their full
row-degree-dependent pullbacks are implemented. This prevents partial gradients.

The Python bridge now passes the existing native GraphMatrix alias directly.
It no longer rebuilds numeric columns, fills failed reads with zeros or drops
node labels through redundant conversion. CSR storage survives this boundary.

Evidence: `tests/matrix_graph_transforms.rs` and
`tests/test_matrix_graph_transforms.py` cover independent weighted formulas,
isolates/self-loops, generalized powers, overflow-safe transitions, sparse
operators, dtype/domain checks and high-level graph identity/routes.
