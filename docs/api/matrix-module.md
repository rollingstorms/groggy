# One matrix implementation module

The canonical native namespace is `groggy::storage::matrix`. It owns:

- `GraphMatrix<T>`: public numerical API, labels and differentiation bindings;
- `UnifiedMatrix<T>`: checked dense/CSR storage, shapes and logical layout;
- `NumericalMatrixData<T>`: owned numerical transfer without graph/tape ownership;
- arithmetic, factors/solvers, graph operators, backend capabilities and buffers;
- generic `matrix::autodiff`, separate from layer-specific `matrix::neural` APIs.

`groggy::storage::advanced_matrix` is a compatibility re-export of this same
implementation. `storage::adjacency` re-exports `matrix::adjacency`.
`matrix::neural::autodiff` remains an alias of `matrix::autodiff`.
These are aliases, not parallel implementations or conversion wrappers. Old
values can pass directly to canonical APIs, including trait bounds and nested
backend/storage paths. Existing storage-root re-exports remain available.

Dense/CSR, graph metadata and thread-transfer data still have distinct roles;
consolidating modules does not collapse them into one graph-bearing struct.
Numerical domains, mutation/delta behavior and Python APIs are unchanged by this
move. Available backend types do not imply BLAS/NumPy acceleration is implemented.

Maintained native/FFI callers use canonical matrix paths. Compatibility tests
exercise legacy imports and type identity, alongside the existing native and
rebuilt Python regression suites. The numerical release receipt is compared
before and after the move. Shared array/scalar foundation work follows in AR7.3.
