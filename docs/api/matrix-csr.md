# Canonical CSR matrix storage

Item 29.20 adds validated owned CSR construction to native and Python GraphMatrix:

```python
matrix = GraphMatrix.from_csr(2, 3, [2., 5., -1.], [0, 2, 1], [0, 2, 3])
assert matrix.get(0, 1) == 0.0
assert matrix.is_sparse
```

Pointers start at zero, have rows+1 entries and end at the value count. Values and
column-index counts match. Each row has strictly increasing unique in-range columns
and no explicitly stored zeros. Malformed input rejects; duplicate entries are not
silently summed. NaN/infinity values follow the matrix's existing numeric domains.
Empty shapes are supported with their corresponding pointer arrays.

Reads return zero for missing entries. Native get/set validate coordinates and relevant
CSR ranges; set validates the full layout, then updates/removes/inserts canonically.
Updates can move O(nnz + rows) elements. Cloning CSR owns independent vectors instead
of panicking. Dense materialization validates shape, pointers and allocation sizes.
Public raw backend fields still permit callers to corrupt metadata; direct mutation
of those fields is outside the supported construction/update API.

GraphMatrix operations implemented through checked get/set can consume CSR inputs:
owned selection/reshape/transpose, checked add/subtract/Hadamard, products and direct
floating solves. They produce **dense results**. GraphMatrix multiplication still uses
the dense logical traversal and numerical kernels, with row validation overhead;
this is correctness support, not sparse-optimized matmul or a large-scale complexity
claim. Dense/CSR combinations share positional dimension and numeric contracts.

is_sparse reports actual CSR storage consistently; zero_count counts implicit zeros
and actual dense zeros. Property analysis reports zero density independently of storage
kind, with zero density zero for empty matrices and None if reading fails. It no longer
uses the former constant-zero placeholder. These scalar density scans are not an index.

Remaining: efficient CSR kernels, conversions/format selection, sparse factorizations,
lower-level backend matmul, sparse fill/map/activation support, lazy/view clone semantics,
GIL release and production benchmarks. Unsupported lower-level operations still error.
This slice does not complete every sparse or advanced-matrix API.

Evidence: tests/matrix_csr.rs validates construction, implicit values, insert/remove,
clone isolation, dense products/transforms/selection, empty shapes, invalid layouts and
row corruption errors; tests/test_matrix_csr.py exercises the exposed constructor and
checked consumers. No external sparse library is used.
