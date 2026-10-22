# Sparse numerical operations

Canonical CSR inputs now preserve CSR storage through pairwise add/subtract,
Hadamard product, matrix product, transpose, logical reshape, scalar maps and positional selections. Repeated
selection indices remain distinct output positions. Explicit to_csr/to_dense
methods return owned conversions and preserve coordinate identity. Mixed
sparse/dense products return dense output and skip structural zeros for finite
inputs without first materializing the sparse operand.

Finite sparse products build sorted output-column candidates from reachable
nonzeros, accumulate each dot with the shared accurate floating/checked integer
reference, and remove resulting zeros. Product scratch follows candidate columns
for one row, with checked binary lookup in source rows. No dense m×n scratch is
created for a CSR pair. Fill-in can still make the actual result dense.
Transpose uses row/column/nnz-sized buffers. Selection uses a positional column
map and sorted sparse row outputs. CSR sum/min/max avoid dense copies
and include implicit zeros where relevant.

IEEE semantics are preserved. In particular zero times infinity is NaN even
when the zero is implicit. A nonfinite sparse product uses full logical dot
traversal and still emits CSR; this can cost dense arithmetic and produce a dense
NaN pattern. Hadamard unions include nonfinite/implicit-zero interactions.
Nonfinite scalar multiplication likewise examines implicit positions. There is
no finite-only shortcut that silently changes these results.

Canonical constructor/metadata/domain checks and integer overflow errors remain.
Canonical CSR omits both signs of zero: explicit conversion and sparse outputs
normalize zero entries to implicit positive zero, including signed underflow.
Dense outputs retain their existing signed-zero behavior.
Sources stay unchanged. Sparse conversion of a dense input explicitly scans it;
explicit dense output/conversion allocates its logical size. General statistics,
solve/factor paths, release memory/time and further
kernel optimization remain in the numerical completion pass.

Native tests include a 10,000×10,000 diagonal product with only 10,000 output
values, canonical/repeated-selection/layout/limit/nonfinite/cancellation cases.
Python compares sparse and mixed results to independent dense expressions.
