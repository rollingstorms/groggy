# Direct floating linear systems

Item 29.16 replaces inverse-based `GraphMatrix.solve(B)` and the determinant-threshold
inverse implementation with shared in-house dense Gaussian elimination. It accepts
square f32/f64 coefficients and a matching-row matrix B with any number of columns.
Integer matrices reject explicitly rather than returning truncated inverse values.
The source matrices remain unchanged; working arithmetic is f64.

Normalize each coefficient row and its RHS by that row's maximum magnitude, then
choose partial pivots and eliminate directly. There is no absolute determinant or
pivot epsilon: a representable nonzero pivot is used; an exact zero pivot errors.
Fused elimination updates and accurately accumulated backward-substitution residuals
reduce arithmetic loss. Inverse solves against identity through the same kernel.
Zero-order inverse is 0x0; a 0x0 system can solve a 0xK RHS. Coefficients are still
validated for a zero-column RHS.

Reject nonfinite input, zero coefficient rows/pivots, nonzero terms lost completely
during row normalization, nonfinite intermediate arithmetic, and output that cannot
remain finite in its output dtype. Finite underflow in final dtype conversion remains
ordinary floating rounding. Errors replace fallback ones/zeros and ignored writes.
The algorithm is bounded O(N^3 + N^2*K) work with owned dense O(N^2 + N*K) scratch.

Results have coefficient column names as row labels (the unknown coordinates) and
RHS column names as output columns. Inverse therefore swaps positional domain/codomain
labels, using defaults where original equation row labels were absent. Labels do not
align/reorder equations automatically; graph bindings/gradient tapes are not inherited.

**Limits:** partial pivoting does not guarantee a well-conditioned problem, exact
rank detection or small forward error. Near-singular inputs can be accepted and
have large sensitivity. Condition estimates, residual-report APIs, iterative
refinement, least squares, sparse solves, complex arithmetic and general decomposition
repairs remain separate work. Finite range checks can reject a solvable system when
this particular elimination/scaling path cannot represent its intermediate values.
No backward-stability or throughput guarantee is claimed.

Evidence: tests/matrix_linear_solve.rs covers pivoting/multiple RHS, tiny/huge scale,
inverse reconstruction, identity metadata, empty shapes and rejected domains/ranges.
Python tests/test_matrix_linear_solve.py uses independent Fraction elimination on
seeded diagonally dominant systems through order eight, with common scaling over
500 decimal orders, plus inverse/domain checks.
