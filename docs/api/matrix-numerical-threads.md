# Python numerical calls and thread ownership

Long untracked matrix products (`multiply` with a matrix and `@`), inverse,
solve/refinement/least-squares, real/general eigen/SVD/QR/LU/Cholesky factors,
rank/condition/determinant/sign-log determinant and explicit decomposition
JVP/VJP calls release Python's GIL around native computation.

Native `GraphMatrix.numerical_data()` creates a checked, owned, Send numerical
snapshot. It preserves dense logical values or canonical CSR plus positional
labels, and excludes graph references, caches and differentiation tapes.
`NumericalMatrixData.into_matrix()` reconstructs owned untracked storage on the
receiving thread. Python FFI copies snapshots under the GIL, performs numerical
work and packs outputs without the GIL, then reconstructs Python-owned results.
No unsafe Send implementation is used for GraphMatrix or its raw-pointer view
storage. Scalar/list/tuple/report transfer is FFI marshalling only.

If any matrix input is tracked, the existing path retains the GIL and tape
ownership. The wrapper does not silently detach gradients to release the lock.
This does not make unsendable Python GraphMatrix objects shareable between
threads. Worker progress tests create no cross-thread references to matrices.

Copies and reconstruction have real allocation/time costs, especially for small
calls; this slice does not claim a universal throughput gain or a 100ns FFI
budget. Reductions, scalar transforms, automatic backward and other small
methods are outside this GIL release slice.

`tests/matrix_numerical_transfer.rs` checks Send data, owned snapshots, CSR,
labels and invalid layouts. `tests/test_matrix_gil.py` checks another Python
thread's progress during dense product/QR/Cholesky and sparse Cholesky, validates
results, and checks that tracked Cholesky chains retain their gradients.
