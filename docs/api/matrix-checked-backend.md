# Shared checked matrix reference kernels

GraphMatrix.matmul now shares multiply's accurate typed product and positional
identity. UnifiedMatrix matmul, element arithmetic, scalar multiplication,
reductions and transpose use checked logical values across dense layouts and
canonical CSR. to_vec returns logical row order, validating dense strides/storage
and sparse metadata. Malformed public strides return errors rather than index
panics. Legacy lazy/view access continues to return unsupported errors.

The shared product accumulates finite f64 products exactly before one output
rounding; f32 narrows afterward. Built-in integer products/subtotals use checked
i128 and reject final dtype overflow. Bool retains OR/AND semiring behavior.
Lower-level sum returns its declared wider Accumulator dtype, while extrema
retain typed values/first ties and propagate NaNs. Empty extrema reject.

Safe backend wrappers validate every slice length and shape before raw calls.
Only actual f64/f32/i64/i32/bool types may cross the raw dtype ABI. In particular,
i128 reports Int64 for historical compatibility but is rejected by that raw ABI;
it remains supported by typed matrix reference kernels. User-defined NumericType
implementations cannot be safely reinterpreted from a dtype tag and are likewise
rejected by raw wrappers. Native raw element operations/reductions share checked
kernels, with staged outputs on errors. Unsafe raw calls still require valid,
aligned, nonaliasing buffers of the documented extents from their caller.

Native GEMM uses the shared dot and a checked alpha/beta combination; zero
alpha/beta skip their respective terms. The dot is rounded before scaling,
so this does not promise an exact triple-product GEMM result across unlimited
intermediate range. Range/overflow errors preserve caller outputs. External
BLAS/NumPy backends remain unavailable stubs; no acceleration is claimed.

Current UnifiedMatrix sparse arithmetic may materialize dense buffers; optimized
sparse output paths are the next completion-pass task. Reference speed, copying,
allocations and release throughput remain explicit performance work. Native
fixtures cover physical layouts, cancellation, integer limits, CSR consumers,
invalid metadata and raw slice/type safety. Rebuilt Python regressions exercise
all existing numerical GraphMatrix behavior.
