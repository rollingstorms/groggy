# 07.3a.2i — Remaining supported floating consumers

Move Python scalar array masks, inequality, sorting, numeric inference and legacy
numeric aggregation computation into shared native array operations. Remove
f32-only epsilon comparisons, mixed-integer narrowing and first-type selection.
Use a coherent mixed sort order (null; numbers with NaN last; other typed values)
rather than mixing numeric ordering with incompatible dtype ranks. Keep existing
edge no-numeric errors and hierarchy empty/all-nonnumeric sum conventions.

Route table-array and graph attribute statistics through checked native numeric
columns; propagate failures. Preserve explicit scalar/vector widths in callback
translation, metadata, table filtering, UID maps, external floating import,
pathfinding auxiliary values and graph position/smoothing adapters. Native f64
position vectors now work. Existing f32 getters/storage/compression remain typed.
Add checked explicit Python float32 construction and use native checked rounding
for float_vec finite-overflow rejection.

Native cases cover mixed widths, wide integers, IEEE masks, coherent sorting,
legacy aggregations/cancellation and graph position vectors. Python cases cover
array operators/inference, grouped means and explicit f32 conversion. Two older
root tests still expected the retired f32 output narrowing: update their grouped
std and matrix materialization expectations to the actual f64 contracts.

Executed native suites, strict Clippy and the in-tree extension. Before default
inference changes, the root test_*.py API baseline (excluding the owner's untracked
algorithm benchmark) passes 967 checks, one existing skip. Module legacy surveys
and other naming-pattern suites are not included in that root baseline. The
focused builder/pipeline gate also passes 173 tests with two existing skips.
Remaining default-inference glue and compatibility certification are 07.3a.3.

Executed 21 documentation examples. Current native release 1k/10k comparison/
sort samples are in `benches/float_consumers_2026-10-23.json`: sorted input, source
setup excluded, result allocation/destruction included, no concurrent build/test.
No before/after speedup, FFI latency or worst-case/random-sort claim is made.
Three existing FFI warnings remain; no packages installed.
