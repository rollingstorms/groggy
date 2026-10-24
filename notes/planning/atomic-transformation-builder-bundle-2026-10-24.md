# Checked expression and transformation bundle — 2026-10-24

Closes **09.4b.2r / 09.4b.2s / 09.4b.2t**, extending **30.1 / AR7.8**.
Continues the owner's requested approximately 3x larger delivery cadence. These
are repairs/extensions to the existing engine; broader execution-boundary parents
remain partial. No new policy decision or approval gate is introduced.

## 1. Checked expression evaluation (09.4b.2r)

Integer add/sub/mul/remainder/power and unary negation/absolute-value reject
overflow; division checks actual zero in every numeric combination. Valid tiny
float divisors are accepted. Structured AST remainder/power now work rather than
falling through as unsupported. Nonfinite scalar/list values and results reject;
invalid domains therefore cannot publish NaN/Inf silently. Float/mixed arithmetic
retains ordinary f64 rounding. Comparisons share exact integer/float ordering with
property filtering; min/max preserve the selected numeric value's type.

Float sum and array mean share the checked native aggregation accumulator.
Integer array sum uses i128 intermediate accumulation with a checked i64 final
result. Boolean and/or short-circuit and conditionals evaluate their selected
branch only. Mode float ties use numeric ordering rather than signed bit order.
Empty-array conventions stay explicit in the API documentation.

Fixed a cross-language mismatch uncovered by executable regressions: the Python
parser emitted literal JSON constants while Rust required tagged parameter values.
Native Expr accepts both forms without narrowing integer literals; oversized raw
integers reject. Native serialization still round-trips its existing tagged form.
Python parser matches consume the full input, keep integer literals as integers,
and reject unsupported suffixes. Native neighbors/count functions now implement
expressions already emitted/documented by Python. Alias rewriting visits all AST
child fields; intended async source redefinitions no longer emit a false warning.

## 2. Scoped maps and failure safety (09.4b.2s)

Certify declarative expression mapping for atomic property builders and nested
loops. Property reads consult the shared staged column buffer, including explicit
nulls; transient attachments remain owned outputs without simulating a graph.
Public ExprContext fields/constructors remain unchanged: a private evaluator
carries run-specific property/variable/topology overlays.

Synchronous maps borrow existing variables and install results after success.
Async maps clone only their source map once, evaluate in ascending selected node-ID
order against earlier local updates, then install source/target together. Other
variables and the owner/history are borrowed. Source=target avoids an extra final
copy; async evaluation no longer builds an unused second result map.

Scalar maps validate active selected nodes without scanning edges. Topology-using
expressions prepare checked selected outgoing adjacency once (both directions for
undirected graphs), validate active edges/endpoints and sort neighbor IDs. Stored
self-loops contribute once. Preserve the distinct contracts: neighbor IDs/counts
retain parallel incidences; neighbor_values deduplicates contributors, includes a
stored self-loop, omits absent rows and flattens numeric lists. Invalid present
values reject. Mixed neighbor arrays promote to FloatList with f64 precision.
Standalone neighbor_agg remains the per-edge exact scalar aggregation alternative.

Opaque Rust callbacks remain uncertified because StepInput exposes owner objects.
Their errors now propagate, including when a source cell exists, rather than
silently retaining that value. Native regression confirms no result is installed.

## 3. Neighbor labels and explicit copies (09.4b.2t)

Certify neighbor_mode_update and update_in_place. Both prepare private targets and
check cancellation before installation, so errors do not leave partially updated
variables. Label propagation remains asynchronous for both ordering settings;
traversal is now stable ID order. Directed votes use outgoing selected edges,
undirected votes both ways, parallel votes stay distinct and loops vote once.
include_self adds a separate self vote. Typed serialized category keys replace
Debug strings. Keep uses first winning vote; lowest/highest use shared exact
comparison, with a serialized-key secondary tie for equal numeric typed values.
Nonfinite labels and incomparable tied scalar domains reject explicitly.

Copy updates validate sparse source IDs against the active selection, retain other
target values and support source/target/output aliasing. They scan no edges.
Supplied ordering/async/include-self flags, output names and mode tie selectors
must have the declared type/value; typos no longer silently select defaults.

## Costs, compatibility and remaining work

Private async/copy/label maps cost O(source/target cells); distinct output aliases
require a copy. Selected adjacency preparation costs O(V+E) storage plus row
sorting; numeric expressions allocate only their required values/results. Type
serialization and checked accumulators add correctness work. No owner/history
clone or FFI business logic is introduced. Individual expression kernels/large
rows/sorts do not provide bounded cancellation latency.

Historical malformed inputs that silently fell back, callback failures that were
swallowed, nonfinite results, and unconsumed expression suffixes now error.
Previously unordered label traversal now has deterministic results. Public string
expressions remain a deliberately limited subset, not an arbitrary Python parser.
Atomic=False defaults, optimizer property effects, outer multi-algorithm
transactions, native mutable escapes and remaining uncertified community,
sampling, temporal/path families remain open.

API/examples: `../../docs/api/builder-transformations.md`.

## Verification and release receipt

Final gates: **958 native tests passed, 52 ignored; 1,390 root Python API tests
passed, 1 skipped; 52 executable documentation examples passed; strict all-target
core Clippy passed.** The focused transformation/LPA/atomic/core builder run
passed 223 cases. Native test and Python extension builds used offline locked
dependencies and in-tree artifacts. Root Python excludes the owner benchmark
test and the separate legacy modules survey.

Release harness: `benches/transformation_builder_steps.rs`.
It measures six native modes at 1,000/10,000 nodes/edges with five observations
each, excluding fixture/spec preparation and including transformation, property
attachment and one checked publication. Every final property/publication is
validated. This is not a matched speedup or FFI/RSS/allocation/production-scale
claim.

Raw receipt: `../benchmarks/transformation-builder-bundle-2026-10-24.json`.

| Mode | Nodes (= edges) | Median ms, five observations |
| --- | ---: | ---: |
| scalar_map | 1,000 | 1.291 |
| staged_attr_map | 1,000 | 2.409 |
| neighbor_map_sync | 1,000 | 1.538 |
| neighbor_map_async | 1,000 | 1.464 |
| neighbor_mode | 1,000 | 1.521 |
| update_in_place | 1,000 | 1.267 |
| scalar_map | 10,000 | 13.334 |
| staged_attr_map | 10,000 | 24.801 |
| neighbor_map_sync | 10,000 | 15.334 |
| neighbor_map_async | 10,000 | 14.586 |
| neighbor_mode | 10,000 | 15.336 |
| update_in_place | 10,000 | 12.576 |
