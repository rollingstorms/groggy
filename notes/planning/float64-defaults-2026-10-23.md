# 07.3a.3 — Ordinary Python f64 defaults

Implements confirmed workbook A06.2/A13.2. Ordinary Python floats and numeric
float collections now infer Float64/Float64Vec, including empty sequences.
Explicit AttrValue f32 constructors, NumPy float32 inputs and existing tagged
f32 storage retain their width. Integer/Boolean/text/structured policies remain.

Reserved empty labels normalize to TextVec. BaseTable validation accepts logical
floating families and explicit widths. Algorithm FFI parameters delegate to the
existing native parameter converter, fixing the previously missed wrapped f64
scalar/vector path. No parameter conversion policy is duplicated in FFI.

Updated precision references exercise the values now actually stored, including
sub-f32 differences. A fixed older-format f32 scalar/vector document checks
load, snapshot and re-export without routing fixture generation through new
inference. Direct/bulk node and edge writes, table projections, snapshot
isolation and checked journal recovery retain f64 precision and range.

The broad run also exposed a stale directed-adjacency test assumption: graph ID
iteration is unordered, while matrix coordinates are canonical ascending IDs.
The regression now follows the documented matrix contract; engine ordering was
not changed to satisfy a nondeterministic test.

## Verification

- `CARGO_INCREMENTAL=0 cargo test --offline --locked --quiet` passed: 896 tests; 52 ignored across unit, integration and doc suites.
- `CARGO_INCREMENTAL=0 cargo clippy --offline --locked --all-targets -- -D warnings` passed.
- In-tree FFI debug library rebuilt with macOS dynamic-lookup linker arguments;
  three existing unrelated FFI warnings remain.
- Root `tests/test_*.py` API baseline, excluding the owner's independent
  `test_algorithm_benchmark.py`: **976 passed, 1 existing skip**. This is not a
  claim that legacy `tests/modules/` survey scripts were certified.
- **22** Python documentation examples passed against that extension.
- Release numerical corpus: **55 cases**, checksums identical to all 55 prior
  f64-output cases and the original 25-case completion receipt. Raw capture:
  `benches/numerical_float64_defaults_2026-10-23.json`. Timings are observations,
  not a new speedup claim or performance SLA.

Black is unavailable in the existing environment; no external formatter or
package was installed. Changed Rust regions were formatted and whitespace
checked. The f64 default closes 07.3a. Remaining 07.3 temporal/reference work,
publication policy migration and immutable legacy-view parents remain open.
