# Normalization safety — 2026-11-25

Finite correction within existing 29.3, 29.4, 30.1 and AR7.8. The ordinary
normalization helpers accepted invalid controls and nonfinite values despite the
checked numerical contract used elsewhere. Three targeted regressions failed
before the fix; this is a demonstrated discrepancy, not a new feature request.
Flat counts remain 217 checked / 51 open while the larger numerical/execution
parents remain partial.

## Completed component checks

- [x] Reject negative/nonfinite epsilon even on empty maps.
- [x] Reject NaN and reversed clip bounds before clamp is called; retain unbounded sides.
- [x] Reject nonfinite inputs and arithmetic intermediates/results.
- [x] Preserve signed max, population z-score and established empty-map behavior.
- [x] Preserve old node/edge target maps on failure, including the node-name alias.
- [x] Verify public node/edge map operations, staged write/ID rollback and
  effective-time recovery for valid results.

## Contract and limits

The existing algorithms remain in Rust and are used by both owner and sealed
read execution. Validation helpers are shared within the normalization module;
FFI contains no numerical rules. Results are assembled locally and installed only
after success. Existing finite, valid inputs retain the established algorithms
and f64 outputs. No new normalization modes or implicit epsilon clamping are
introduced.

Inputs and epsilon must be finite; epsilon is nonnegative and clip bounds are
ordered and not NaN. Infinite clip bounds remain valid for unbounded sides, as
required by the existing registry defaults. Floating overflow or another nonfinite intermediate/result rejects.
This intentionally bounds the current f64 algorithms: some extreme inputs whose
mathematical normalized ratios would be finite still require a future scaled
kernel to avoid unrepresentable intermediates. They now produce an explicit
error rather than a misleading zero/NaN. No throughput improvement or complete
extreme-value normalization claim is made.

Signed max and population rather than sample variance remain unchanged. Valid
empty max/minmax, standardization and clipping return empty maps; empty sum
normalization still rejects its zero denominator. Register-plan min/mean options,
retained-history producers and other composition work remain in existing parents.

## Evidence

Five native safety tests cover invalid controls, nonfinite inputs, overflowing
intermediates, valid definitions, empty cases and output preservation for both
map kinds. The first baseline run had one pass and three failures before the
correction (the output-preservation test was added afterward).

`tests/test_normalization_safety.py` exercises 28 public node/edge cases: all
existing normalization methods, z-scores, bounded/unbounded clipping, late rejection after staged
writes/provisional creation, journal bytes, ID membership and effective-time
reopening. The public contract/example is in
[durable pipelines](../../docs/api/durable-pipelines.md).

Local gates pass **1,695 native tests** (54 ignored), **3,597 packaged Python
tests** (one skipped, 23 expected warnings), and **302 documentation examples**.
Strict all-target Clippy, formatting, whitespace, Python 3.8 syntax and all 62
packaged Python/stub byte comparisons pass. All 500 production source files
match runtime `6ec32ed1959543036f638abedf0dbe23df21a6f7`. See the [local receipt](../benchmarks/normalization-safety-local-qualification-2026-11-25.json)
and [wheel receipt](../benchmarks/normalization-safety-wheel-qualification-2026-11-25.json).
The recovered isolated Python environment's actual build is recorded. Earlier
platform and alias measurements remain pinned to their own candidates; new
normalization measurements and candidate CI are recorded separately.

## Matched valid-input timings

[Raw observations, matched source/build and artifact hashes](../benchmarks/normalization-performance-2026-11-25.json)
compare qualified sampling runtime `bd1c42ff` with numerical safety runtime
`6ec32ed1`. Only `steps/normalization.rs` differs among production sources; Cargo
inputs and the Python build match. Three processes per runtime run in alternating
order, yielding 21 observations per case. Local builds/tests are finished before
formal measurements. The earlier smoke under builds is excluded.

The 24 workloads cover 1k/10k nodes, one/ten invocations, five numerical operations
and an alias control. Setup/compilation/warm-up are excluded; public transient
execution, final attachment and Python output conversion are included. Every
output is checked independently and graph history stays unchanged.

For 10k rows and ten invocations:

| Method | Before ms | After ms | Median change |
| --- | ---: | ---: | ---: |
| control | 4.387 | 4.096 | -6.6% |
| sum | 11.455 | 11.688 | +2.0% |
| max | 11.925 | 12.244 | +2.7% |
| minmax | 12.475 | 12.582 | +0.9% |
| standardize | 7.176 | 7.496 | +4.5% |
| clip | 7.221 | 6.478 | -10.3% |

Four normalization/standardization medians are 0.9–4.5% higher; clipping is 10.3%
lower and its matching control 6.6% lower. Across fixtures control changes range
from -6.6% to +26.2%, so these are descriptive observations, not an isolated
per-kernel causal estimate or a general speed claim. No allocator/RSS, storage
sync or production SLA is measured; A15 budgets and broader numerical throughput
remain open. Correct rejection behavior is retained regardless of these timings.
