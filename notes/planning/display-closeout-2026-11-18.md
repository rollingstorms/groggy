# AR7.6 closeout — 2026-11-18

**One existing checkbox closed: AR7.6.** No rows added or requirements removed.
Fixed inventory: **216 checked / 52 unchecked / 268 total**. Of the unchecked
rows, **43 are in the expanded 0.6.0 target and nine remain deferred**. Counts
include overlapping parents; they are not a completion percentage.

## Completion boundary

The existing row requires native display/configuration/formatting/truncation
ownership, thin language/interactive adapters, preserved intentional presentation
and working examples. Its earlier native type/theme move is retained; this pass
finishes the remaining Python adapters, shared scalar/selection compatibility and
bounded performance evidence. Whole-engine numerical/scale certification, graph
layout solver completeness and remaining public API migration retain their
existing 28.15/19.2/29/32/AR7.11 requirements.

| Surface | Canonical owner / retained adapter |
| --- | --- |
| Typed table/graph HTML and compact/full text | `display::render`; `core` and `viz::display` retain type-identical compatibility aliases and the existing theme assets. |
| Historical Python array/matrix/table geometry | `display::render::interactive`; Python sends selected owned scalar strings. No Python border/width/alignment renderer remains. |
| Row/column omission plans and text clipping | Native typed positions distinguish omitted slots from actual cell values; native compatibility helpers project the same plan without marker slots. Unicode-safe clipping is shared. |
| Dictionary dtype spelling | `display::format_dtype`, with explicit legacy size-hint policy. The old hints remain presentation labels, not measured cardinality or schema guarantees. |
| Python representation | One private `_presentation` adapter retains Python object `str`, integer widths, f/g float formatting and context-specific null/boolean spelling. Imports are resolved once rather than for each cell. |
| Configuration | Existing native dictionary and typed configurations retain their distinct field meanings. Python constructor defaults/explicit overrides share one settings adapter; `configure_display` is now effective for new dictionary formatters. |
| Terminal capability / ordered summary | Thin Python adapter; GIL retained for Python callbacks and input projection. Only complete owned text/configuration enters GIL-released native layout work. |

The public dictionary function/class imports remain. Normal defaults and the
intentional native JSON versus Python presentation distinctions remain; no
replacement theme, facade engine or graph-computation implementation is added.
Private old Python border/row helpers are removed. Full policies and executable
examples: [display foundation](../../docs/api/display-foundation.md).

## Corrections within this requirement

- Actual source row indices for tiny prefixes; typed omission positions instead
  of guessing from a literal ellipsis or `['…']` value.
- Tail dtype headers follow their selected column names; ragged table gaps are
  empty rather than shifting later values into another column.
- Single bounded date ellipsis; UTF-8 safe native array/matrix clipping; native
  dictionary matrix `max_rows`; native tiny-budget compatibility prefixes.
- Native dictionary bool translation precedes integer conversion. Nonfinite
  numbers display through null instead of invented zero. Out-of-range Python
  integers retain spelling rather than silently converting through f64; existing
  NumPy numeric protocols remain accepted. Visible native object `str` failures
  propagate their Python exception rather than raising a formatter panic.
- Select Python table cells before calling scalar formatting. Hidden Python
  objects are not evaluated. Complete configuration validation precedes new
  defaults; explicit options win, existing instances stay independent.
- Rust workers own their input strings/configuration and release the GIL for
  both interactive layouts and native dictionary rendering. No Python callback,
  live graph handle or borrowed Python string enters those workers.

Text budgets count Unicode scalar characters, not terminal/grapheme width. Lone
Python surrogate text rejects explicitly. Total width retains the historical
readability minimum and is not a hard line-width ceiling; array/matrix max_width
retains its prior compatibility meaning. Native JSON materialization can still
scale with supplied input size. No storage/history format or graph mutation
contract changes.

## Compatibility and measurements

Captured baseline: `37f1ff2d`. **61 byte-exact historical Python layouts** cover
empty/tiny/normal/truncated limits, Unicode, object/bool/large integer spellings,
nonfinite/signed zero, width allocation and the three formatter families. Corrected
indices/type headers have separate semantic regressions. The initial 72-case
survey identified eight array-index/omission and eighteen truncated-type-header
changes; these are intentional repairs, not silently approved snapshot changes.

**All 12 existing native ASCII hashes and byte lengths match** the previous
compact/full/HTML corpus. Three fresh quiet native processes each retain five
samples of 100 calls, with the same release toolchain/harness. Small timing
variation is not interpreted as a causal speedup or a production budget.

Python comparison: three fresh processes, seven warm time observations per
implementation/case; twelve cases at 1/10/1,000/100,000 rows. Input construction,
imports, output checking and separate allocation observation are excluded from
timing. `tracemalloc` reports **Python traced allocations only**, not native
allocations, allocator retention or RSS. Medians below combine 21 observations.

| Fixture | Previous time | Final time | Previous / final traced peak |
| --- | ---: | ---: | ---: |
| Array, 1 value | 8.13 µs | 7.50 µs | 1,252 / 1,022 bytes |
| Array, 10 values | 32.21 µs | 23.46 µs | 3,117 / 2,148 bytes |
| Matrix, 10×6 | 77.00 µs | 58.67 µs | 8,128 / 11,500 bytes |
| Table, 10×6 | 120.88 µs | 83.00 µs | 15,515 / 19,000 bytes |
| Table, 100,000×6, visible budget 10 | 351.526 ms | 84.96 µs | 42,198,034 / 18,885 bytes |

Small-call timings vary across observations; the final one-element median is
7.50 µs and is not a zero-overhead or latency guarantee. Some small traced
Python peaks grow while complete layouts marshal owned strings; native allocation
is excluded from that measurement. Large-table improvement comes from avoiding
hidden-cell formatting, not graph loading or query acceleration. It is not a
whole-engine speed or memory claim.

The first adapter version repeated imports per cell and made matrix rendering
slower (10×6: 75.17 → 181.75 µs in that comparison). Those initial three receipts
are retained explicitly. Moving Python-only presentation helpers into one
noncyclic private adapter with module-level imports removed that observed cost;
the final matched matrix comparison above improves instead. No precision or
presentation contract was changed for the optimization.

Reproduce with `notes/scripts/benchmark_display_adapters.py --baseline 37f1ff2d
--output <receipt>` and `cargo bench --bench display_rendering`. Raw receipts:

- `notes/benchmarks/display-closeout-python-2026-11-18-{1,2,3}.json`
- `notes/benchmarks/display-closeout-python-initial-2026-11-18-{1,2,3}.json`
- `notes/benchmarks/display-closeout-native-2026-11-18-{1,2,3}.json`

Receipts retain source/harness hashes; the final measured native module matches
the root qualified wheel byte-for-byte. Typed native rendering keeps its existing
same-toolchain fingerprints. Table/matrix output comparisons match baseline;
truncated array fingerprints deliberately change with corrected index/omission
labels.

## Executed local gates

- Full native: **1,633 passed / 54 ignored / zero failed**, including 10 doctests;
  separate serial killed-checkpoint recovery: **one passed**.
- Strict root all-target Clippy, root formatting, selected FFI formatting and
  `git diff --check`: passed. Ruff formatting and F/E402 checks pass for changed
  Python modules/tests/harness. Black/isort/pre-commit remain unavailable locally;
  no missing tool is credited as passed.
- Rebuilt in-tree release extension; three established FFI warnings remain.
- Fresh unpacked macOS arm64 CPython 3.9 wheel, isolated smoke/105 exports/version
  and archive/journal workflow; **3,385 maintained Python passed / one skipped /
  14 established warnings**; **290 executable documentation examples passed**.
  Packaged display demos (three modules) and all six display README Python blocks
  also execute with site packages disabled; placeholder schema examples now use
  concrete valid values. All 62 packaged Python/stub files match source.
- Focused display: **80 new Python cases** plus the retained HTML boundary case;
  four new native tests in the eight-case native display suite. New coverage
  includes numeric protocol/UTF-8 rejection, worker-owned text, configuration
  atomicity, no hidden callbacks and identity-preserving compatibility projection.

Wheel receipt:
`notes/benchmarks/display-closeout-wheel-qualification-2026-11-18.json`.
Platform qualification is tracked separately; the older e517e639 run does not
qualify this new runtime. No release/tag/package publication is performed here.

## Remaining work

No known unfinished part of the stated AR7.6 completion boundary is moved to a
new row. Existing broader engine/API/algorithm/scale/service requirements remain
open at their original scope. The pending provisional-entity history question,
schema-history policy and production budgets are unchanged and unanswered.


## Successful platform follow-up

Runtime **cfc5628c** passed manual run
[44403061335](https://github.com/rollingstorms/groggy/actions/runs/44403061335):
all 21 required jobs, 19 wheel combinations and extracted-source rebuild;
publication skipped. The platform receipt retains 20 qualification reports and
40 upload IDs/digests. The downloaded source archive is 5,302,196 bytes, SHA256
`9f43fa54508510f2ad8133db08aa5a3fccf8f7f862fd67307de9faf3ad8e02f4`;
all 497 production source files match current bytes and Finder metadata is absent.
CI-regenerated lockfiles are retained with actual hashes, separately from local
locks. The source archive precedes later capability/workflow documentation, which
has no production changes.

Only Linux Python 3.12 executes the maintained gate; the other 18 wheels use
ownership plus isolated exports/storage workflows. Extracted-source testing uses
`tests=none` with isolated smoke/storage workflows. These distinct gates are not
credited as full maintained coverage on every platform. Evidence:
`notes/benchmarks/display-closeout-platform-qualification-2026-11-18.json`.
