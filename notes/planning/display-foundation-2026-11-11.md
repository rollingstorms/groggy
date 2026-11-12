# Native display foundation — 2026-11-11

Completes native ownership/caller migration within existing **AR7.6 / AR7.11**,
with demonstrated rendering repairs under the existing numerical/API gates.
No new checklist rows are added; the remaining AR7.6 adapter inventory stays open.

## One native typed rendering owner

Move typed DataWindow/schema, display configuration/engine, compact/HTML
renderers, themes and their existing CSS assets from viz/display into
`src/display/render`. Native storage, matrix, graph and streaming callers now
refer to that owner. The base table no longer imports presentation types through
the misleading `core` compatibility facade. `viz::display` and `core` retain
public aliases to the same types, including old display submodule imports.
Type-identity/cross-import tests guard that compatibility.

The legacy root display JSON dictionary/config APIs remain separate frontends:
total-width/color settings do not mean the same thing as typed cell-width/theme
settings. No configuration field/default is dropped or guessed during the move.
Standalone historical Python formatting classes retain their layouts. Their
remaining adapters/shared scalar policies are explicit AR7.6 work; this slice
does not claim every historical renderer has been retired. All CSS assets move
with their owner; no new theme variants or dynamic wrapper engines are added.

## Reproduced failures and repairs

- HTML JSON rendering sliced bytes inside a Chinese UTF-8 character. Public
  native rendering failed at a narrow character budget. It now uses shared safe
  character clipping; HTML cell and tooltip content stay escaped.
- HTML signed-integer grouping negated i64::MIN, panicking in debug and missing
  grouping in release. Unsigned magnitude handles the complete signed domain.
  The public Python table regression also fails on the previous release build.
- Legacy zero row/column budgets underflowed subtraction; zero text budgets
  emitted an over-budget ellipsis. Saturating counts/explicit zero behavior keep
  the existing positive-budget layout. Compact tiny-width prefix presentation
  remains intentionally distinct from HTML JSON/legacy ellipsis presentation.

Native tests sweep each cutoff in a Unicode JSON fixture and inspect visible
escaped content, signed extremes, zero/maximum budgets, compatibility type
identity and a million-character source prefix. Shared clipping examines only
the visible budget plus one character, avoiding a new whole-property character
scan just to select a prefix. This is Unicode scalar width, not a grapheme or
terminal-glyph-width promise. Whole HTML still escapes its complete tooltip.

## Bounded release evidence and limits

`cargo bench --bench display_rendering`; output path through
GROGGY_DISPLAY_PROFILE. Five samples of 100 calls. The baseline and final ASCII-fast-path receipts are `notes/benchmarks/display-foundation-before-2026-11-11.json` and `display-foundation-ascii-final-2026-11-11.json`. Intermediate first-after/after and quiet-repeat receipts remain alongside them for transparency.
Rust 1.96.1, same release profile. All **12 ASCII output fingerprints and byte
lengths match** across compact/full Unicode and semantic HTML at 0/1/10/100 rows.
Fingerprints use DefaultHasher for this same-toolchain comparison only.

Full-render timings are mixed. Early post-build small cases showed multi-fold
spikes, but three additional quiet runs of the same bounded-character binary
returned near the baseline: one-row compact **1.847 → 1.935 μs**, one-row HTML
**7.172 → 6.888 μs** (median across repeat samples). A moderate 100-row full-width
cost remained (**55.434 → 63.996 μs**), motivating a bounded ASCII prefix fast
path in the shared helpers rather than scanning characters for ordinary text.

Final ASCII-fast-path 100-row medians are compact **64.069 → 59.692 μs**,
full-width **55.434 → 57.845 μs**, HTML **450.848 → 451.063 μs**. Small initial
post-build samples remain variable; no universal performance parity or causal
claim about machine scheduling/code generation is inferred. The existing 32.2/
AR7.6 performance gate retains that measured boundary without another checkbox.
The ASCII path checks only the bounded prefix, preserving Unicode safety and
avoiding a whole-input ASCII scan for large text.

The final benchmark also compares the previous safe text helper from 97705924
and the bounded helper **within the same binary**, with equal outputs at budget
20. At 1k input characters, median **0.398 → 0.125 μs**; at one million,
**142.943 → 0.128 μs**. This isolates prefix work rather than whole rendering,
which retains full tooltip escaping. Construction is excluded; rendering/prefix
allocations are included in wall time but not counted. There is no allocation/
RSS, Python GIL, broad throughput or production-readiness claim.

## Verification and remaining inventory

Full final native gate: **1,547 passed / 54 ignored**. Strict root all-target
Clippy passed. Native focused display tests pass; two failing pre-fix native
regressions and the pre-fix Python formatting failure are reproduced in the turn
logs. API and executable Python example: `docs/api/display-foundation.md`.
The final release extension was rebuilt in tree (three existing FFI warnings). Final maintained Python gate: **2,612 passed / 1 skipped / 15 warnings**; **261 executable docs** passed. One warning is the observed ownership/finalization issue documented below; the other 14 are the established builder validation warnings.

Remaining AR7.6 scope: historical Python formatter adapter migration, shared
legacy/typed scalar formatting policy where presentation intentionally differs,
and the broader display compatibility/performance exit gate. Existing topology/
typed-effect and numerical/storage scale parents retain their previous scope.

### Observed bridge lifetime warning (existing 19.2)

An intermediate full Python gate passed but emitted one extra unraisable-exception
warning: `PyGraph is unsendable, but is being dropped on another thread`. Pytest
reported it during a captured A* test; that collection location does not establish
the object's allocation path or cause. Other gates in the window had the usual
14 warnings. The final gate passed and emitted this warning again; it is not considered resolved. This requires
focused lifetime/GC investigation under existing 19.2, not a new checkbox or a
claim that the display renderer causes it. No unsafe Send workaround is added.
