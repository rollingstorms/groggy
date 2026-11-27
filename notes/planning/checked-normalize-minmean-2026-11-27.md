# Checked minimum and mean normalization — 2026-11-27

Finite component of existing 29.3, 30.1, 09.4b.2 and AR7.8. No stable checklist
row is added: 217 checked / 51 open (42 core, nine deferred).

- [x] Add signed minimum and arithmetic mean to shared native normalization.
- [x] Lower both register-plan options through those same sealed kernels.
- [x] Preserve selected node/edge maps, aliasing, finite controls and atomic output installation.
- [x] Check negative/tiny/zero divisors, empty inputs, nonfinite values and overflow.
- [x] Compare memory, worker and recovered values, including independent arithmetic expectations.
- [x] Qualify the packaged surface and retain explicit remaining boundaries.

Normalization divides by the signed statistic and rejects absolute divisors at
or below epsilon. It does not clamp negative divisors. All intermediate arithmetic
must remain finite; scaled extreme-value arithmetic is not claimed. Empty sum
rejects, other empty normalization modes return empty maps. Legacy uncertified
mixed plans keep their historical interpreter behavior. Edge-aware registers
remain unresolved; this changes options within one opcode, not the 35/36 opcode
inventory or the broad completion status.

Evidence: expanded `tests/checked_batch_lowering.rs` worker/memory/reopening matrix,
16 new Python cases in `tests/test_checked_batch_lowering.py`, and native control,
value, cancellation/output and independent arithmetic tests. The first Python
recovery fixture incorrectly created nodes at wall time before querying timestamp
19; creation now uses an explicit effective-time-zero batch. Full qualification
was rerun after that correction.

Local gates: **1,707 Rust tests** (54 ignored), **3,657 Python tests** (one skipped,
23 expected warnings), **304 documentation examples**, strict all-target Clippy,
formatting, Python 3.8 syntax and all 62 packaged Python/stub bytes. All 500
production source files match runtime `4be4c415` (same production tree as
`5155a878`). [Local receipt](../benchmarks/normalize-minmean-local-qualification-2026-11-27.json)
and [wheel receipt](../benchmarks/normalize-minmean-wheel-qualification-2026-11-27.json).
No new performance or cross-platform qualification claim.


## Completed control-presence follow-up

Source `1ffdf5eb` closes a concrete factory gap found during review: malformed
explicit epsilon, method or clip-alias values used to become defaults. Defaults
now apply only when omitted. The two normalization aliases share method/epsilon
parsing; standardization uses the same epsilon checks and clipping uses the same
kernel bound validator. Both explicit clip spellings validate; valid
`min_value`/`max_value` precedence remains. Invalid numeric controls reject during
preparation, including a zero-iteration ordinary body. Construction may reject
earlier than execution. Nested bodies retain existing uncertified-loop diagnostics;
no unsupported body is silently skipped. Direct Rust constructors still validate
at execution as before.

Clipping also checks its computed output: equal infinite bounds cannot install
nonfinite values from finite inputs. Conventional outward-unbounded clip ranges
still work; source-dependent arithmetic validation remains at computation time.
Two native registry tests and extended kernel/old-output cases cover nonfinite
controls/results. Forty-seven additional Python cases cover explicit wrong types,
null versus omitted controls, both aliases, zero/one iteration, integer epsilon,
valid precedence and no changed journal/properties/IDs.

Fresh local gates pass **1,716 native tests** (54 ignored), **3,753 packaged Python
tests** (one skipped, 23 expected warnings), **306 documentation examples**, strict
all-target Clippy, formatting, Python 3.8 syntax and all source/artifact byte checks.
All 500 production sources match `1ffdf5eb5e15abcd5fdd564634a2bb826215830a`.
[Final local receipt](../benchmarks/normalization-final-controls-local-qualification-2026-11-27.json)
and [final wheel receipt](../benchmarks/normalization-final-controls-wheel-qualification-2026-11-27.json).

Final code [CI run 46079596058](https://github.com/rollingstorms/groggy/actions/runs/46079596058)
is separate qualification evidence. The prior control candidate 46079465175 was
cancelled as superseded after the clipped-result guard changed production source;
it is not counted as a successful gate. The existing normalization checklist item
is completed more thoroughly; no new stable roadmap row was introduced.
