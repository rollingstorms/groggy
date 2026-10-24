# Conditional message-pass execution — 2026-10-24

Closes **09.4b.2j**, extending **30.1 / AR7.8**.

## Implemented contract

- Execution blocks now implement `compare` with the same six operators and numeric
  0.0/1.0 masks as ordinary builder steps. Shared CompareOp prevents a separate
  block-specific comparison rule.
- Replace placeholder `where` with condition/true-value/false-value selection.
  It preserves selected values' types and resolves only the selected value at
  that operation. Earlier body computations remain eager and ordered.
- Ordinary node/edge map selection and message-pass selection share one native
  mask helper. Preserve the existing `abs(value) > f64::EPSILON` convention;
  nonnumeric masks error. Existing f64 comparison conversion/tolerance remains,
  including precision limits for large integers; this is not an exact-comparison
  migration or a new truthiness policy.
- Validate conditional arity and comparison operators during preparation, before
  any atomic outer step executes. Incorrect old two-input where specifications,
  unknown operators and non-text operator metadata reject explicitly.
- Certify supported conditional message-pass blocks for the shared outer staging
  boundary. A runtime error discards earlier staged attachments. Regular legacy
  blocks also receive the corrected conditional behavior.

## Remaining work

Synchronous/unordered message passing and streaming remain uncertified. Other step
families, alternate batch plans, effect-aware optimization, default atomic policy,
outer pipeline transactions and historical run/provenance policy remain parent
work. Previous notes identifying placeholder where as open are superseded by this
slice; no unrelated roadmap parent is marked complete.

No new graph storage, graph clone, index, dependency or FFI business logic is
introduced. This repair adds the existing comparison/selection operations to the
ordered body evaluator. It makes no speedup, allocation or throughput claim;
existing fused/message-pass release receipts remain bounded observations of their
own workloads.

## Verification

Focused Python coverage: all six comparison operators through the public builder,
normal/block parity at mask epsilon boundaries, malformed arities/operators,
typed selected values, missing selected values, invalid masks and rollback of
prefix attachments. Native validation tests cover supported conditional bodies
and malformed specifications. The public documentation includes a runnable
conditional block example. Full gate results are recorded below.

- Full offline/locked native suite: **941 passed, 52 ignored**.
- Strict all-target native Clippy with warnings denied: passed.
- Rebuilt/staged in-tree extension using existing macOS dynamic lookup settings;
  three existing FFI warnings remain. No package installation.
- **50 focused atomic-builder tests passed**.
- Full root Python API gate: **1,126 passed, 1 skipped**, excluding the unrelated
  owner benchmark test and the separate legacy modules survey.
- **40 executable documentation examples passed**.
- `git diff --check` passed; owner development/benchmark changes remain separate.
