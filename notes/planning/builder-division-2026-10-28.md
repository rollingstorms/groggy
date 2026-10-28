# Exact zero checks for builder division — existing 30.1 component

Completed the interrupted tiny-divisor repair. The native binary arithmetic
helper used machine epsilon as a zero cutoff, rejecting valid expressions such
as 1e-20 / 1e-20. It now rejects exact positive/negative zero. Finite tiny nonzero
and subnormal divisors follow ordinary f64 division. Explicit reciprocal epsilon
regularization retains its separate contract.

One native regression covers node/edge maps, positive/negative normal and smallest
subnormal divisors, signed zero and preserving an existing output on error.
Twenty-four new Python cases cover optimized/unoptimized and ordinary/atomic
execution, cross-language subnormal conversion, signed zero, and atomic failure
preserving graph/history and discarding a staged prefix. The relevant math file
now has 92 passing cases. A fourth executable block in
`docs/api/builder-elementary-math.md` demonstrates tiny division.

Final gates: **1,417 native passed / 56 ignored**, **2,236 root Python passed /
1 skipped**, **208 executable documentation examples**, strict all-target core
Clippy, scoped Rust formatting, Python syntax and diff checks. The existing
CPython 3.9 full-ABI release extension was rebuilt and atomically staged in tree;
three pre-existing unrelated FFI warnings remain. Root Python verification
excludes the owner's separate algorithm benchmark and nested historical surveys.
No external packages or graph libraries were installed/used. Owner benchmark/
notebook/review changes remain excluded from this commit.

Native commands use `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target`
with offline locked `cargo test`, targeted `--test builder_unary_math`, and
`cargo clippy --all-targets -- -D warnings`. Python verification runs the root
contract suite and the existing executable documentation runner.

This is a correctness fix within the existing builder inventory; no new checklist
row or general arithmetic certification claim. Fixed inventory remains
**210 checked / 58 unchecked**. Other typed effects, alternate plans, unfinished
facades and compound topology acceptance retain their existing scope.
