# Attribute statistics closeout — 2026-10-22

Checklist 29.61a closes quantile, Pearson/Spearman/Kendall correlation and
covariance on the canonical attribute BaseArray. Window, cumulative and table
statistics remain separately open in 29.61.

## Preserved conventions

- Numeric inputs are Int, SmallInt and Float. Null and nonnumeric cells are
  omitted; paired operations omit a row unless both cells are numeric.
- Results retain attribute Float (f32) storage. Normal representable rounding is
  permitted; overflowing/nonfinite results return errors.
- Empty quantiles and fewer than two correlation pairs return Null. Constant
  Pearson/Spearman inputs return zero. Kendall remains tau-a (ties contribute
  zero, denominator includes every pair), and Spearman uses average tie ranks.
- Covariance uses N-ddof and returns Null when there are insufficient pairs.
  Quantile nearest ties select the higher value; all five interpolation modes
  remain supported.

## Corrected failure and precision behavior

Paired input lengths must match; negative ddof, nonfinite numeric inputs and
invalid quantile methods/probabilities return errors. Quantile arguments are
validated even for empty inputs. This replaces silent truncation, invalid empty
success, NaN ranking stalls and unchecked f32 overflow.

Exact numeric ordering precedes conversion. Integer/float residuals are retained
for centering and interpolation; finite product accumulation uses the common
numeric accumulator. Scalar's default difference hook preserves existing typed
reduction behavior while allowing mixed attributes to retain small differences
near the i64 boundary. This is not a promise of exact arbitrary real arithmetic.

## Verification

Native regression fixtures cover i64 extremes, mixed integer/f32 residuals,
compact integers, missing cells, ties, constant input, invalid empty arguments,
shape/ddof errors, nonfinite inputs and narrowed covariance overflow. Python
fixtures exercise the rebuilt FFI surface against rational/tie references.

Executed: `cargo test --offline --locked --quiet` (full native suites),
`cargo clippy --offline --locked --all-targets -- -D warnings` (clean), rebuilt
FFI extension with locked offline cargo rustc, and the expanded numerical/array
Python gate including the new attribute statistics file: **466 passed**.
