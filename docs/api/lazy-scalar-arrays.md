# Checked lazy scalar arrays (native)

`LazyArrayIterator` defers a sequence of scalar operations and applies it in the
order written. `filter`, `take`, `skip`, random `sample`, and explicit
`sample_seeded` are supported. Filters compile once per collection and preserve
original values, types and order. Filtering and slicing reuse the owned vector's
allocation. A returned Vec may retain its input capacity; callers can use
Vec::shrink_to_fit when compact long-lived output matters. Seeded sampling uses
a private RNG and preserves the existing
reservoir output convention; it does not promise sorted sample output.

```rust
use groggy::storage::array::LazyArrayIterator;
let selected = LazyArrayIterator::new(vec![1_i64, 2, 3, 4, 5, 6])
    .take(4)
    .skip(1)
    .filter("value % 2 == 0 AND value >= 2")
    .collect()?;
assert_eq!(selected, vec![2_i64, 4]);
let sample = || LazyArrayIterator::new((0_i64..100).collect())
    .sample_seeded(5, 42).collect();
assert_eq!(sample()?, sample()?);
# Ok::<(), groggy::GraphError>(())
```

Scalar filters accept `AttrValue`, i32/i64, f32/f64, bool and usize elements.
A usize outside the supported i64 property-value range rejects explicitly.
`collect_as_array` shares the same validation/execution path. No graph history is
written. `ArrayOps.lazy_iter()` still copies its source into an owned vector;
this is deferred vector execution, not a streaming or zero-copy graph view.

## Shared grammar and explicit array semantics

Array `QueryEvaluator` and `BatchQueryEvaluator` use the graph query lexer and
boolean precedence/parenthesis parser. Query construction remains infallible for
native compatibility; `validate`, `evaluate` and batch collection surface errors.
Invalid queries reject even for empty input, after `take(0)`, or in a boolean
branch that would not be evaluated for any row. The complete plan is validated
before execution. Unknown expressions no longer pass all rows silently.

- `value` compares the current logical scalar; `length` reports UTF-8 byte length
  for text and element count for vector/byte/ID collections. Other values have
  length zero. Compact/compressed representations preserve logical length.
- Numeric ordering shares the exact scalar comparator, including mixed i64/f64
  boundaries and IEEE unordered NaNs. Array numeric equality remains mathematical
  (`2 == 2.0`); graph property equality retains type identity. Floating literals
  default to f64; explicit f32 suffixes use the shared literal rules. Array
  infinity/NaN literals preserve the existing IEEE behavior; graph query literals
  retain their finite-only policy.
- `AND`, `OR`, `NOT`, parentheses, quoted/escaped strings and boolean constants
  are parsed as tokens. Quoted `AND`, `OR`, or comparison characters are text.
  Array compatibility still accepts bare text on a comparison's right side.
- Array text ordering remains lexical. Boolean equality stays separate from
  numeric comparison. Mismatched scalar domains do not match, including `!=`;
  this intentionally retains the array policy rather than expanding graph query
  ordering domains. Null is an explicit value, not an absent array slot.
- Existing `value % 2 == 0` shorthand works within boolean expressions. Other
  arithmetic/remainder expressions reject. This extension does not add modulo
  filtering to the graph query language.

`estimated_len()` is a safe upper bound computed in statement order; filters do
not halve it heuristically. `is_empty()` reports a guaranteed empty upper bound,
not the result of executing a predicate. Exact cardinality requires collection.

## Unimplemented families

Generic lazy graph predicates and type-changing map/collapse/expansion/grouping
operations return an explicit error at collection, including on empty sources.
They previously returned pass-through, random or fabricated empty results.
Use checked graph views and the concrete graph/collection operations for those
workflows. Generic eager marker-trait iterators retain separate unfinished
migration work. This does not complete streaming, cancellation, typed graph/table
intermediates or the whole pipeline builder (30.1/AR7.8).

Compatibility: `Token::Remainder` and `LazyOperation::SampleSeeded` add enum
variants; downstream exhaustive native matches need to account for them. Existing
query constructors and collection result types remain available. Native lazy
iterators are not a Python API addition in this slice.
