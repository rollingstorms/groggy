# 07.3a.2h — Graph query literal precision and validation

The graph parser rounded decimal tokens to f32, omitted exponent/leading-fraction
syntax and returned parsed prefixes without requiring token exhaustion. Numeric
literals now default to f64, including decimal/scientific notation and explicit
signs. f32/f64 suffixes select floating storage identity. Reject malformed tokens,
nonfinite/overflowing numeric results, incomplete strings and trailing input.
Keep unsuffixed integer i64 bounds and compact inference.

Preserve AttributeFilter equality/inequality as typed property identity; do not
silently substitute the array/table numeric equality contract. Explicit f32
suffixes provide queries against retained f32 values after the literal default
changes. Ordering already uses the shared exact scalar comparator. Neither stored
value tags nor ordinary Python property inference changes in this slice.

Native tests verify precise f64 and wide-integer comparisons, explicit f32
identity, exponent/sign/fraction syntax, both node/edge kinds and live/frozen
state behavior. Error cases cover width overflow, malformed/unknown suffixes,
trailing tokens/parentheses, incomplete strings, integer limits, parser reuse,
signed zero and byte offsets after quoted Unicode. Python regressions exercise
legacy and immutable string filters through the same native parser.

No query executor or numeric kernel optimization is introduced, and no latency or
compiled-plan claim is made. Broader AR7.7, vector/other consumer migration and
ordinary Python defaults remain open. See docs/api/graph-query-literals.md.

Executed: full locked offline native tests, strict all-target Clippy, in-tree
FFI rebuild, 46 focused Python graph/frozen/property/table/numerical checks and
20 documentation examples. Three existing FFI warnings remain. No external
packages installed; ordinary property defaults and broader parser/planner work
remain open.
