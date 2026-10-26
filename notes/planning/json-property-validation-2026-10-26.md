# Reject malformed native JSON before graph mutation — 07.4 / 05.3 component

No new checklist rows; **201 checked / 67 open**. A native regression reproduced
ordinary Graph accepting AttrValue::Json("{") and only discovering invalid syntax
at retained save time. Such a value can leave invalid retained raw data after an
otherwise successful operation. The failure now occurs before mutation.

The portable value codec owns the existing syntax rule (IgnoredAny parsing avoids
bounded-number conversion). Graph's shared property validator applies that rule
alongside semantic-kind validation. Scalar node/edge setters, bulk columns, new
node/edge rows, mixed GraphBatch preparation, atomic property staging and managed
upsert preparation share it. Ordinary Text remains arbitrary text, no JSON-looking
string inference was introduced, and non-JSON intrinsic validation is a type match.
No business logic moved into FFI.

Four native contract tests cover publishing/manual paths, malformed later rows,
node/edge creation, mixed writes, upserts, unchanged values/versions/IDs/reservations,
and valid JSON retaining exact whitespace/large integers/1e999 spelling through
history/archive/journal reopen. Plain Text and JSON null retain their distinct
logical tags. The older export-failure test now expects JSON rejection at mutation
and retains its unsupported-reference encoding/error/destination-preservation case.

## Cost and evidence

Release receipt: `notes/benchmarks/json-property-validation-2026-10-26.json`;
`benches/json_property_validation.rs`, prepared inputs, three samples of 100 calls.
Shared validation only, not complete writes or FFI:

- 1k rows of 46-byte JSON: median **0.121 ms**, text control **0.067 ms**.
- 1k rows of ~16 KiB JSON: **1.786 ms**, text control **0.055 ms**.
- One ~1 MiB JSON property: **0.105 ms**; text/f64 controls ~0.000094 ms.

Text is not equivalent JSON semantics; the extra parse is the explicit correctness
cost. Shared boundaries can validate more than once during preparation. No
constant-time JSON write or 100 ns FFI guarantee is inferred. Valid JSON text is
never parsed into a rewritten bounded-number AST.

Final gate: **1,274 native passed / 52 ignored**, dedicated serial process-death
matrix **1 passed** (six boundaries), **2,004 root Python passed / 1 skipped**,
**160 documentation examples**, strict all-target Clippy and in-tree FFI rebuild.
Broader property/reference/deduplication and raw-retention work keeps its existing
07.4/15/AR7.9 scope; this fix does not silently discard already retained data.
