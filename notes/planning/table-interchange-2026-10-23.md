# 07.3a.2f — Table interchange and typed grouping

Move plain CSV/JSON value formatting to one table text adapter. CSV/JSON floating
inference uses f64; finite floating vectors serialize as numeric JSON arrays.
Enable serde_json float_roundtrip parsing instead of accepting one-ulp parsing
loss. Existing versioned graph property codecs remain distinct from plain table
interchange; plain formats do not retain every source type tag.

Reject nonfinite scalar/vector JSON values before creating/truncating a target.
No zero substitution. This protects value/serialization rejection, not arbitrary
mid-write I/O failure. Remove manual CSV quoting because csv::Writer already owns
escaping. Preserve quoted multiline and Unicode text.

Group-by now hashes borrowed typed properties and copies actual group keys into
the result. It no longer allocates debug strings or guesses keys by parsing those
strings. Preserve explicit scalar/vector identity and existing unspecified group
order. Python BaseTable grouping and CSV/JSON I/O release the GIL during Rust work.

Native/Python cases cover sub-f32 and large finite values, vectors/empty vectors,
signed zero, nonfinite rejection with an existing target, CSV escaping and typed
keys (including text that resembles a number/null). Release 1k/10k key-grouping
fixtures retain receipts. Remaining scalar/vector/view/step adapters and ordinary
Python float defaults remain open; this does not complete parent 07.3a.

Executed: full locked offline native tests and final targeted CSV/JSON/grouping
regressions, strict all-target Clippy, in-tree FFI rebuild, 19 focused Python
checks and 18 documentation examples. Release grouping receipts at 1k/10k rows:
`benches/typed_table_groups_2026-10-23.json`; scope/limits:
`notes/benchmarks/typed-table-groups-2026-10-23.md`. Three pre-existing FFI warnings
remain; no external packages installed. CSV vector text keeps its prior compact
format; untyped vector import remains text.
