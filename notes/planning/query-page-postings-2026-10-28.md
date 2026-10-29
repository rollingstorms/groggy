# Page immutable property postings directly — 2026-10-28

Measured existing 21.2/32.2 and shared query semantics in AR7.7. The preceding
million-node workflow spends 75.618 µs per exact indexed ten-row page despite
already having the relevant immutable index. No new checklist row or query API.

## Implemented

ID-ordered property-presence and exact-equality queries now borrow their existing
ascending ID posting/run and copy only the requested page. Equality uses the same
binary-search range helper as the ordinary indexed predicate path. Node/edge
filters marshal their own typed variants into one shared posting accessor.
The scan and index executors share an overflow-safe ID-page helper; descending
pages reverse only the selected window, rather than all matching IDs.

Presence paging takes work proportional to the returned page. Exact equality adds
binary-search comparisons over indexed values. Existing immutable index storage
remains allocated, and property comparisons retain their original cost. A missing
index returns to the existing checked error/explicit allow_scan path. Property
ordering, Boolean/range/topology predicates and owned full find results retain the
ordinary executor. No automatic index construction, mutation maintenance, current
owner access, changed filtering domain or new source/provenance type.

Full matching-row scalar validation still precedes property-ordered pages, even
for limit zero or an out-of-range offset. This optimization cannot hide an invalid
ordering domain or missing requested index behind an empty result.

## Verification

Two added native integration cases compare indexed pages to independent captured
column scans for node/edge presence and both equality filter variants. Coverage
includes selected edge/node scope, typed numeric widths, Null, text aliases, NaN,
bytes, absent targets, ascending/descending, zero/full/overflowing page sizes and
usize::MAX offsets. Explicit missing-index and invalid scalar-order checks remain
errors for empty pages. One Python case checks exact large integers/Null, selected
parallel edges, later owner mutation, extreme counts and explicit scan opt-in.

Final gates: **1,444 native passed / 56 ignored**, strict all-target core Clippy,
**2,315 root Python passed / 1 skipped**, **227 executable docs**. Full-ABI CPython
3.9 release extension rebuilt and atomically staged; three pre-existing FFI warnings
remain. Scoped Rust formatting, Python syntax and diff checks pass. The owner’s
separate review/benchmark files remain excluded. Offline locked commands use
`CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target`, full cargo test,
all-target Clippy, FFI release build and the benchmark target.

## Matched repeated-query measurements

`benches/query_page_postings.rs`: same source compiled on baseline `ec24e465` and
the changed core, with separate native release executables. Two processes per
variant reverse order. Each case has five samples per process. Setup is excluded;
fixtures use one repeating i64 bucket plus built-in entity_type, explicit immutable
indexes and no edges. Each page skips one ID and requests up to ten; the 1k exact
match has only nine IDs remaining. Presence uses four iterations/sample; equality
uses 100. Assertions check the full returned arithmetic ID sequence. Raw evidence:
`notes/benchmarks/query-page-postings-2026-10-28.json`.

Forward ID page medians over ten samples per variant:

| Nodes | Presence before → after | Exact equality before → after |
| --- | --- | --- |
| 1,000 | 7.083 → 0.083 µs | 1.392 → 1.275 µs |
| 100,000 | 740.771 → 0.073 µs | 9.509 → 2.065 µs |
| 1,000,000 | 7,606.391 → 0.073 µs | 79.173 → 2.508 µs |

At one million nodes descending presence measures 7,609.885 → 0.073 µs and
exact equality 82.777 → 2.512 µs. The large presence gain removes full-match
materialization; the equality gain avoids copying/sorting/rebuilding 10k matching
IDs for a small page. These are repeated hot native operations, not an FFI budget,
Python latency, index-build/capture budget or arbitrary scalar/predicate workload.

## Remaining priorities

Full current value capture and retained reconstruction/recovery still allocate
whole logical states. The preceding archive receipt explicitly retains its mixed
million-scale RSS result. Legacy connected Subgraph defaults remain live and need
26.1/AR4/AR7.5 migration; compound topology/custom effects remain 09.4b.2. With
capture reuse and bounded ID pages measured, advance those functional ownership/
execution requirements before expanding adjacent performance audits. No new
checkboxes/parent closure: **211 checked / 57 unchecked**.
