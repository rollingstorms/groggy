# Graph-table bulk read completion — 2026-10-25

Existing 27.1 / AR7.10 adapter inventory. Closes three existing native operations;
no new APIs, checkbox rows or graph direction/subsetting policy.

- get_nodes/get_edges share BaseTable checked identity scanning. They validate
  ordered columns once, scan IDs once and copy only selected first whole rows.
  Repeated requests deduplicate; unknown requests remain omitted. Duplicate raw
  table IDs retain first-row lookup semantics. Malformed structural IDs/schema
  now reject instead of being accidentally skipped by repeated optional lookups.
- neighbors validates and scans borrowed endpoints once. It preserves the
  undirected external-neighbor set convention, excluding selected members/loops
  and deduplicating parallel routes. No filtered edge tables or full endpoint
  arrays are allocated.

Four native tests cover first-row/unknown/duplicate behavior, an independent
oracle across all 32 subsets of a looped/multiedge fixture, malformed IDs and
empty queries, exact wide IDs and owned-output independence.

Lookup work is O(input rows + requested IDs + selected cells copied), with
membership/result storage proportional to the request and selected outputs.
Neighbor work is O(edges + selected IDs), with membership/result storage rather
than copied edge properties. The benchmark uses 100 requested IDs on chain
fixtures at 1k/10k/100k rows and five observations. Complete validation/output
allocation is timed; fixture construction, assertions and output destruction
are excluded. No matched speedup, RSS, FFI or constant-time claim.
Receipt: `notes/benchmarks/graph-table-bulk-reads-2026-10-25.json`.

API contract: native bulk-read section in `docs/api/entity-table-adapters.md`.
Remaining 27.1 composite subsetting/routing and legacy migration retain their
existing scope. No assumption is recorded as an owner answer to the outstanding
induced-subgraph versus independent-bundle selection question.


### Checked bulk table lookups and neighbor reads — 2026-10-25

Closed the existing native get_nodes/get_edges/neighbors adapter family under
27.1/AR7.10. Bulk identity lookup scans IDs once and copies selected first whole
rows; duplicate requests collapse and unknown IDs remain omitted. Structural
errors reject consistently. Neighbors scan borrowed endpoints once, preserving
the existing external-neighbor set semantics without copied edge tables/arrays.
No public API or graph direction/subsetting policy was added.

Evidence: `graph-table-bulk-reads-2026-10-25.md`; four native regressions, including
all 32 subsets of an independent looped/multiedge oracle. Gate: **1,108 native
passed / 51 ignored; 1,843 root Python passed / 1 skipped; 118 executable docs
passed; strict all-target Clippy passed**. Receipt:
`notes/benchmarks/graph-table-bulk-reads-2026-10-25.json`.
At 100k input rows with 100 requested IDs, current medians are 2.222 ms node
lookup, 2.345 ms edge lookup and 3.486 ms external neighbors; no matched speedup,
RSS or FFI claim. No new rows: **192 checked / 76 unchecked** after 27.2 closure.
Composite subsetting/routing and legacy view migration remain open. The induced
node-subset edge policy question remains unanswered; it was not silently applied.
