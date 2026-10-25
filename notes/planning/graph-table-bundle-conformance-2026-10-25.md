# Graph-table bundle conformance — 2026-10-25

Existing scope: 27.1 / AR7.10 and the table-export boundary of 10.4.
This completes the standalone inline table bundle family; it does not certify
full graph/pool/history portability or compact journal recovery.

## Completed inventory

| Existing API | Result |
| --- | --- |
| save_bundle | Typed v3 frame; full preparation, synced temporary file, atomic replacement and directory sync |
| load_bundle | Verified v3 or legacy v2 reader; exact schema/row counts and logical values |
| validate_bundle / Python verify_bundle | Shared native checksum/format/value/schema checks; no file-existence-only success |
| bundle_info / Python get_bundle_info | Metadata from verified native reader, preserving nested schema/policy and integer counts |
| from_federated_bundles | Existing domain merge now consumes checked typed v3 inputs through the same loader |

One GraphTable-owned private codec reuses the existing property codec and CBOR
dependency. It eliminates CSV writer placeholders and FFI persistence business
logic. Existing public metadata types remain available for legacy compatibility.
V3 has one authoritative file, so there is no partial multi-file publication
window or metadata/data mismatch on replacement. Pooled refs and malformed
inline values reject before touching an existing save. Canonical structural
columns and complete ordered schemas are checked before encoding.

Logical optimized values are normalized, f32/f64 bit patterns retained,
column/row order and IDs unchanged, and policy retained. Source pins/history
are explicitly detached from exported values. Topology validity remains distinct
from encoded integrity. Checksum algorithm names are accurate: xxh3_64 in v3;
legacy DefaultHasher for historical v2 *_sha256 fields. Checksums are not
cryptographic authentication. V2 CSV inference cannot recover already lost types.

The authoritative v3 file takes precedence over legacy sidecars after resaving;
old files are not deleted. Unknown versions reject. There is no claimed v1
placeholder support. Save replacement is a single-writer/caller-coordinated
operation; a directory-sync error after publication may leave a complete file
and requires reconciliation. Whole-table encoding/decoding is currently in memory.

## Evidence

Eight native tests cover logical inline values, exact NaN/zero/vector float bits,
temporal data, empty schemas, corruption/truncation/trailing bytes, malformed
checksummed payloads, unsuccessful encode/replacement, legacy checksums and
migration, and detached captures. Eleven Python cases cover the exposed paths.
Frozen compatibility fixture: `tests/data/graph_table_bundle_v2`.
API examples: `docs/api/graph-table-bundles.md`.

Benchmark: `benches/graph_table_bundles.rs`; three observations at 1k/10k/100k
nodes plus chain edges, scalar and four-f64 vectors. Save includes sync and
replacement; load/verify include checksum/decode and metadata construction.
Input construction is excluded, output destruction included. Warm cache current
costs only; no cold-open, streaming, RSS, matched speedup or FFI claim.
Receipt: `notes/benchmarks/graph-table-bundles-2026-10-25.json`.


### Typed and verified standalone table bundles — 2026-10-25

Closed the existing save/load/validate/info/verify bundle family under
27.1/AR7.10. New v3 CBOR stores ordered logical inline values and policy using
the shared property codec, exact float bits, checked framing/checksum and atomic
synced replacement of one authoritative file. V2 CSV reads now verify historical
checksums, metadata, manifest and schema/counts; resave migrates to v3. FFI
persistence logic was replaced with native delegation and GIL release.

Pooled refs reject rather than becoming strings; source pins/history are detached,
not represented as portable. Legacy CSV cannot recover lost type information.
Topology validation and encoded integrity remain separate. Whole-table memory
cost and post-publication directory-sync reconciliation are documented.

Evidence: `graph-table-bundle-conformance-2026-10-25.md`; frozen v2 fixture;
8 native / 11 Python regression cases. Gate: **1,098 native passed / 51 ignored;
1,830 root Python passed / 1 skipped; 115 executable docs passed; strict
all-target Clippy passed**. Benchmark receipt:
`notes/benchmarks/graph-table-bundles-2026-10-25.json`. No new checklist rows;
191 checked / 77 unchecked. Composite adapters and separate full retained-pool,
history and storage lifecycle requirements remain open.
