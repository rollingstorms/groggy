# 11.2 property history — exit review

Completed under the existing item; no new checklist rows.

- Logical recorded changes carry optional old/new typed values. Unset is absence,
  not Null, and removes only the current column mapping.
- Scalar/bulk native Graph and publishing writers plus Python GraphBatch/GraphWriter
  expose node/edge unsets. Mixed operation validation rejects duplicate/conflicting,
  inactive and managed-identity targets before allocation. Meta source references
  remain required. Caller/durable schema checks see absence in the proposed state.
- Per-change/default effective clocks, recorded cutoffs and accepted-order ties
  apply to unset events. Already-absent unsets remain events so backdated writes
  are meaningful. Absence persists until a later effective assignment.
- Manual scalar/bulk/unset sequences coalesce first-old/final-new in operation
  order. Fixed scalar-after-bulk intermediate-value retention; ordinary publications
  remain separate events. Pending recording appends in amortized O(1), then uses
  keyed coalescing at publication; it retains pending operations until publication.
- Recorded/effective snapshots, branch switching and temporal indexes preserve
  null/absence. Window counts include unsets; numeric reductions reject missing
  and nonnumeric events. Python typed history represents an unset by bare None,
  an explicit Null by an AttrValue whose value is None.
- All native property representations, including compressed/compact values, raw
  float bits, reference arrays, temporal values and frozen groups, remain retained.
  Durable typed inline/group histories survive close/reopen and full backup.
- Existing assignment-only delta hashes remain byte-compatible. Optional journal
  unset columns are omitted when empty; older records remain readable. Binaries
  predating unset support cannot read records carrying those new fields.

## Verification

- `cargo test --offline --locked --quiet`: **1,190 passed / 51 ignored**.
- Focused native property-unset suite: **6 passed**, plus temporal reducer checks.
- In-tree Python suite: **1,925 passed / 1 skipped**; **142 docs examples**.
- Strict all-target Clippy; in-tree FFI build (three existing unrelated warnings),
  scoped Rust formatting, Python/stub syntax and diff checks.

Native source compatibility change: ChangeSet/Delta new property values, index
strategy new indices and property-history entries now use Option. Python existing
assignment entries remain typed wrappers. See `docs/api/property-history.md`.

## Remaining work stays in its original scope

12.2 owns full configuration/owned snapshot materialization and bounded switch
allocation. 10.3/10.4 own arbitrary backing-pool persistence/import and recovery
checkpoints/compaction. 13.3 owns reversal/rewiring; 14.2 owns merge planning;
15/18 own reclamation and index maintenance. No byte-level value-difference codec
or production durability policy is inferred from this item.
