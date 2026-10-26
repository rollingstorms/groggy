# 14.1 branch lifecycle — exit review

Closed the existing branch lifecycle inventory without adding checklist rows.

| Existing requirement | Verified result |
| --- | --- |
| Creation and switching | References point to accepted heads; duplicate/unknown references and dirty checkout reject. Manual branch metadata does not accept pending edits. Same-head/empty-root selection remains valid. |
| Selected reads | Fixed recorded timestamp selection: Graph and DurableGraph use current first-parent ancestry, with deterministic second-resolution ties. Siblings, merge secondary parents and later retained descendants of an owned older selection do not leak into reads. Explicit state inspection stays available across retained branches. |
| Identity/liveness | Source branches restore exact topology, Null/absence and active properties; ID/name reservations survive alternate/empty selections and allocations never rewind. |
| Immutable referents | Existing views, typed values and frozen source groups remain stable across branch deactivation/switching, owner destruction, reopen and backup. |
| Durable policy | Expected-state, duplicate/unknown references and target schema validate before append. Selection/references recover, including retained frozen sources of inactive live entities. |
| Failure behavior | Eight branch creation/checkout combinations cover journal failures before/after header, after payload and after sync: bytes/selection unchanged, retry accepted, reopen exact. |

Native bare-history timestamp APIs retain their explicit global scope; new
on_lineage selectors are available when supplying a head. Graph facade selection
uses cached first-parent ancestry. Empty lineages or bounds before their first
recorded state reject; effective reconstruction can still return the empty graph.
Python Graph.reset_hard now marshals the existing native operation for a complete
dirty-workflow example, retaining allocation marks/backing values.

Verification: **1,210 native tests passed / 51 ignored**, **1,953 Python tests
passed / 1 skipped**, **148 executable documentation examples**; strict all-target
Clippy, in-tree FFI build (three unrelated existing warnings), Rust formatting,
Python/stub syntax and diff checks.

**200 checked / 68 open**, no new rows. Merge planning remains 14.2. Destructive
branch pruning/history collection, raw pool/identity reclamation and persistent
compaction remain 15/10.4; this verification does not close them.
