# Exact retained delta interning — 2026-10-27 UTC

Concrete repair discovered while exercising exact first-parent merge deltas,
under existing 11.2/12.2/14.2 and integrity/format requirements. No new checklist row.

## Failure and repair

Legacy Delta hashes use logical AttrValue hashing, which intentionally normalizes
signed zero and NaN payload bits. HistoryForest reused a delta solely by that key.
Two accepted writes from the same before-value to +0/-0 or different NaN payloads
could therefore reconstruct the wrong retained value. The same issue existed when
importing complete archives. A merge test reproduced the loss before publication.

Hashes remain unchanged for compatibility. Matching keys now require exact typed
payload comparison before sharing an Arc. The usual unique-hash path retains its
single representative without a per-delta vector allocation; a separate lazy map
stores only real collision variants. Failure to prove identity never means a match
or a late ordinary-write failure. Full archive import uses the same interning owner.
Independent owner copies carry the collision registry. Explicit GC rebuilds both
indexes from retained unique Arc identities, removing unreferenced variants and
promoting a surviving representative. Statistics count actual retained payloads,
including collision variants. Open immutable records preserve their own Arcs.

Full archives requiring this behavior use document version 5; readers accept 1–5,
and versions 1–4 reject distinct exact payloads sharing a legacy hash. This avoids
silent loss in old readers. Framing and original delta/commit hashes are unchanged.
Ordinary archives keep their prior minimum version; purge/compaction preserve the
required version. Already discarded historical bits cannot be recovered from data
that no longer contains them.

## Evidence

Dedicated native regressions cover f64 signed zero, f32 NaN payloads and f64 vector
NaN payloads; exact duplicate Arc sharing; GC removal/promotion and subsequent
writes; full archive import; and version-5/downgrade rejection. Existing native,
Python, retained storage and merge gates run with this repair. The subsequent
merge exit record contains the completed full-suite counts. No performance budget
claim is made for the rare collision-comparison path. It is not added to ordinary
unique mutations or query paths.
