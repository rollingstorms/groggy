# Tracker history authority guards — AR3 compatibility component

The canonical-record inventory found three legacy tracker helpers with insufficient
history authority: generic reverse generation and conflict prediction returned empty
vectors, while merge returned a fake conflict describing an unimplemented method.

These helpers now return GraphResult. Nontrivial inverse/conflict/merge requests
reject explicitly and identify Graph.restore_state / prepare_merge_read as the
owning APIs. Empty inverse/conflict requests and merging an empty source are safe
no-ops. Rejected calls preserve both trackers and their actual physical records.
No implicit restore payload, merge base or caller conflict policy is inferred.

This advances existing AR3; complete compound topology coordination remains
09.4b.2. No checklist row is added or broad coordinator completion claimed.
Native signature migrations are documented in docs/api/change-records.md;
Python surfaces and archive/journal formats are unchanged.

Verification: full native suites 1,384 passed / 56 ignored; focused tracker,
tracked-restore and merge suites passed; strict all-target Clippy and FFI rebuild
passed (three existing FFI warnings). This native-only correction changes no
Python execution path. Checklist remains 208 checked / 60 open.
