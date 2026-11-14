# Embedded core acceptance workflow

The [executable example](../../notes/scripts/core_release_workflow.py) connects
the implemented core surfaces in one small application workflow. It requires
Groggy and the Python standard library; no external graph or numerical library.

From the repository root, using the built in-tree extension:

```sh
PYTHONPATH=python-groggy/python python notes/scripts/core_release_workflow.py
```

The same fixture is exercised by `tests/test_core_release_workflow.py` using both
a retained in-memory graph archive and a journal-backed DurableGraph.

1. Create nodes, external IDs, parallel edges and a loop in one checked batch.
   Use an explicit effective timestamp, an f64 property and a binary payload.
2. Capture an immutable graph. Sort/project its node table and write selected
   properties back in one new batch, preserving row identity.
3. Reconstruct states before and after the update. Reject an invalid mixed
   batch without accepting its valid assignment or advancing history.
4. Export an exact edge selection to a matrix and compute its determinant.
   The excluded parallel edge cannot leak into the numerical input. Analysis
   changes no graph state.
5. Label an explicit checkpoint, then restore the initial state using one new
   tracked delta. Earlier effective-time reads and retained captures stay exact.
6. Save/close, reopen and verify identities, properties, history, checkpoint
   metadata and effective-time reconstruction. A retained view outlives its owner.

The timestamps 100/200/300 are deterministic fixture values in Unix microseconds,
not current application times. This is a supported-surface acceptance example,
not a scale benchmark, complete format-migration proof or service-concurrency
test. Writable owners stay on their creating thread. See the unresolved lifetime
boundary in the [0.6.0 release notes](../../RELEASE_NOTES_0.6.0.md).
