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
test. Writable owners stay on their creating thread; final destruction can run
on any thread. See [Python owner lifetimes](python-owner-lifetimes.md).

## Run against a release artifact

```sh
python notes/scripts/verify_release_wheel.py /path/to/wheel-directory --tests ownership --report /tmp/wheel-qualification.json
```

The wheel verifier first checks metadata, all declared exports, typing assets and
both workflow storage paths in a subprocess with site-packages disabled. The
optional `ownership` gate then runs the two workflow tests and 34 ownership cases
against the unpacked wheel. It requires pytest; it installs no wheel. A successful
JSON receipt includes the wheel SHA256. Use a fresh report path for each run.

Use `--tests maintained` for the maintained root suite and curated documentation
examples against that same unpacked artifact. This requires the existing test
dependencies; historical nested surveys are outside that gate. Detailed release
commands and the pending platform gate are in the
[release runbook](release-qualification.md).
